"""
kshiraj/aiml_client/client.py

Client adapter and integration layer for the AI/ML component.

Responsibilities:
  1. `adapt_retrieved_standards`: Adapts CandidateStandard / RetrievalResult objects from
     kshiraj.knowledge.retrieval_service into a deduplicated list[Standard] suitable
     for AimlRequest.retrieved_standards.
  2. `AimlClient`: Async client for sending AimlRequest payloads to either:
       - a mock execution engine (for local development and offline testing)
       - an HTTP service endpoint (when `settings.aiml_service_url` is configured)
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Union

import httpx
from pydantic import ValidationError

from shared.config import settings
from shared.contracts import AimlFinding, AimlRequest, AimlResponse
from shared.models import Standard, Verdict
from shared.utils import get_logger

from kshiraj.aiml_client.schemas import (
    AimlClientError,
    AimlResponseError,
    AimlTimeoutError,
)
from kshiraj.knowledge.retrieval_service import CandidateStandard, RetrievalResult

logger = get_logger(__name__)


# ===========================================================================
# Retrieval output adapter
# ===========================================================================

def adapt_retrieved_standards(
    retrieved: Union[RetrievalResult, List[CandidateStandard], List[Standard], None],
) -> List[Standard]:
    """
    Transform candidate retrieval results into a deduplicated, deterministically-ordered
    list of Standard objects suitable for `AimlRequest.retrieved_standards`.

    Requirements fulfilled:
      - Extracts `CandidateStandard.standard`.
      - Preserves `relevance_score`.
      - Deduplicates by `Standard.id`.
      - Preserves deterministic ordering.
      - Populates `Standard.text_excerpt` only if a defensible source exists
        (existing text_excerpt, candidate evidence excerpts, or scope). Does not invent content.
      - Isolated from HTTP transport.
    """
    if retrieved is None:
        return []

    raw_items: List[Union[CandidateStandard, Standard]]
    if isinstance(retrieved, RetrievalResult):
        raw_items = list(retrieved.candidates)
    elif isinstance(retrieved, list):
        raw_items = list(retrieved)
    else:
        return []

    adapted: List[Standard] = []
    seen_ids: set[str] = set()

    for item in raw_items:
        if isinstance(item, CandidateStandard):
            std = item.standard
            score = std.relevance_score if std.relevance_score is not None else item.score
            evidence_excerpts = [
                ev.excerpt.strip()
                for ev in item.evidence
                if ev.excerpt and ev.excerpt.strip()
            ]
        elif isinstance(item, Standard):
            std = item
            score = std.relevance_score
            evidence_excerpts = []
        else:
            continue

        if std.id in seen_ids:
            continue
        seen_ids.add(std.id)

        # Determine defensible text_excerpt without inventing data
        text_excerpt: Optional[str] = None
        if std.text_excerpt and std.text_excerpt.strip():
            text_excerpt = std.text_excerpt.strip()
        elif evidence_excerpts:
            text_excerpt = "\n---\n".join(evidence_excerpts)
        elif std.scope and std.scope.strip():
            text_excerpt = std.scope.strip()

        updated_std = std.model_copy(
            update={
                "relevance_score": score,
                "text_excerpt": text_excerpt,
            }
        )
        adapted.append(updated_std)

    return adapted


def _is_base(designation: str) -> str:
    """
    Base IS number for comparison: whitespace-free, casefolded, and cut at the
    first part/year separator — "IS 1554 : Part 1" and "IS 1554:1988" both give
    "is1554", while "IS 15544" stays distinct (a substring test matched it).
    """
    return re.split(r"[:(]", re.sub(r"\s+", "", designation).casefold())[0]


# ===========================================================================
# AI/ML Client
# ===========================================================================

class AimlClient:
    """
    Async client for sending AimlRequest objects to the AI/ML engine.

    Supports both:
      - Mock execution (default when `settings.aiml_service_url` is empty)
      - HTTP execution (when `settings.aiml_service_url` is configured)
    """

    def __init__(
        self,
        service_url: Optional[str] = None,
        timeout: float = 30.0,
        force_mock: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        service_url:
            Optional override for AI/ML HTTP service URL. Defaults to `settings.aiml_service_url`.
        timeout:
            Request timeout in seconds for HTTP requests.
        force_mock:
            If True, forces mock execution even if service_url is set.
        """
        configured_url = service_url if service_url is not None else settings.aiml_service_url
        self.service_url = configured_url.strip() if configured_url else ""
        self.timeout = timeout
        self.force_mock = force_mock

    @property
    def is_mock(self) -> bool:
        """Return True if running in mock execution mode."""
        return self.force_mock or not bool(self.service_url)

    async def run_analysis(self, request: AimlRequest) -> AimlResponse:
        """
        Execute AI/ML analysis for the provided AimlRequest.

        Parameters
        ----------
        request:
            The AimlRequest payload.

        Returns
        -------
        AimlResponse
            The validated AI/ML response.

        Raises
        ------
        AimlClientError
            If request is invalid.
        AimlTimeoutError
            If HTTP request times out.
        AimlResponseError
            If HTTP request fails or response payload cannot be parsed as AimlResponse.
        """
        if not isinstance(request, AimlRequest):
            raise AimlClientError(
                f"Invalid request object type: {type(request)}. Expected AimlRequest.",
                code="INVALID_REQUEST",
            )

        if self.is_mock:
            return self._run_mock_analysis(request)
        else:
            return await self._run_http_analysis(request)

    # ------------------------------------------------------------------
    # Mock execution engine
    # ------------------------------------------------------------------

    def _run_mock_analysis(self, request: AimlRequest) -> AimlResponse:
        """
        Generate a deterministic, structurally valid AimlResponse for testing and development.
        """
        logger.info(
            "Executing MOCK AI/ML analysis for analysis_id=%s (requirements=%d, standards=%d)",
            request.analysis_id,
            len(request.requirements),
            len(request.retrieved_standards),
        )

        findings: List[AimlFinding] = []

        for idx, req in enumerate(request.requirements):
            # Only this requirement's own candidates — never the pooled list.
            candidates = request.requirement_candidates.get(req.id, [])
            matched_std_ids: List[str] = []

            # The mock can only recognise an explicit citation. Anything else is
            # left unmapped rather than assigned a candidate it cannot justify.
            if req.is_reference:
                ref_base = _is_base(req.is_reference)
                matched_std_ids = [
                    std.id for std in candidates if _is_base(std.is_number) == ref_base
                ]

            verdict_val = (
                Verdict.JUSTIFIED.value
                if matched_std_ids
                else Verdict.REQUIRES_HUMAN_VERIFICATION.value
            )
            reason_str = (
                f"Mock analysis: requirement evaluated against {len(candidates)} candidate standard(s)."
            )
            rec_action = (
                "Verify standard compliance details."
                if matched_std_ids
                else "Manually verify specification against applicable standards."
            )

            finding = AimlFinding(
                finding_id=f"mock-finding-{idx + 1}-{req.id[:8]}",
                requirement_id=req.id,
                verdict=verdict_val,
                reason=reason_str,
                recommended_action=rec_action,
                applicable_standard_ids=matched_std_ids,
                evidence_ids=[],
                confidence=0.90 if matched_std_ids else 0.65,
            )
            findings.append(finding)

        return AimlResponse(
            analysis_id=request.analysis_id,
            findings=findings,
            extraction_metadata={
                "execution_mode": "mock",
                "requirements_count": len(request.requirements),
                "retrieved_standards_count": len(request.retrieved_standards),
            },
        )

    # ------------------------------------------------------------------
    # HTTP execution transport
    # ------------------------------------------------------------------

    async def _run_http_analysis(self, request: AimlRequest) -> AimlResponse:
        """
        Send AimlRequest via async HTTP POST to configured AI/ML service URL.
        """
        logger.info(
            "Executing HTTP AI/ML analysis for analysis_id=%s at service_url=%s",
            request.analysis_id,
            self.service_url,
        )

        payload_json = request.model_dump_json()
        headers = {"Content-Type": "application/json"}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.service_url,
                    content=payload_json,
                    headers=headers,
                )
                response.raise_for_status()
                raw_json = response.json()
                return AimlResponse.model_validate(raw_json)

        except httpx.TimeoutException as exc:
            logger.error("HTTP request to AI/ML service timed out (%s s): %s", self.timeout, exc)
            raise AimlTimeoutError(
                f"AI/ML service request timed out after {self.timeout}s."
            ) from exc

        except httpx.HTTPStatusError as exc:
            logger.error(
                "HTTP error from AI/ML service status=%s: %s",
                exc.response.status_code,
                exc.response.text,
            )
            raise AimlResponseError(
                f"AI/ML service returned HTTP error status {exc.response.status_code}: {exc.response.text}",
                code=f"HTTP_{exc.response.status_code}",
            ) from exc

        except httpx.RequestError as exc:
            logger.error(
                "Network connection error calling AI/ML service at %s: %s",
                self.service_url,
                exc,
            )
            raise AimlResponseError(
                f"Failed to connect to AI/ML service at {self.service_url}: {exc}",
                code="CONNECTION_ERROR",
            ) from exc

        except (ValueError, KeyError, ValidationError) as exc:
            logger.error("Invalid/malformed response payload from AI/ML service: %s", exc)
            raise AimlResponseError(
                f"Invalid response payload from AI/ML service: {exc}",
                code="INVALID_RESPONSE_PAYLOAD",
            ) from exc
