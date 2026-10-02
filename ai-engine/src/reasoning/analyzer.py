"""
ai-engine/src/reasoning/analyzer.py

Orchestrates per-requirement analysis:
  1. ML classification of requirement type (or keyword fallback)
  2. Deterministic currentness check (superseded/outdated standards)
  3. LLM reasoning with richer context (structured requirements + standard summaries)

Fixes:
  - extraction_metadata now correctly reflects the actual reasoning mode
  - LLM prompt now receives structured technical parameters, not just raw text
"""

import logging
import os
import re
import uuid

from src.reasoning.providers.ml import MLReasoner
from src.reasoning.providers.mock import MockReasoner
from src.reasoning.providers.gemini import GeminiReasoner

logger = logging.getLogger(__name__)


def _classify_requirement(text: str) -> str:
    """
    Simple keyword-based requirement classification.
    Replaces the ML model dependency (which requires a trained .joblib file).
    """
    text_lower = text.lower()
    if any(k in text_lower for k in ["watt", " w ", "power", "efficacy", "lm/w", "lumens"]):
        return "power_performance"
    if any(k in text_lower for k in ["volt", " v ", "vac", "current", "hz", "frequency"]):
        return "electrical"
    if any(k in text_lower for k in ["ip ", "ingress", "protection", "weatherproof", "outdoor"]):
        return "environmental_protection"
    if any(k in text_lower for k in ["bis", "isi", "crs", "qco", "certification", "mandatory", "mark"]):
        return "certification"
    if any(k in text_lower for k in ["is ", "iec ", "iso ", "standard", "compliance"]):
        return "standards_reference"
    if any(k in text_lower for k in ["surge", "spd", "lightning", "transient"]):
        return "protection"
    if any(k in text_lower for k in ["thd", "harmonic", "power factor", "pf "]):
        return "power_quality"
    if any(k in text_lower for k in ["cct", "color temp", "colour temp", "kelvin", " k "]):
        return "optical"
    return "general"


def _is_base(designation: str) -> str:
    """
    Base IS number for comparison: whitespace-free, casefolded, cut at the first
    part/year separator. "IS 1554 : Part 1" and "IS 1554:1988" give "is1554";
    "IS 15544" stays distinct, which the old substring test did not.
    """
    return re.split(r"[:(]", re.sub(r"\s+", "", designation).casefold())[0]


def _ids_for_is_number(designation: str, candidates: list) -> list:
    base = _is_base(designation)
    return [s.id for s in candidates if _is_base(s.is_number) == base]


def _select_candidate_ids(res: dict, verdict: str, req, candidates: list) -> list:
    """
    Turn a provider result into applicable standard IDs, drawn only from this
    requirement's candidates. Selections outside the candidate set are dropped,
    never replaced, so an invalid selection cannot widen the result.

    Precedence, first one the provider expressed wins:
      1. applicable_standard_ids — authoritative, including an explicit [].
      2. matched_is_number       — a string names a standard; null means none.
      3. neither (e.g. the mock) — a "justified" verdict is taken to refer to the
         requirement's own citation, if that citation is among its candidates.
    An explicit empty answer from 1 or 2 is never overridden by the citation.
    """
    candidate_ids = {s.id for s in candidates}

    if "applicable_standard_ids" in res:
        proposed = res.get("applicable_standard_ids") or []
        if not isinstance(proposed, list):
            proposed = []
        selected = []
        for sid in proposed:
            if sid in candidate_ids and sid not in selected:
                selected.append(sid)
        rejected = [sid for sid in proposed if sid not in candidate_ids]
        if rejected:
            logger.warning(
                "Dropped %d non-candidate standard ID(s) for requirement_id=%s: %s",
                len(rejected), req.id, rejected,
            )
        return selected

    if "matched_is_number" in res:
        matched_is = res.get("matched_is_number")
        if isinstance(matched_is, str) and matched_is.strip().lower() not in ("", "null", "none"):
            selected = _ids_for_is_number(matched_is, candidates)
            if not selected:
                logger.warning(
                    "matched_is_number=%r for requirement_id=%s is not among its "
                    "%d candidate(s) — no standard selected.",
                    matched_is, req.id, len(candidates),
                )
            return selected
        return []

    if verdict == "justified" and req.is_reference:
        return _ids_for_is_number(req.is_reference, candidates)
    return []


class Analyzer:
    def __init__(self):
        # AI_MODE selects the reasoning provider:
        #   "ml"     — the trained applicability model, offline and deterministic
        #   "mock"   — keyword heuristics, for tests and local development
        #   anything else (default) — Gemini
        #
        # MLReasoner was imported here but never constructed, so the applicability
        # model could not be reached through /analyze at all no matter how the
        # environment was configured. It is also the right thing to fall back to
        # when Gemini cannot start: the model is real and offline, whereas the
        # mock is keyword matching dressed as analysis, and a demo that silently
        # degrades to it looks like a working system when it is not.
        mode = os.getenv("AI_MODE", "ml")
        if mode == "mock":
            self.provider = MockReasoner()
            self._mode = "mock"
        elif mode == "gemini":
            try:
                self.provider = GeminiReasoner()
                self._mode = "gemini"
            except Exception as exc:
                logger.warning(
                    "Could not initialise GeminiReasoner (%s) — falling back to "
                    "the applicability model.", exc,
                )
                try:
                    self.provider = MLReasoner()
                    self._mode = "ml_fallback"
                except Exception as ml_exc:
                    logger.warning(
                        "The applicability model is unavailable too (%s) — "
                        "falling back to mock.", ml_exc,
                    )
                    self.provider = MockReasoner()
                    self._mode = "mock_fallback"
        else:
            # Default to ML
            try:
                self.provider = MLReasoner()
                self._mode = "ml"
            except Exception as exc:
                logger.warning(
                    "The applicability model is unavailable (%s) — "
                    "falling back to mock.", exc,
                )
                self.provider = MockReasoner()
                self._mode = "mock_fallback"

    def process(self, request) -> dict:
        import concurrent.futures

        findings = []

        def process_req(req):
            # 1. Classify requirement type
            req_type = _classify_requirement(req.text)

            # Only the standards retrieved for THIS requirement, with its own
            # scores. request.retrieved_standards pools every requirement's
            # candidates and must never be reasoned over for a single one.
            candidates = list(request.requirement_candidates.get(req.id, []))

            # 2. ML/LLM reasoning with richer context
            res = self.provider.analyze(
                req_text=req.text,
                req_type=req_type,
                standards=candidates,
                is_reference=req.is_reference,
                cited_year=req.cited_year,
            )

            verdict = res.get("verdict", "requires_human_verification")
            reason = res.get("reason", "")
            action = res.get("action", "Manually verify specification against standards.")
            raw_conf = float(res.get("confidence", 0.5))
            conf = 0.70 + (raw_conf * 0.29)

            matched_ids = _select_candidate_ids(res, verdict, req, candidates)

            # 3. Deterministic currentness check on the SELECTED standard(s)
            outdated_stds = [
                s for s in candidates
                if s.id in matched_ids
                and s.status.lower() in ("superseded", "withdrawn", "cancelled")
            ]

            if outdated_stds:
                # Deterministic override — superseded standard is always flagged
                names = ", ".join(s.is_number for s in outdated_stds[:2])
                verdict = "outdated_reference"
                reason = (
                    f"The selected standard(s) {names} are marked as superseded or withdrawn "
                    "in the BIS metadata. This specification should reference the current edition."
                )
                action = "Update the tender specification to cite the current edition of the standard."
                raw_conf = 0.95
                conf = 0.95

            return {
                "finding_id": str(uuid.uuid4()),
                "requirement_id": req.id,
                "verdict": verdict,
                "reason": reason,
                "recommended_action": action,
                "applicable_standard_ids": matched_ids,
                "evidence_ids": [],
                "confidence": conf,
                "raw_confidence": raw_conf,
            }

        # Process all requirements concurrently (up to 5 at a time) to prevent massive latency
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            findings = list(executor.map(process_req, request.requirements))

        return {
            "analysis_id": request.analysis_id,
            "findings": findings,
            "extraction_metadata": {
                "reasoning_mode": self._mode,
                "requirements_analysed": len(findings),
            },
        }
