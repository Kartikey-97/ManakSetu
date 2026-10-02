"""
kshiraj/knowledge/retrieval_service.py

Candidate-retrieval service for BIS Indian Standards.

Role in the pipeline
--------------------
This service sits between the raw stores (StandardsStore / EvidenceStore)
and the AI/ML analysis layer.  Its single responsibility is: given a
procurement requirement text, return a ranked list of candidate Standard
objects that the AI/ML layer should reason over.

It does NOT:
  - call BIS, CPPP, or any external URL
  - call an LLM
  - generate embeddings or vector representations
  - make compliance decisions
  - produce final Findings
  - infer whether a standard is legally applicable

Algorithm (lexical, MVP)
-------------------------
The query text is tokenised into normalised tokens (lowercase, punctuation
stripped).  Each candidate Standard from the store is scored by counting
weighted token hits across key text fields:

  Field                  Weight
  ─────────────────────  ──────
  is_number (exact)       10.0   — "IS 10322" match outweighs all others
  designation (exact)     10.0   — full canonical form match
  is_number (token)        5.0   — token appears inside the IS-number string
  title (token)            3.0   — title word hit
  scope (token)            1.5   — scope word hit
  technical_committee      1.0   — committee mention
  division_council         1.0   — council mention

Candidates with a score of 0 are excluded.
Candidates are sorted descending by score, then ascending by is_number for
determinism when scores are equal.

The service is stateless beyond its store references; it is safe to call
concurrently from multiple threads.  It does not modify any stored object.
The Standard objects it returns have their ``relevance_score`` field
populated (model_copy), leaving the stored objects untouched.

Evidence association (optional)
---------------------------------
When ``include_evidence=True`` in the query, evidence records whose ``url``
matches a standard's ``source_url`` are attached to the corresponding
CandidateStandard.  This is a conservative URL-equality heuristic:
  - deterministic and produces no false positives when URLs are set correctly
  - produces no results when source_url is None (correct — no spurious links)
The Evidence model has no standard_id foreign key in the current shared
models, so URL matching is the only available non-heuristic link.

Extension points
----------------
To add semantic retrieval later, replace or subclass RetrievalService and
override _score_standard().  The RetrievalQuery / CandidateStandard /
RetrievalResult shapes are designed to be stable across algorithm changes.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import List, Optional

from shared.models import Evidence, Standard, StandardStatus

from kshiraj.knowledge.evidence_store import EvidenceStore
from kshiraj.knowledge.standards_store import StandardsStore


# ---------------------------------------------------------------------------
# Public value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalQuery:
    """Input specification for a standards retrieval request.

    Parameters
    ----------
    query_text:
        The procurement requirement text to match against.
        May be the raw extracted requirement, a normalised version, or a
        direct IS citation (e.g. ``"IS 10322"``).
        Empty strings are accepted and produce zero results.
    status_filter:
        If ``None``, all status values are considered.
        If an empty list, no candidates are returned (explicit empty filter).
        If a non-empty list, only standards whose status is in the list are
        returned.
    include_evidence:
        If ``True``, each CandidateStandard in the result will include any
        Evidence records associated with that standard via URL matching.
        Defaults to ``False`` for performance.
    top_k:
        If set, the result is truncated to the top-*k* candidates by score.
        ``None`` means no truncation.
    """

    query_text: str
    status_filter: Optional[List[StandardStatus]] = None
    include_evidence: bool = False
    top_k: Optional[int] = None


@dataclass
class CandidateStandard:
    """One candidate result from a retrieval query.

    Attributes
    ----------
    standard:
        The Standard object (model_copy with ``relevance_score`` populated).
        The original stored object is not mutated.
    score:
        Unnormalised lexical match score.  Higher is a stronger match.
        Populated by the retrieval algorithm; ``None`` means no match was
        scored (should not normally appear in results).
    matched_terms:
        Deduplicated list of query tokens that contributed to the score.
        Useful for debugging and future highlight rendering.
    evidence:
        Evidence records associated with this standard.  Populated only when
        ``RetrievalQuery.include_evidence`` is ``True``.
    """

    standard: Standard
    score: float
    matched_terms: List[str] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)


@dataclass
class RetrievalResult:
    """The output of a single retrieval operation.

    Attributes
    ----------
    query:
        The original query, echoed back for traceability.
    candidates:
        Ranked list of candidate standards.  Ordered by score descending;
        ties broken by is_number ascending (lexicographic).
    total_candidates:
        Total number of candidates in the result (equals ``len(candidates)``
        after any top_k truncation — ``total_candidates`` reflects the
        pre-truncation count so callers can detect truncation).
    """

    query: RetrievalQuery
    candidates: List[CandidateStandard]
    total_candidates: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Pre-compiled punctuation removal table.
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

# Minimum token length to consider for matching (filters out "a", ":", etc.)
_MIN_TOKEN_LEN = 2

# Field weights for the lexical scoring formula.
_W_IS_NUMBER_EXACT = 10.0
_W_DESIGNATION_EXACT = 10.0
_W_IS_NUMBER_TOKEN = 5.0
_W_TITLE_TOKEN = 3.0
_W_SCOPE_TOKEN = 1.5
_W_KEYWORD_TOKEN = 2.0
_W_COMMITTEE_TOKEN = 1.0
_W_COUNCIL_TOKEN = 1.0


def _tokenize(text: str) -> set[str]:
    """Return a set of normalised tokens from *text*.

    Lowercases, removes punctuation, splits on whitespace, and discards
    tokens shorter than ``_MIN_TOKEN_LEN``.
    """
    cleaned = text.lower().translate(_PUNCT_TABLE)
    return {t for t in cleaned.split() if len(t) >= _MIN_TOKEN_LEN}


def _text_tokens(value: str | None) -> set[str]:
    """Return tokens for an optional string field; empty set if ``None``."""
    return _tokenize(value) if value else set()


def _normalize_query(raw: str) -> str:
    """Strip and casefold the query for exact-match comparisons."""
    return raw.strip().casefold()


# ---------------------------------------------------------------------------
# Retrieval service
# ---------------------------------------------------------------------------


class RetrievalService:
    """Lexical retrieval service that scores Standard candidates against a query.

    The service is constructed with references to the two stores.  It does not
    hold any internal mutable state beyond those references, making it safe to
    share across request handlers.

    Parameters
    ----------
    standards_store:
        The populated :class:`~kshiraj.knowledge.standards_store.StandardsStore`.
    evidence_store:
        The populated :class:`~kshiraj.knowledge.evidence_store.EvidenceStore`.
    """

    def __init__(
        self,
        standards_store: StandardsStore,
        evidence_store: EvidenceStore,
    ) -> None:
        self._standards = standards_store
        self._evidence = evidence_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_standards(self, query: RetrievalQuery) -> RetrievalResult:
        """Retrieve and rank candidate standards matching *query*.

        Parameters
        ----------
        query:
            A :class:`RetrievalQuery` describing what to search for and how
            to filter the results.

        Returns
        -------
        RetrievalResult
            Ranked candidates, with evidence attached if requested.

        Notes
        -----
        - An empty ``query_text`` always produces zero candidates.
        - An empty ``status_filter`` list (not ``None``) always produces zero
          candidates; this is intentional and allows callers to explicitly
          request nothing.
        - The returned Standard objects are copies (``model_copy``) with
          ``relevance_score`` set; the originals in the store are untouched.
        """
        # Fast-exit: empty query text → no results
        if not query.query_text or not query.query_text.strip():
            return RetrievalResult(query=query, candidates=[], total_candidates=0)

        # Fast-exit: explicit empty status filter → no results
        if query.status_filter is not None and len(query.status_filter) == 0:
            return RetrievalResult(query=query, candidates=[], total_candidates=0)

        # 1. Fetch all standards as a snapshot (store handles thread safety).
        all_standards: list[Standard] = self._standards.list_all()

        # 2. Apply status filter if specified.
        if query.status_filter is not None:
            allowed = set(query.status_filter)
            all_standards = [s for s in all_standards if s.status in allowed]

        # 3. Score each candidate.
        query_tokens = _tokenize(query.query_text)
        query_norm = _normalize_query(query.query_text)

        scored: list[CandidateStandard] = []
        seen_ids: set[str] = set()  # guard against any duplicate IDs from the store

        for std in all_standards:
            if std.id in seen_ids:
                continue
            seen_ids.add(std.id)

            score, matched = self._score_standard(std, query_tokens, query_norm)
            if score <= 0.0:
                continue

            # Return a copy with relevance_score populated; never mutate the stored object.
            scored_std = std.model_copy(update={"relevance_score": score})
            scored.append(
                CandidateStandard(
                    standard=scored_std,
                    score=score,
                    matched_terms=matched,
                )
            )

        # 4. Sort: score descending, then is_number ascending for tie-breaking.
        scored.sort(key=lambda c: (-c.score, c.standard.is_number))

        total_before_truncation = len(scored)

        # 5. Apply top_k.
        if query.top_k is not None and query.top_k > 0:
            scored = scored[: query.top_k]

        # 6. Optionally hydrate evidence.
        if query.include_evidence:
            self._attach_evidence(scored)

        return RetrievalResult(
            query=query,
            candidates=scored,
            total_candidates=total_before_truncation,
        )

    # ------------------------------------------------------------------
    # Internal scoring
    # ------------------------------------------------------------------

    def _score_standard(
        self,
        std: Standard,
        query_tokens: set[str],
        query_norm: str,
    ) -> tuple[float, list[str]]:
        """Compute a lexical relevance score for one Standard.

        Parameters
        ----------
        std:
            The candidate standard to score.
        query_tokens:
            Normalised token set derived from the query text.
        query_norm:
            The query string after strip + casefold, for exact-match checks.

        Returns
        -------
        tuple[float, list[str]]
            ``(score, matched_terms)`` where matched_terms is the deduplicated
            list of tokens (or phrases) that contributed to the score.
        """
        if not query_tokens:
            return 0.0, []

        score = 0.0
        matched: set[str] = set()

        # --- Exact IS-number match ---
        is_norm = std.is_number.strip().casefold()
        desig_norm = std.designation.strip().casefold()

        if query_norm == is_norm or is_norm in query_norm:
            score += _W_IS_NUMBER_EXACT
            matched.add(std.is_number)

        # --- Exact designation match ---
        if query_norm == desig_norm or desig_norm in query_norm:
            score += _W_DESIGNATION_EXACT
            matched.add(std.designation)

        # --- Token-level IS-number match ---
        is_tokens = _tokenize(std.is_number)
        common = query_tokens & is_tokens
        if common:
            score += _W_IS_NUMBER_TOKEN * len(common)
            matched.update(common)

        # --- Title token overlap ---
        title_tokens = _text_tokens(std.title)
        common = query_tokens & title_tokens
        if common:
            score += _W_TITLE_TOKEN * len(common)
            matched.update(common)

        # --- Scope token overlap ---
        scope_tokens = _text_tokens(std.scope)
        common = query_tokens & scope_tokens
        if common:
            score += _W_SCOPE_TOKEN * len(common)
            matched.update(common)

        # --- Keyword token overlap ---
        if hasattr(std, 'keywords') and std.keywords:
            kw_tokens = set()
            for kw in std.keywords:
                kw_tokens.update(_text_tokens(kw))
            common = query_tokens & kw_tokens
            if common:
                score += _W_KEYWORD_TOKEN * len(common)
                matched.update(common)

        # --- Technical committee token overlap ---
        comm_tokens = _text_tokens(std.technical_committee)
        common = query_tokens & comm_tokens
        if common:
            score += _W_COMMITTEE_TOKEN * len(common)
            matched.update(common)

        # --- Division council token overlap ---
        council_tokens = _text_tokens(std.division_council)
        common = query_tokens & council_tokens
        if common:
            score += _W_COUNCIL_TOKEN * len(common)
            matched.update(common)

        return score, sorted(matched)

    # ------------------------------------------------------------------
    # Evidence hydration
    # ------------------------------------------------------------------

    def _attach_evidence(self, candidates: list[CandidateStandard]) -> None:
        """Populate the ``evidence`` field of each candidate.

        Uses URL equality: an Evidence record is associated with a Standard
        if ``evidence.url == standard.source_url`` (after stripping whitespace).

        This is conservative and deterministic.  Standards with ``source_url=None``
        receive no evidence (correct — no spurious links).
        The Evidence model carries no ``standard_id`` FK in the current shared
        models; URL matching is the available non-heuristic link.
        """
        all_evidence: list[Evidence] = self._evidence.list_all()

        # Build URL → [Evidence] map once for efficiency.
        url_to_evidence: dict[str, list[Evidence]] = {}
        for ev in all_evidence:
            if ev.url is not None:
                key = ev.url.strip()
                url_to_evidence.setdefault(key, []).append(ev)

        for candidate in candidates:
            src_url = candidate.standard.source_url
            if src_url is not None:
                candidate.evidence = url_to_evidence.get(src_url.strip(), [])
            # else: evidence stays as empty list (default)
