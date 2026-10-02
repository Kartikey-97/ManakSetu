"""
kartikey/analysis/certification_engine.py

Standalone QCO (Quality Control Order) applicability engine.

kartikey/analysis/compliance.py answers a narrower question: "is this
already-matched Standard object QCO-notified?" — it reads qco_notified /
qco_issuing_ministry etc. straight off a Standard, and those fields only
exist because kartikey/data/merger.py already merged the QCO database onto
the BIS catalogue ahead of time. That means a QCO can only ever be reported
if its IS number was already retrieved as a candidate standard.

This module answers the broader question a procurement officer actually
needs answered up front: "given what this tender is procuring, which QCOs
in our knowledge base apply at all?" — including ones the tender never
cited and retrieval never surfaced, because the officer omitted the
standard entirely rather than citing an outdated one. That is a materially
different failure mode from "cited IS 1554 for XLPE cable" (wrong scope,
compliance.py catches it) — this is "never cited a cable standard at all."

It reads kartikey/data/mock_qco_database.json directly rather than the
reconciled BIS+QCO catalogue, so it keeps working immediately after the QCO
database is hand-edited, without waiting for the ETL merge/export pipeline
(kartikey/data/merger.py -> ml_exporter.py) to be re-run.

Two independent ways a QCO can surface as applicable
------------------------------------------------------------------
1. Exact IS-number match — matched_is_numbers (IS numbers already cited by
   the tender and/or retrieved by the knowledge base) is compared against
   each QCO record's is_number using the same fuzzy, punctuation-insensitive
   normalization kartikey/data/merger.py uses, so "IS 1391 (Part 2)" and
   "is 1391(part 2)" resolve to the same QCO.

2. Product-keyword match — each QCO record's products_covered list is
   matched, bag-of-words and case-insensitively, against the free text
   describing the product being procured (product_profile): a products_covered
   entry matches if all of its significant words appear somewhere in the
   profile text, in any order, with any words in between (so "ergonomic
   chairs" matches "ergonomic revolving office chairs" — real tender
   language is rarely as tidy as the database's own phrasing). This is what
   lets the engine say "you didn't cite IS 1391, but you're buying split
   ACs, and that IS is QCO-mandatory" instead of staying silent because
   nothing upstream ever touched IS 1391.

Both paths can fire for the same QCO; when they do, the entry is returned
once, with match_reasons listing every way it was found and confidence set
by the strongest one.

This module does not call the LLM, does not mutate the knowledge base, and
performs no I/O beyond reading the QCO JSON file — it is pure, synchronous,
and safe to call from a background pipeline step or a unit test alike.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_DEFAULT_QCO_DB_PATH = _HERE.parent / "data" / "mock_qco_database.json"

# Text fields on product_profile worth scanning for product-keyword matches.
# "product" and "application" are populated by
# kartikey/analysis/profile_extractor.py::extract_profile().as_dict();
# "category" / "description" / "environment" are covered too so callers can
# pass the full profile dict as-is without pruning it first.
_PROFILE_TEXT_FIELDS = ("product", "application", "category", "environment", "description")

# Dropped when tokenizing a products_covered phrase for matching, so "shoes
# for services" reduces to the two tokens that actually carry meaning
# ("shoes", "services") instead of requiring the filler words to appear too.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "to", "for", "with",
    "all", "its", "at", "up", "down", "into", "onto", "as", "is", "are",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Confidence assigned per combination of match reasons. Both paths agreeing
# is the strongest possible signal; an exact IS-number match alone is still
# a hard fact (the standard really is on this tender); a keyword-only match
# is a heuristic over free text and should read as "worth a human look",
# not as settled — consistent with the >=0.55 threshold
# kartikey/analysis/compliance.py uses for REQUIRES_HUMAN_VERIFICATION-style
# findings.
_CONFIDENCE_BOTH = 1.0
_CONFIDENCE_IS_NUMBER_ONLY = 0.95
_CONFIDENCE_KEYWORD_ONLY = 0.65


def _normalize_is_key(raw: str) -> str:
    """
    Canonical match key for an IS number string.

    Mirrors kartikey/data/merger.py::_normalize_is_key exactly, kept as a
    private copy rather than an import so this module has no dependency on
    the ETL pipeline: it must keep working standalone against a freshly
    hand-edited mock_qco_database.json, with no merge/export step run in
    between.
    """
    s = str(raw).strip().lower()
    s = re.sub(r"\s*:\s*", ":", s)
    s = re.sub(r"\s*\(\s*", "(", s)
    s = re.sub(r"\s*\)\s*", ")", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _load_qco_records(qco_db_path: str | None) -> list[dict[str, Any]]:
    """Load and validate the QCO database JSON array."""
    path = Path(qco_db_path) if qco_db_path else _DEFAULT_QCO_DB_PATH
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(
            f"QCO database at {path} must be a JSON array of records, "
            f"got {type(data).__name__}."
        )
    return data


def _singularize(token: str) -> str:
    """
    Cheap heuristic singularizer so "chairs"/"chair", "cabinets"/"cabinet",
    "cables"/"cable" compare equal without pulling in a real NLP dependency.
    Deliberately conservative (only strips a trailing "s"/"ies") — this is a
    tie-breaker for plurals, not a stemmer, so it won't mangle words like
    "glass" (ends in "ss", left alone) or short words like "gas".
    """
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, and singularize into comparable tokens."""
    return [_singularize(t) for t in _TOKEN_RE.findall(text.lower())]


def _profile_token_set(product_profile: dict) -> set[str]:
    """
    Flatten the relevant product_profile fields and tokenize them into a
    normalized set for keyword matching. Unknown keys and non-string/list
    values are ignored rather than raising, so callers can pass a raw
    profile dict without pruning or validating it first.
    """
    parts: list[str] = []
    for field in _PROFILE_TEXT_FIELDS:
        value = product_profile.get(field)
        if isinstance(value, str) and value.strip():
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(str(v) for v in value if isinstance(v, str) and v.strip())
    return set(_tokenize(" ".join(parts)))


def _matched_products(profile_tokens: set[str], products_covered: list[Any]) -> list[str]:
    """
    Return the products_covered entries whose significant words all appear
    somewhere in profile_tokens, in any order and with any words in between.

    This is bag-of-words, not phrase matching: products_covered entry
    "ergonomic chairs" matches profile text "ergonomic revolving office
    chairs" because both "ergonomic" and "chair" (singularized) are present,
    even though "ergonomic chairs" never appears as a contiguous phrase in
    real procurement language. Stopwords ("for", "and", "of", ...) are
    dropped from the phrase first so filler words aren't required to match;
    a phrase that is nothing but stopwords/empty after that is skipped
    rather than matching everything.
    """
    matches: list[str] = []
    for product in products_covered:
        if not isinstance(product, str) or not product.strip():
            continue
        phrase_tokens = [t for t in _tokenize(product) if t not in _STOPWORDS]
        if not phrase_tokens:
            continue
        if all(t in profile_tokens for t in phrase_tokens):
            matches.append(product)
    return matches


def check_qco_applicability(
    product_profile: dict,
    matched_is_numbers: list[str],
    qco_db_path: str | None = None,
) -> list[dict]:
    """
    Determine which Quality Control Orders apply to a procurement.

    Parameters
    ----------
    product_profile:
        A dict describing what is being procured — e.g. the payload
        produced by kartikey/analysis/profile_extractor.py::extract_profile()
        .as_dict(), or any dict with a subset of "product", "application",
        "category", "environment", "description" string/list-of-string
        fields. Unrecognized keys are ignored, so the full profile dict can
        be passed as-is.
    matched_is_numbers:
        IS numbers already surfaced elsewhere in the analysis — cited by
        the tender, retrieved by the knowledge base, or both. Used for the
        exact-match path. Order and duplicates don't matter; unmatched or
        empty strings are ignored.
    qco_db_path:
        Optional override for the QCO database JSON path (e.g. a test
        fixture). Defaults to kartikey/data/mock_qco_database.json.

    Returns
    -------
    list[dict]
        One entry per applicable QCO (deduplicated by is_number), each
        shaped as:
            {
                "is_number": str,
                "notification_title": str | None,
                "issuing_ministry": str | None,
                "gazette_so_number": str | None,
                "effective_date": str | None,
                "publication_date": str | None,
                "certification_scheme": str | None,
                "is_mandatory": bool,
                "mandate_text": str | None,
                "source_url": str | None,
                "exemptions": str | None,
                "products_covered": list[str],
                "match_reasons": list[str],     # "is_number_match" and/or "product_keyword_match"
                "matched_products": list[str],  # products_covered entries found in the profile text
                "confidence": float,
            }
        Sorted by confidence (descending), then is_number, so the strongest
        and most actionable findings come first. Returns [] when no QCO in
        the database applies — a normal, expected result, not an error.

    Raises
    ------
    FileNotFoundError, ValueError:
        If the QCO database path does not exist or is not a JSON array.
        Callers that want a soft-fail (e.g. a pipeline step that must never
        crash the analysis) should catch and log around the call, the way
        kartikey/orchestration/pipeline.py::_step_enrich already does for
        its other best-effort enrichment calls.
    """
    records = _load_qco_records(qco_db_path)

    normalized_matched = {_normalize_is_key(n) for n in matched_is_numbers if n}
    profile_tokens = _profile_token_set(product_profile)

    results: list[dict[str, Any]] = []
    for record in records:
        is_number = str(record.get("is_number", "")).strip()
        if not is_number:
            continue

        match_reasons: list[str] = []

        if _normalize_is_key(is_number) in normalized_matched:
            match_reasons.append("is_number_match")

        products_covered = record.get("products_covered") or []
        matched_products = (
            _matched_products(profile_tokens, products_covered) if profile_tokens else []
        )
        if matched_products:
            match_reasons.append("product_keyword_match")

        if not match_reasons:
            continue

        if len(match_reasons) == 2:
            confidence = _CONFIDENCE_BOTH
        elif "is_number_match" in match_reasons:
            confidence = _CONFIDENCE_IS_NUMBER_ONLY
        else:
            confidence = _CONFIDENCE_KEYWORD_ONLY

        results.append({
            "is_number": is_number,
            "notification_title": record.get("notification_title"),
            "issuing_ministry": record.get("issuing_ministry"),
            "gazette_so_number": record.get("gazette_so_number"),
            "effective_date": record.get("effective_date"),
            "publication_date": record.get("publication_date"),
            "certification_scheme": record.get("certification_scheme"),
            "is_mandatory": bool(record.get("is_mandatory", False)),
            "mandate_text": record.get("mandate_text"),
            "source_url": record.get("source_url"),
            "exemptions": record.get("exemptions"),
            "products_covered": list(products_covered),
            "match_reasons": match_reasons,
            "matched_products": matched_products,
            "confidence": confidence,
        })

    results.sort(key=lambda r: (-r["confidence"], r["is_number"]))
    return results
