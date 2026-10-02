"""
shared/build_reconciled_catalogue.py

Build the backend's canonical standards catalogue from the real BIS sources.

Why this exists
---------------
There were two catalogues describing the same standards, and they disagreed:

    backend/shared/bis_full_catalog_1015.json      1015 records, 1006 unique
    ai-engine/data/bis_full_knowledge_base.json    1028 records
    ai-engine/data/curated_additions.json            13 records

Measured, not assumed: every is_number in the backend catalogue is present in
the ai-engine knowledge base (`backend - ai-engine == 0`), as are all 13 curated
additions. The knowledge base then adds 10 further real BIS records — and they
are the ones that matter most, because they are the IS 10322 luminaire parts and
the IS 1944 public-lighting parts that an LED street-lighting tender actually
cites. The backend catalogue also carried no scope, no normative references, no
related standards, and no certification data at all: 0 of 1015 for each.

So the knowledge base is a strict superset in both coverage and content, and it
is the canonical source. The backend catalogue contributes exactly one thing the
knowledge base lacks — the BIS list `price` — which is joined back in.

The provenance filter
---------------------
The knowledge base is not uniformly real, and this script does not pretend it is.
Its `scope` field is marked `source_type: synthetic` for 1005 of 1028 records:
those values are the title restated as a sentence, generated to give the
retriever something to embed. Importing them would put synthetic prose in front
of a procurement officer as though BIS had written it.

So each field is imported only when its own provenance says it is real, and the
result is recorded per field in `field_availability`:

    scope          official/curated  -> verified      (23 records)
                   synthetic         -> not_available (1005 records)
    certification  verified: true    -> verified      (17 records)
                   otherwise         -> not_available
    references     non-empty         -> verified      (23 records)
                   empty             -> not_available
    currentness    status verified   -> verified, else unverified

`not_available` is a claim about BIS's published data, and the loader and API
carry it through rather than emitting a bare null. Nothing here invents a value.

Usage
-----
    .venv/bin/python -m shared.build_reconciled_catalogue

Writes shared/bis_catalogue_reconciled.json and prints a before/after summary.
Deterministic: same inputs produce a byte-identical file.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent

BACKEND_CATALOGUE = _HERE / "bis_full_catalog_1015.json"
AIML_KNOWLEDGE_BASE = _REPO / "ai-engine" / "data" / "bis_full_knowledge_base.json"
CURATED_ADDITIONS = _REPO / "ai-engine" / "data" / "curated_additions.json"
OUTPUT = _HERE / "bis_catalogue_reconciled.json"

# Scope provenance we accept as real. "synthetic" is excluded on purpose — see
# the module docstring.
_REAL_SCOPE_SOURCES = frozenset({"official", "curated", "bis.gov.in"})

# ai-engine status.value -> our StandardStatus value. Mirrors the map in
# kartikey/orchestration/semantic_retrieval.py so both readers agree.
_STATUS_MAP = {
    "active": "active",
    "active (concurrent)": "active",
    "under revision": "under_revision",
    "reaffirmed": "reaffirmed",
    "superseded": "superseded",
    "withdrawn": "withdrawn",
}

# ai-engine standard_type -> our DocumentType value. Safety, Installation and
# Symbols & Markings have no honest equivalent, so they stay "other".
_TYPE_MAP = {
    "specification": "product_specification",
    "code of practice": "code_of_practice",
    "testing": "method_of_test",
    "glossary": "terminology",
    "guide": "guide",
}

# BIS scheme names as the sources spell them -> our CertificationScheme value.
_SCHEME_MAP = {
    "crs": "crs",
    "bis crs": "crs",
    "isi": "isi_mark",
    "isi mark": "isi_mark",
    "bis isi": "isi_mark",
    "hallmarking": "hallmarking",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, list) else data.get("standards", [])


def _join_key(is_number: str) -> str:
    """
    Match the same standard across catalogues despite spacing differences.

    The knowledge base writes "IS 10322: Part 5: Sec 3" in one field and
    "IS 10322 : Part 5 : Sec 3" in another, so whitespace is removed entirely
    rather than normalised. Parts and sections are preserved: IS 7098 Part 1 and
    Part 2 are different standards with different requirements and must not merge.
    """
    return "".join(str(is_number).split()).upper()


# ---------------------------------------------------------------------------
# Field extraction, each with its own provenance test
# ---------------------------------------------------------------------------

def _unwrap(field: Any) -> Any:
    """Read a value the ai-engine may wrap in a {value, source_type, verified} envelope."""
    if isinstance(field, dict):
        return field.get("value")
    return field


def _scope(record: dict) -> tuple[str | None, str]:
    """
    Return (scope_text, availability).

    A synthetic scope is dropped, not downgraded. It restates the title, so
    keeping it as "unverified" would still put generated prose in the report.
    """
    raw = record.get("scope")
    if isinstance(raw, dict):
        value = (raw.get("value") or "").strip()
        source_type = (raw.get("source_type") or "").strip().casefold()
        if value and source_type in _REAL_SCOPE_SOURCES:
            return value, "verified"
        return None, "not_available"

    value = (raw or "").strip() if isinstance(raw, str) else ""
    # A bare string carries no provenance envelope. The curated file uses this
    # shape and is real by construction; nothing else in these inputs does.
    return (value, "verified") if value else (None, "not_available")


def _certification(record: dict) -> tuple[dict | None, str]:
    """
    Return (certification, availability).

    Only a `verified: true` block is imported. An unverified block in these
    sources is uniformly `mandatory: null, scheme: null` — no information — and
    presenting it as "certification not mandatory" would be a false negative on
    a question that decides whether a bid is legal.
    """
    cert = record.get("certification")
    if not isinstance(cert, dict) or not cert.get("verified"):
        return None, "not_available"

    mandatory = cert.get("mandatory")
    if mandatory is None:
        return None, "not_available"

    scheme_raw = (cert.get("scheme") or "").strip()
    return {
        "mandatory": bool(mandatory),
        "scheme": _SCHEME_MAP.get(scheme_raw.casefold()),
        "scheme_label": scheme_raw or None,
        "qco": cert.get("qco"),
        "qco_effective_date": cert.get("qco_effective_date"),
        "source": cert.get("source"),
    }, "verified"


def _references(record: dict) -> tuple[list[str], list[str], str]:
    """Return (normative_references, related_standards, availability)."""
    def clean(key: str) -> list[str]:
        value = record.get(key)
        if not isinstance(value, list):
            return []
        # dict.fromkeys preserves catalogue order while dropping repeats.
        return list(dict.fromkeys(
            item.strip() for item in value if isinstance(item, str) and item.strip()
        ))

    normative, related = clean("normative_references"), clean("related_standards")
    availability = "verified" if (normative or related) else "not_available"
    return normative, related, availability


def _currentness(record: dict) -> tuple[dict, str]:
    """
    Return (currentness, availability).

    `supersedes` lives on `status` in this knowledge base and on `version` in the
    curated file, so both are read. Availability is "verified" only when the
    source says the status was checked — for the rest we hold a year and a status
    we did not confirm, which VersionChecker must be told about rather than
    trusting silently.
    """
    status_block = record.get("status") if isinstance(record.get("status"), dict) else {}
    version_block = record.get("version") if isinstance(record.get("version"), dict) else {}

    def pick(key: str) -> Any:
        return status_block.get(key) or version_block.get(key)

    status_value = (_unwrap(record.get("status")) or "").strip().casefold()
    verified = bool(status_block.get("verified")) or "verified" in {
        status_block.get("verification_status"), version_block.get("verification_status"),
    }

    return {
        "status": _STATUS_MAP.get(status_value, "unknown"),
        "status_label": _unwrap(record.get("status")) or None,
        "supersedes": pick("supersedes"),
        "superseded_by": pick("superseded_by"),
        "latest_known_edition": pick("latest_known_edition"),
        "latest_known_year": pick("latest_known_year"),
        "concurrent_running": status_block.get("concurrent_running"),
    }, ("verified" if verified else "unverified")


def _provenance(record: dict, record_source: str) -> dict:
    source = record.get("source")
    if isinstance(source, dict):
        return {
            "organization": source.get("organization") or "BIS",
            "source_type": source.get("source_type"),
            "verified": bool(source.get("verified")),
            "url": source.get("url"),
            "record_source": record_source,
        }
    return {
        "organization": "BIS",
        "source_type": source if isinstance(source, str) else None,
        "verified": False,
        "url": None,
        "record_source": record_source,
    }


def _str_list(record: dict, key: str) -> list[str]:
    value = record.get(key)
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(
        item.strip() for item in value if isinstance(item, str) and item.strip()
    ))


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

def reconcile() -> tuple[list[dict], dict]:
    """
    Build the canonical catalogue. Returns (records, stats).

    The knowledge base drives the record set. The curated file is applied on top
    of it for the standards it covers — it is the more carefully checked of the
    two for those, and it is where `scope_summary` comes from. The backend
    catalogue contributes `price` only.
    """
    kb = _load(AIML_KNOWLEDGE_BASE)
    curated = {_join_key(r.get("is_number", "")): r for r in _load(CURATED_ADDITIONS)}
    backend = _load(BACKEND_CATALOGUE)

    prices: dict[str, float] = {}
    valid_upto: dict[str, str] = {}
    for record in backend:
        key = _join_key(record.get("is_number", ""))
        if isinstance(record.get("price"), (int, float)):
            prices[key] = float(record["price"])
        if record.get("valid_upto"):
            valid_upto[key] = record["valid_upto"]

    out: list[dict] = []
    seen: set[str] = set()
    stats: Counter = Counter()

    for record in kb:
        is_number = (record.get("is_number") or "").strip()
        title = (record.get("title") or record.get("raw_title") or "").strip()
        if not is_number or not title:
            stats["skipped_incomplete"] += 1
            continue

        key = _join_key(is_number)
        if key in seen:
            # The backend catalogue had 9 duplicate is_numbers; do not carry them
            # forward. First occurrence wins, which is the knowledge base's order.
            stats["duplicates_collapsed"] += 1
            continue
        seen.add(key)

        # The curated record, where one exists, is the better-checked source for
        # scope and certification.
        curated_record = curated.get(key)
        scope, scope_availability = _scope(curated_record or record)
        if scope is None and curated_record is not None:
            scope, scope_availability = _scope(record)

        certification, cert_availability = _certification(curated_record or record)
        if certification is None and curated_record is not None:
            certification, cert_availability = _certification(record)

        normative, related, refs_availability = _references(curated_record or record)
        if not normative and not related and curated_record is not None:
            normative, related, refs_availability = _references(record)

        currentness, currentness_availability = _currentness(record)
        source_record = curated_record or record

        out.append({
            "is_number": is_number,
            "normalized_is_number": (record.get("normalized_is_number") or is_number).strip(),
            "title": title,
            "year": record.get("edition_year") or record.get("year"),
            "document_type": _TYPE_MAP.get(
                (record.get("standard_type") or "").strip().casefold(), "other",
            ),
            "standard_type_label": record.get("standard_type"),

            "scope": scope,
            "scope_summary": (source_record.get("scope_summary") or "").strip() or None,

            "normative_references": normative,
            "related_standards": related,

            "certification": certification,
            "currentness": currentness,

            "keywords": _str_list(record, "keywords"),
            "product_categories": _str_list(record, "product_categories"),
            "test_methods": _str_list(source_record, "test_methods"),
            "amendments": record.get("amendments") or [],

            "price": prices.get(key),
            "valid_upto": record.get("valid_upto") or valid_upto.get(key),

            "provenance": _provenance(
                source_record,
                "curated_additions" if curated_record else "bis_full_knowledge_base",
            ),
            "field_availability": {
                "scope": scope_availability,
                "certification": cert_availability,
                "normative_references": refs_availability,
                "currentness": currentness_availability,
                "test_methods": (
                    "verified" if _str_list(source_record, "test_methods")
                    else "not_available"
                ),
                "price": "verified" if key in prices else "not_available",
            },
        })

        stats[f"scope_{scope_availability}"] += 1
        stats[f"certification_{cert_availability}"] += 1
        stats[f"references_{refs_availability}"] += 1
        stats[f"currentness_{currentness_availability}"] += 1

    stats["records"] = len(out)
    stats["backend_records_in"] = len(backend)
    stats["kb_records_in"] = len(kb)
    stats["curated_records_in"] = len(curated)
    stats["priced"] = sum(1 for r in out if r["price"] is not None)

    # Coverage check: every backend is_number must survive, or we have lost data.
    backend_keys = {_join_key(r.get("is_number", "")) for r in backend}
    stats["backend_keys_missing"] = len(backend_keys - seen)

    return out, stats


def main() -> None:
    records, stats = reconcile()

    OUTPUT.write_text(
        json.dumps(records, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    print(f"wrote {OUTPUT.relative_to(_REPO)}")
    print(f"  inputs : backend={stats['backend_records_in']} "
          f"kb={stats['kb_records_in']} curated={stats['curated_records_in']}")
    print(f"  output : {stats['records']} records "
          f"({stats['duplicates_collapsed']} duplicate is_numbers collapsed)")
    print(f"  backend is_numbers lost: {stats['backend_keys_missing']}")
    for field in ("scope", "certification", "references", "currentness"):
        verified = stats.get(f"{field}_verified", 0)
        unverified = stats.get(f"{field}_unverified", 0)
        missing = stats.get(f"{field}_not_available", 0)
        print(f"  {field:22s} verified={verified:5d} "
              f"unverified={unverified:5d} not_available={missing:5d}")
    print(f"  price                  verified={stats['priced']:5d}")

    if stats["backend_keys_missing"]:
        raise SystemExit("refusing to ship a catalogue that drops backend records")


if __name__ == "__main__":
    main()
