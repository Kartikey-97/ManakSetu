"""
shared/test_reconciled_catalogue.py

Tests for the reconciled real-data catalogue and the loader that reads it.

Why this file exists
--------------------
The backend used to load shared/bis_full_catalog_1015.json, which carried, for
all 1015 of its records: 0 scopes, 0 normative references, 0 related standards,
0 certification records. It was also missing every standard a lighting or
cabling tender actually cites — there was no IS 10322, no IS 16107, no IS 1554,
no IS 1944, no IS 7098. The pipeline compensated at runtime by synthesising
Standard objects from ai-engine HTTP payloads, which meant the report's "source"
column pointed at a record nobody had verified.

shared/bis_catalogue_reconciled.json replaces it, built from the real sources by
build_reconciled_catalogue.py. The tests below pin the two properties that make
it safe to depend on:

  1. It loses nothing. Every is_number the old catalogue had is still there, and
     every is_number the ai-engine can rank resolves.
  2. It invents nothing. The 1005 synthetic scopes in the ai-engine knowledge
     base stay out, and every field that is absent says so explicitly rather
     than arriving as a null the frontend renders as a blank cell.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.build_reconciled_catalogue import (
    BACKEND_CATALOGUE,
    OUTPUT,
    _join_key,
    reconcile,
)

_REPO = Path(__file__).resolve().parent.parent.parent
_AIML_KB = _REPO / "ai-engine" / "data" / "bis_full_knowledge_base.json"


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def catalogue() -> list[dict]:
    """The catalogue as it is committed, not as rebuilt — that is what ships."""
    with OUTPUT.open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def by_key(catalogue) -> dict[str, dict]:
    return {_join_key(r["is_number"]): r for r in catalogue}


def _load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, list) else data.get("standards", [])


# ===========================================================================
# 1. Nothing was lost
# ===========================================================================

class TestNoDataLoss:
    """
    A reconciliation that drops records is worse than the drift it replaced: the
    missing standard becomes invisible rather than merely under-described.
    """

    def test_every_backend_is_number_survives(self, by_key):
        backend_keys = {_join_key(r["is_number"]) for r in _load(BACKEND_CATALOGUE)}

        assert not (backend_keys - set(by_key))

    def test_every_aiml_recommendable_is_number_is_present(self, by_key):
        """
        The ai-engine ranks over this knowledge base, and semantic_retrieval.py
        now skips anything it cannot resolve instead of synthesising it. So a
        gap here is a gap in the demo, silently.
        """
        kb_keys = {
            _join_key(r["is_number"]) for r in _load(_AIML_KB) if r.get("is_number")
        }

        assert not (kb_keys - set(by_key))

    def test_is_numbers_are_unique(self, catalogue):
        """The old catalogue had 9 duplicates; they must not have come along."""
        keys = [_join_key(r["is_number"]) for r in catalogue]

        assert len(keys) == len(set(keys))

    def test_rebuild_is_deterministic(self, catalogue):
        """
        The committed file must be what the script produces. Otherwise a hand
        edit can smuggle in a value no source supports.
        """
        rebuilt, _ = reconcile()

        assert rebuilt == catalogue


# ===========================================================================
# 2. Nothing was invented
# ===========================================================================

class TestNothingInvented:

    def test_synthetic_scopes_are_excluded(self, by_key):
        """
        1005 of the ai-engine's 1028 scopes are `source_type: synthetic` — the
        title restated as a sentence, generated for the embedder. Those must not
        reach a procurement report as though BIS had written them.
        """
        synthetic = {
            _join_key(r["is_number"])
            for r in _load(_AIML_KB)
            if isinstance(r.get("scope"), dict)
            and (r["scope"].get("source_type") or "").casefold() == "synthetic"
        }
        assert len(synthetic) > 900, "fixture drifted; expected ~1005 synthetic scopes"

        leaked = [k for k in synthetic if by_key[k]["scope"] is not None]

        assert leaked == []

    def test_excluded_scopes_are_reported_as_unavailable(self, catalogue):
        """An absent scope must be a stated fact, not a null."""
        for record in catalogue:
            expected = "verified" if record["scope"] else "not_available"
            assert record["field_availability"]["scope"] == expected

    def test_certification_is_only_asserted_when_verified(self, catalogue):
        """
        "Certification not mandatory" is a false negative that makes an illegal
        bid look legal. It is only ever stated from a `verified: true` source.
        """
        for record in catalogue:
            cert = record["certification"]
            if cert is None:
                assert record["field_availability"]["certification"] == "not_available"
            else:
                assert cert["mandatory"] in (True, False)
                assert record["field_availability"]["certification"] == "verified"

    def test_unconfirmed_currentness_is_labelled_unverified(self, catalogue):
        """
        We hold a year and a status for every record but confirmed only 22 of
        them. VersionChecker is entitled to know which it is looking at.
        """
        availabilities = {r["field_availability"]["currentness"] for r in catalogue}

        assert availabilities == {"verified", "unverified"}

    def test_no_record_carries_a_placeholder_title(self, catalogue):
        for record in catalogue:
            assert record["title"].strip()
            assert "unknown" not in record["title"].casefold()
            assert "placeholder" not in record["title"].casefold()

    def test_every_record_names_its_source(self, catalogue):
        for record in catalogue:
            provenance = record["provenance"]
            assert provenance["organization"] == "BIS"
            assert provenance["record_source"] in {
                "bis_full_knowledge_base", "curated_additions",
            }


# ===========================================================================
# 3. The records the demo depends on carry real content
# ===========================================================================

class TestDemoStandardsAreRich:
    """
    Not a demand that every standard be complete — most BIS records genuinely
    are not. These are the specific ones whose real data the scenarios rely on,
    and each assertion below is a value that exists in the source, not one we
    would like to exist.
    """

    def test_led_street_luminaire_carries_its_reference_chain(self, by_key):
        record = by_key[_join_key("IS 10322 : Part 5 : Sec 3")]

        assert record["year"] == 2026
        assert record["scope"]
        assert record["normative_references"] == ["IS 10322 : Part 1"]
        assert record["related_standards"] == ["IS 1944 : Part 1 and 2"]
        assert record["currentness"]["supersedes"] == "IS 10322: Part 5: Sec 3: 2012"
        assert record["certification"]["mandatory"] is True
        assert record["certification"]["scheme"] == "crs"
        assert "IPX9 ingress protection" in record["test_methods"]

    def test_led_performance_standard_carries_its_qco(self, by_key):
        record = by_key[_join_key("IS 16107")]

        assert record["certification"]["qco"] == "QCO 2020"
        assert record["certification"]["qco_effective_date"] == "2020-01-01"
        assert record["normative_references"] == ["IS 10322", "IS 1944"]
        assert len(record["test_methods"]) >= 5

    def test_lv_cable_standard_carries_its_dependencies(self, by_key):
        record = by_key[_join_key("IS 1554 : Part 1")]

        assert record["year"] == 1988
        assert record["normative_references"] == ["IS 694", "IS 8130"]
        assert record["certification"]["mandatory"] is True

    def test_motor_efficiency_standard_carries_its_qco(self, by_key):
        record = by_key[_join_key("IS 12615")]

        assert record["certification"]["qco"] == "QCO 2018"
        assert record["certification"]["qco_effective_date"] == "2018-10-01"
        assert record["scope"] and "IE1" in record["scope"]


# ===========================================================================
# 4. The loader does not lose any of it
# ===========================================================================

class TestLoader:
    """
    Every field the reconciliation established has to survive the trip into a
    Standard. A catalogue nobody reads is not an improvement.
    """

    @pytest.fixture(scope="class")
    def store(self):
        from kshiraj.knowledge.standards_store import StandardsStore
        from kartikey.orchestration.knowledge_registry import _standard_from_catalogue

        store = StandardsStore()
        with OUTPUT.open(encoding="utf-8") as handle:
            for item in json.load(handle):
                store.add(_standard_from_catalogue(item))
        return store

    def test_the_whole_catalogue_loads(self, store, catalogue):
        assert len(store.list_all()) == len(catalogue)

    def test_real_scope_and_references_reach_the_model(self, store):
        std = store.get_by_is_number("IS 10322 : Part 5 : Sec 3")[0]

        assert std.year == 2026
        assert std.scope
        assert std.normative_references == ["IS 10322 : Part 1"]
        assert std.related_standards == ["IS 1944 : Part 1 and 2"]
        assert std.supersedes == "IS 10322: Part 5: Sec 3: 2012"
        assert std.test_methods == ["IPX9 ingress protection", "photobiological safety"]

    def test_certification_reaches_the_model(self, store):
        from shared.models import CertificationScheme

        std = store.get_by_is_number("IS 16107")[0]

        assert std.qco_notified is True
        assert std.qco_gazette_so_number == "QCO 2020"
        assert std.qco_effective_date is not None
        assert std.qco_effective_date.isoformat() == "2020-01-01"
        assert std.required_certification_scheme is CertificationScheme.CRS

    def test_field_availability_reaches_the_model(self, store):
        std = store.get_by_is_number("IS 16107")[0]

        assert std.field_availability["scope"] == "verified"
        assert std.field_availability["certification"] == "verified"

    def test_provenance_reaches_the_model(self, store):
        std = store.get_by_is_number("IS 10322 : Part 5 : Sec 3")[0]

        assert std.provenance is not None
        assert std.provenance["organization"] == "BIS"

    def test_a_standard_with_no_real_scope_says_so(self, store):
        """
        The common case: 1005 records have no published scope we can use. The
        model must carry that as an explicit state, because the alternative is a
        frontend that cannot distinguish it from a loading failure.
        """
        blank = [
            s for s in store.list_all()
            if s.field_availability.get("scope") == "not_available"
        ]

        assert len(blank) > 900
        assert all(s.scope is None for s in blank)
        # Still perfectly usable records — they have the identity fields.
        assert all(s.is_number and s.title for s in blank)
