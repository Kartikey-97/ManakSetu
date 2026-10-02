"""
kartikey/orchestration/test_qco_scope.py

Which IS numbers the QCO applicability check in `_step_enrich` is given.

`check_qco_applicability` treats an IS-number match as a hard fact that the
standard is on this tender. So it must be fed the standards that are applicable
to a requirement or cited by one — never candidates that were merely retrieved.
It is also fed `is_number`, not `designation`: every catalogue standard carries a
year, and "IS 694:2010" never equals the QCO key "IS 694".

Uses the real QCO database (kartikey/data/mock_qco_database.json), whose
records include IS 694, IS 732, IS 10322 and IS 10322 : Part 5 : Sec 3.
"""

from __future__ import annotations

import asyncio

import pytest

from shared.contracts import AimlFinding, AimlResponse
from shared.models import (
    Analysis,
    AnalysisStatus,
    InputType,
    Requirement,
    Standard,
    StandardStatus,
    Verdict,
)
from kartikey.analysis import certification_engine
from kartikey.analysis.certification_engine import check_qco_applicability
from kartikey.orchestration.knowledge_registry import initialize_knowledge_registry
from kartikey.orchestration.pipeline import _step_enrich, run_analysis_pipeline

# Matches no products_covered entry in the QCO database, so only the IS-number
# path can produce a result.
NEUTRAL_PROFILE = {"product": "office stationery paper", "category": "Stationery"}


@pytest.fixture
def registry(monkeypatch):
    registry = initialize_knowledge_registry()
    registry.standards_store.clear()
    registry.evidence_store.clear()
    monkeypatch.setattr("shared.config.settings.enable_bis_sync", False)
    return registry


@pytest.fixture
def qco_calls(monkeypatch):
    """Record the matched_is_numbers each QCO check receives."""
    calls: list[list[str]] = []
    real = certification_engine.check_qco_applicability

    def _spy(product_profile, matched_is_numbers, qco_db_path=None):
        calls.append(list(matched_is_numbers))
        return real(product_profile, matched_is_numbers, qco_db_path)

    monkeypatch.setattr(certification_engine, "check_qco_applicability", _spy)
    return calls


def _std(registry, is_number: str, year: int | None = 2010, **kwargs) -> Standard:
    std = Standard(
        is_number=is_number, year=year, title=f"Standard {is_number}",
        status=kwargs.pop("status", StandardStatus.ACTIVE), **kwargs,
    )
    registry.standards_store.add(std)
    return std


def _enrich(reqs, candidates, selections, retrieved, profile=NEUTRAL_PROFILE) -> Analysis:
    """
    Run _step_enrich as the pipeline would after retrieval and AI analysis.

    candidates: requirement → standards retrieved for it (the Stage 1 map)
    selections: requirement → standards the AI/ML selected as applicable
    """
    analysis = Analysis(
        input_type=InputType.TEXT, raw_text="tender", status=AnalysisStatus.QUEUED,
    )
    for req in reqs:
        req.analysis_id = analysis.id
    analysis.requirements = list(reqs)
    analysis.product_profile = dict(profile)
    analysis.metadata["requirement_candidates"] = {
        req.id: {s.id: 1.0 for s in candidates.get(req.id, [])} for req in reqs
    }
    response = AimlResponse(
        analysis_id=analysis.id,
        findings=[
            AimlFinding(
                finding_id=f"f-{req.id[:8]}", requirement_id=req.id,
                verdict=Verdict.JUSTIFIED.value, reason="stub", confidence=0.9,
                applicable_standard_ids=[s.id for s in selections.get(req.id, [])],
            )
            for req in reqs
        ],
    )
    asyncio.run(_step_enrich(analysis, retrieved, response))
    return analysis


def _is_matches(analysis: Analysis) -> list[str]:
    return [
        q["is_number"] for q in (analysis.qco_findings or [])
        if "is_number_match" in q["match_reasons"]
    ]


# ===========================================================================
# Which standards reach the IS-number path
# ===========================================================================

def test_retrieved_only_standard_does_not_match(registry, qco_calls) -> None:
    is694 = _std(registry, "IS 694")
    is732 = _std(registry, "IS 732")
    req = Requirement(analysis_id="x", text="Five-year warranty required.")

    analysis = _enrich(
        [req], candidates={req.id: [is694, is732]}, selections={},
        retrieved=[is694, is732],
    )

    assert qco_calls == [[]]
    assert _is_matches(analysis) == []


def test_applicable_standard_matches_without_citation(registry, qco_calls) -> None:
    is694 = _std(registry, "IS 694")
    is732 = _std(registry, "IS 732")
    req = Requirement(analysis_id="x", text="Five-year warranty required.")

    analysis = _enrich(
        [req], candidates={req.id: [is694, is732]}, selections={req.id: [is694]},
        retrieved=[is694, is732],
    )

    assert qco_calls == [["IS 694"]]
    assert _is_matches(analysis) == ["IS 694"]


def test_cited_standard_matches_when_ai_applicability_is_empty(registry, qco_calls) -> None:
    is694 = _std(registry, "IS 694")
    req = Requirement(
        analysis_id="x", text="Cables shall conform to IS 694.", is_reference="IS 694",
    )

    analysis = _enrich(
        [req], candidates={req.id: [is694]}, selections={}, retrieved=[is694],
    )

    assert analysis.findings[0].applicable_standards == []
    assert _is_matches(analysis) == ["IS 694"]


def test_standard_with_year_matches_its_base_qco_key(registry) -> None:
    is694 = _std(registry, "IS 694", year=2010)
    assert is694.designation == "IS 694:2010"   # the key that never matched
    req = Requirement(analysis_id="x", text="Five-year warranty required.")

    analysis = _enrich(
        [req], candidates={req.id: [is694]}, selections={req.id: [is694]},
        retrieved=[is694],
    )

    entry = next(q for q in analysis.qco_findings if q["is_number"] == "IS 694")
    assert entry["match_reasons"] == ["is_number_match"]
    assert entry["confidence"] == 0.95


def test_cited_and_applicable_standard_submitted_once(registry, qco_calls) -> None:
    is694 = _std(registry, "IS 694")
    cited = Requirement(
        analysis_id="x", text="Cables shall conform to IS 694.", is_reference="IS 694",
    )
    other = Requirement(analysis_id="x", text="Insulation grade as per IS 694.", is_reference="IS 694")

    analysis = _enrich(
        [cited, other],
        candidates={cited.id: [is694], other.id: [is694]},
        selections={cited.id: [is694], other.id: [is694]},
        retrieved=[is694],
    )

    assert qco_calls == [["IS 694"]]
    assert [q["is_number"] for q in analysis.qco_findings].count("IS 694") == 1


def test_keyword_results_unchanged(registry) -> None:
    """With no applicable or cited standard, results are the keyword path alone."""
    is10322 = _std(registry, "IS 10322 : Part 5 : Sec 3", year=2012)
    profile = {"product": "90W LED street light luminaires", "category": "Outdoor Lighting"}
    req = Requirement(analysis_id="x", text="Five-year warranty required.")

    analysis = _enrich(
        [req], candidates={req.id: [is10322]}, selections={},
        retrieved=[is10322], profile=profile,
    )

    expected = check_qco_applicability(profile, [])
    assert expected, "profile should produce keyword matches for this test to mean anything"
    assert analysis.qco_findings == expected


def test_superseding_standard_injected_after_enrichment_does_not_alter_qco(
    registry, monkeypatch,
) -> None:
    """GET /analyses/{id} injects a superseding standard into the finding; QCO stays put."""
    from kartikey.api.routes import analyses as analyses_route

    replacement = _std(registry, "IS 732", year=2019)
    old = _std(
        registry, "IS 9999", year=2001,
        status=StandardStatus.SUPERSEDED, superseded_by="IS 732",
    )
    req = Requirement(analysis_id="x", text="Five-year warranty required.")
    analysis = _enrich(
        [req], candidates={req.id: [old]}, selections={req.id: [old]}, retrieved=[old],
    )
    before = [dict(q) for q in analysis.qco_findings]

    async def _get(_analysis_id):
        return analysis

    monkeypatch.setattr(analyses_route.repository, "get", _get)
    response = asyncio.run(analyses_route.get_analysis(analysis.id))

    # The injection really happened …
    assert replacement.id in {s.id for s in analysis.findings[0].applicable_standards}
    assert replacement.id in {s.id for s in response.standards}
    # … and the QCO result computed at enrichment did not move.
    assert analysis.qco_findings == before
    assert "IS 732" not in [q["is_number"] for q in analysis.qco_findings]


# ===========================================================================
# Demo scenarios through the real pipeline and the real catalogue
# ===========================================================================

SCENARIO_LED = (
    "Supply, installation, testing and commissioning of 90W LED street light "
    "luminaires for the municipal road lighting project. The luminaires shall "
    "conform to IS 10322 (Part 5/Sec 3) : 2012 and shall comply with the "
    "performance requirements of IS 16107. Minimum luminaire efficacy shall be "
    "110 lm/W with IP66 ingress protection. Bidders must submit a valid BIS "
    "registration certificate under the CRS scheme."
)

SCENARIO_CABLE = (
    "Supply of 1.1 kV grade XLPE insulated armoured power cables, "
    "3.5 core 185 sqmm aluminium conductor, conforming to IS 1554 : Part 1. "
    "Conductors shall conform to IS 8130 and insulation to IS 694. "
    "Test certificates from an NABL accredited laboratory shall be furnished."
)


def _run(text: str) -> Analysis:
    analysis = Analysis(input_type=InputType.TEXT, raw_text=text, status=AnalysisStatus.QUEUED)
    store = {analysis.id: analysis}
    asyncio.run(run_analysis_pipeline(analysis.id, store))
    return store[analysis.id]


@pytest.fixture(scope="module")
def real_registry():
    return initialize_knowledge_registry()


@pytest.fixture(scope="module")
def led(real_registry) -> Analysis:
    from shared.config import settings
    previous, settings.enable_bis_sync = settings.enable_bis_sync, False
    try:
        return _run(SCENARIO_LED)
    finally:
        settings.enable_bis_sync = previous


@pytest.fixture(scope="module")
def cable(real_registry) -> Analysis:
    from shared.config import settings
    previous, settings.enable_bis_sync = settings.enable_bis_sync, False
    try:
        return _run(SCENARIO_CABLE)
    finally:
        settings.enable_bis_sync = previous


def test_led_matches_part_5_section_3_qco(led) -> None:
    entry = next(
        (q for q in led.qco_findings if q["is_number"] == "IS 10322 : Part 5 : Sec 3"), None,
    )
    assert entry is not None, led.qco_findings
    assert "is_number_match" in entry["match_reasons"]
    assert entry["certification_scheme"] == "crs"


def test_led_does_not_match_plain_indoor_is_10322_record(led) -> None:
    assert "IS 10322" not in _is_matches(led)


def test_cable_does_not_match_is_694_merely_because_retrieved(cable, real_registry) -> None:
    # Precondition: IS 694 really is in this analysis's retrieved candidates.
    candidate_ids = {
        sid for per_req in cable.metadata["requirement_candidates"].values() for sid in per_req
    }
    retrieved_numbers = {
        real_registry.standards_store.get_by_id(sid).is_number for sid in candidate_ids
    }
    assert "IS 694" in retrieved_numbers
    # No requirement has it applicable or resolves a citation to it.
    assert not any(
        s.is_number == "IS 694" for f in cable.findings for s in f.applicable_standards
    )
    assert "IS 694" not in _is_matches(cable)
