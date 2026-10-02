"""
kartikey/orchestration/test_citation_designation.py

A part-qualified citation keeps its part and section all the way through:
Requirement.is_reference carries the catalogue designation, so exact lookup,
the compliance-only path, the cited-QCO path and certification resolve the
cited part — while the lexical retrieval query still gets the base number, so
the candidates retrieved are exactly what they were before.

Runs against the real reconciled catalogue.
"""

from __future__ import annotations

import asyncio

import pytest

from shared.contracts import AimlFinding, AimlResponse
from shared.models import Analysis, AnalysisStatus, InputType, Requirement, Verdict
from shared.utils import AnalysisError

from kartikey.analysis import certification_engine
from kartikey.document_processing.extractor import base_is_number
from kartikey.orchestration import pipeline
from kartikey.orchestration.knowledge_registry import initialize_knowledge_registry

LED = (
    "Supply, installation, testing and commissioning of 90W LED street light "
    "luminaires for the municipal road lighting project. The luminaires shall "
    "conform to IS 10322 (Part 5/Sec 3) : 2012 and shall comply with the "
    "performance requirements of IS 16107. Minimum luminaire efficacy shall be "
    "110 lm/W with IP66 ingress protection."
)
SEC3 = "IS 10322 : Part 5 : Sec 3"


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr("shared.config.settings.enable_bis_sync", False)
    return initialize_knowledge_registry().standards_store


def _run(text: str) -> Analysis:
    analysis = Analysis(input_type=InputType.TEXT, raw_text=text, status=AnalysisStatus.QUEUED)
    asyncio.run(pipeline.run_analysis_pipeline(analysis.id, {analysis.id: analysis}))
    return analysis


def _cited(analysis: Analysis, is_reference: str) -> Requirement:
    return next(r for r in analysis.requirements if r.is_reference == is_reference)


def _finding_for(analysis: Analysis, req: Requirement):
    return next(f for f in analysis.findings if f.requirement_id == req.id)


# ===========================================================================
# Both extraction paths
# ===========================================================================

MIXED = (
    "Luminaires shall conform to IS 10322 (Part 5/Sec 3) : 2012.\n\n"
    "Cables shall conform to IS 1554 : Part 1 : 1988.\n\n"
    "Structural steel shall conform to IS : 2062 - 2011.\n\n"
    "Pipes shall conform to IS 1554 (भाग 1) : 1988."
)
EXPECTED = {
    ("IS 10322 : Part 5 : Sec 3", 2012, "IS 10322 (Part 5/Sec 3) : 2012"),
    ("IS 1554 : Part 1", 1988, "IS 1554 : Part 1 : 1988"),
    ("IS 2062", 2011, "IS : 2062 - 2011"),
    ("IS 1554", 1988, "IS 1554 (भाग 1) : 1988"),
}


def _fields(requirements) -> set[tuple]:
    return {(r.is_reference, r.cited_year, r.cited_designation) for r in requirements if r.is_reference}


def test_requirement_extraction_preserves_citation_fields() -> None:
    from kartikey.analysis.requirement_extractor import extract_requirements

    requirements, _ = extract_requirements("a1", MIXED)

    assert _fields(requirements) == EXPECTED


def test_regex_fallback_matches_requirement_extraction(store, monkeypatch) -> None:
    """The fallback used when extraction is unavailable yields the same citation fields."""
    from kartikey.analysis import requirement_extractor

    def _unavailable(**_kwargs):
        raise AnalysisError("no extractor", code="LLM_NOT_CONFIGURED")

    monkeypatch.setattr(requirement_extractor, "extract_requirements", _unavailable)
    analysis = Analysis(input_type=InputType.TEXT, raw_text=MIXED, status=AnalysisStatus.QUEUED)

    asyncio.run(pipeline._step_extract(analysis))

    assert analysis.metadata.get("extraction_fallback") is True
    assert _fields(analysis.requirements) == EXPECTED


# ===========================================================================
# Lexical retrieval is unchanged
# ===========================================================================

def test_lexical_query_and_candidates_use_the_base_number(store, monkeypatch) -> None:
    registry = initialize_knowledge_registry()
    service = registry.retrieval_service
    real_search = service.search_standards
    queries: list[str] = []

    def _spy(query):
        queries.append(query.query_text)
        return real_search(query)

    monkeypatch.setattr(service, "search_standards", _spy)
    analysis = Analysis(input_type=InputType.TEXT, raw_text=LED, status=AnalysisStatus.QUEUED)
    asyncio.run(pipeline._step_extract(analysis))
    asyncio.run(pipeline._step_retrieve(analysis, LED))

    req = _cited(analysis, SEC3)
    query = next(q for q in queries if "IS 10322" in q)
    assert query.endswith("IS 10322") and "Part" not in query

    # What the base query retrieves is exactly this requirement's search candidates.
    monkeypatch.setattr(service, "search_standards", real_search)
    from kshiraj.knowledge.retrieval_service import RetrievalQuery
    base = real_search(RetrievalQuery(query_text=query, top_k=8, include_evidence=False))
    recorded = set(analysis.metadata["requirement_candidates"][req.id])
    exact = {s.id for s in store.get_by_is_number(req.is_reference)}
    assert recorded == {c.standard.id for c in base.candidates} | exact


def test_base_is_number_matches_the_previous_query_value() -> None:
    """Before this change the query was the base number itself; it still is."""
    for text, previous in [
        ("IS 10322 (Part 5/Sec 3) : 2012", "IS 10322"),
        ("IS 1554 : Part 1 : 1988", "IS 1554"),
        ("IS 2062", "IS 2062"),
        ("IS : 2062 - 2011", "IS 2062"),
    ]:
        from kartikey.document_processing.extractor import scan_is_references
        ref = scan_is_references(text)[0]
        assert base_is_number(ref["designation"]) == ref["is_number"] == previous


# ===========================================================================
# Downstream resolution
# ===========================================================================

def test_led_citation_resolves_to_the_exact_part(store) -> None:
    analysis = _run(LED)
    req = _cited(analysis, SEC3)

    assert req.cited_year == 2012
    assert req.cited_designation == "IS 10322 (Part 5/Sec 3) : 2012"
    assert [s.is_number for s in store.get_by_is_number(req.is_reference)] == [SEC3]


def test_absent_part_keeps_the_family_fallback(store) -> None:
    text = "Supply of 1.1 kV PVC insulated power cables conforming to IS 1554 : Part 2 : 2010."
    analysis = _run(text)
    req = _cited(analysis, "IS 1554 : Part 2")

    assert req.cited_year == 2010
    assert store.get_by_is_number(req.is_reference) == store.get_by_is_number("IS 1554")


def test_colon_form_year_reaches_the_version_check(store) -> None:
    """Colon and bracket notation of the same citation now agree."""
    colon = _run("Supply of 1.1 kV PVC insulated power cables conforming to IS 1554 : Part 1 : 1988.")
    bracket = _run("Supply of 1.1 kV PVC insulated power cables conforming to IS 1554 (Part 1) : 1988.")

    fc = _finding_for(colon, _cited(colon, "IS 1554 : Part 1"))
    fb = _finding_for(bracket, _cited(bracket, "IS 1554 : Part 1"))
    assert fc.verdict == fb.verdict == Verdict.JUSTIFIED
    assert fc.dimensions["currentness"]["label"] == "CURRENT"


def test_cited_qco_path_uses_the_exact_cited_part(store, monkeypatch) -> None:
    """With nothing selected as applicable, only the cited part reaches the QCO check."""
    calls: list[list[str]] = []
    real = certification_engine.check_qco_applicability

    def _spy(product_profile, matched_is_numbers, qco_db_path=None):
        calls.append(list(matched_is_numbers))
        return real(product_profile, matched_is_numbers, qco_db_path)

    monkeypatch.setattr(certification_engine, "check_qco_applicability", _spy)

    analysis = Analysis(input_type=InputType.TEXT, raw_text="tender", status=AnalysisStatus.QUEUED)
    req = Requirement(analysis_id=analysis.id, text="LED luminaires", is_reference=SEC3, cited_year=2012)
    analysis.requirements = [req]
    analysis.product_profile = {"product": "office stationery"}
    sec3 = store.get_by_is_number(SEC3)[0]
    analysis.metadata["requirement_candidates"] = {req.id: {sec3.id: 1.0}}
    response = AimlResponse(analysis_id=analysis.id, findings=[AimlFinding(
        finding_id="f", requirement_id=req.id, verdict="justified", reason="stub",
        confidence=0.9, applicable_standard_ids=[],
    )])

    asyncio.run(pipeline._step_enrich(analysis, [sec3], response))

    assert calls == [[SEC3]]
    matched = [q["is_number"] for q in analysis.qco_findings if "is_number_match" in q["match_reasons"]]
    assert matched == [SEC3]


def test_certification_details_come_from_the_cited_section(store) -> None:
    analysis = _run(LED)
    finding = _finding_for(analysis, _cited(analysis, SEC3))
    sec3 = store.get_by_is_number(SEC3)[0]
    sec1 = store.get_by_is_number("IS 10322 : Part 5 : Sec 1")[0]
    assert sec1.qco_issuing_ministry != sec3.qco_issuing_ministry   # the test means something

    cert = finding.dimensions["certification"]
    assert cert["qco_notified"] is True
    assert cert["issuing_ministry"] == sec3.qco_issuing_ministry
    assert cert["effective_date"] == sec3.qco_effective_date.isoformat()


def test_compliance_only_path_recognises_the_part(store, monkeypatch) -> None:
    from kshiraj.aiml_client.client import AimlClient

    async def _unavailable(self, request):
        raise AnalysisError("forced", code="FORCED")

    monkeypatch.setattr(AimlClient, "run_analysis", _unavailable)
    analysis = _run(LED)
    finding = _finding_for(analysis, _cited(analysis, SEC3))

    assert analysis.metadata["analysis_mode"] == "fallback"
    assert [s.is_number for s in finding.applicable_standards] == [SEC3]
    assert finding.verdict == Verdict.OUTDATED_REFERENCE   # 2012 cited, later edition in catalogue


def test_second_citation_in_a_clause_is_still_deduplicated() -> None:
    """Unchanged here: co-citation handling is a separate fix."""
    from kartikey.analysis.requirement_extractor import extract_requirements

    requirements, _ = extract_requirements(
        "a1", "Luminaires shall conform to IS 10322 (Part 5/Sec 3) and IS 16107.",
    )

    assert [r.is_reference for r in requirements if r.is_reference] == [SEC3]
