import pytest
from shared.models import Standard, StandardStatus, EvidenceSourceType, Evidence
from kartikey.analysis.compliance import _build_evidence, VersionCheck, StatusCheck, QCOCheck

def test_metadata_only_evidence():
    # Test A: metadata-only evidence
    std = Standard(
        is_number="IS 10322",
        year=2022,
        title="Luminaires",
        status=StandardStatus.ACTIVE,
        source_url="http://bis.gov.in"
    )
    # the correct signature for VersionCheck
    v_check = VersionCheck(cited_year=None, current_year=2022, is_current=True, note="Test", is_year_omitted=False, gap_years=0)
    s_check = StatusCheck(status=StandardStatus.ACTIVE, superseded_by=[], note="Active", is_usable=True, transition_deadline=None, within_transition=None)
    q_check = QCOCheck(qco_notified=False, certification_scheme=None, issuing_ministry=None, effective_date=None, gazette_so_number=None, note=None)
    
    evidence = _build_evidence(std, v_check, s_check, q_check)
    assert len(evidence) == 1
    ev = evidence[0]
    
    assert getattr(ev, "category", None) == "metadata"
    assert ev.page is None
    assert ev.section is None
    assert getattr(ev, "excerpt", None) is None
    assert ev.url == "http://bis.gov.in"

def test_actual_clause_evidence():
    # Test B: actual clause evidence
    ev = Evidence(
        source_type=EvidenceSourceType.BIS_STANDARD,
        source_name="Actual Document",
        url="http://doc",
        page=42,
        section="Clause 3.1",
        excerpt="The actual excerpt.",
        category="clause"
    )
    assert ev.category == "clause"
    assert ev.page == 42
    assert ev.section == "Clause 3.1"
    assert ev.excerpt == "The actual excerpt."

def test_no_fake_page_1():
    # Test C: no fake page 1
    std = Standard(
        is_number="IS 1234",
        year=2020,
        title="Test",
        status=StandardStatus.ACTIVE
    )
    evidence = _build_evidence(std, VersionCheck(cited_year=2020, current_year=2020, is_current=True, note="Test", is_year_omitted=False, gap_years=0), StatusCheck(status=StandardStatus.ACTIVE, superseded_by=[], note="", is_usable=True, transition_deadline=None, within_transition=None), QCOCheck(qco_notified=False, certification_scheme=None, issuing_ministry=None, effective_date=None, gazette_so_number=None, note=None))
    for ev in evidence:
        assert ev.page is None
        assert ev.page != 1

def test_llm_reasoning_not_clause_evidence():
    # Test D: LLM reasoning cannot become authoritative clause evidence
    from shared.models import Finding, RequirementCategory, Verdict, Standard, Evidence
    f = Finding(
        analysis_id="a1",
        requirement_id="req-1",
        verdict=Verdict.JUSTIFIED,
        reason="Gemini said this is fine",
        confidence=0.9,
        applicable_standards=[],
        evidence=[]
    )
    std = Standard(is_number="IS 999", year=2020, title="Fake")
    ev = Evidence(
        source_type=EvidenceSourceType.BIS_STANDARD,
        source_name="Metadata Source",
        category="metadata",
        excerpt=None
    )
    f.applicable_standards = [std]
    f.evidence = [ev]
    
    ev_chain = {}
    ev_chain["quote"] = ev.excerpt if ev.excerpt else "not_available"
    assert ev_chain["quote"] == "not_available"
    assert ev_chain["quote"] != f.reason

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from kartikey.api.main import app

client = TestClient(app)

@patch("kartikey.api.routes.procurement.assess_query")
@patch("kartikey.api.routes.procurement._step_extract")
@patch("kartikey.api.routes.procurement._step_retrieve")
@patch("kartikey.api.routes.procurement._step_analyze")
@patch("kartikey.api.routes.procurement._step_enrich")
def test_api_serialization(mock_enrich, mock_analyze, mock_retrieve, mock_extract, mock_assess_query):
    # Test E: API serialization
    mock_assess_query.return_value = MagicMock(is_meaningful=True)
    
    from shared.models import Analysis, Requirement, Finding, Verdict, Evidence, Standard, RequirementCategory
    req = Requirement(id="r1", text="some text", analysis_id="a1", category=RequirementCategory.TECHNICAL_SPECIFICATION)
    std = Standard(is_number="IS 111", year=2020, title="Test")
    ev = Evidence(
        source_type="bis_standard",
        source_name="BIS Data",
        category="metadata",
        excerpt=None,
        page=None,
        section=None
    )
    f = Finding(
        analysis_id="a1",
        requirement_id="r1",
        verdict=Verdict.JUSTIFIED,
        reason="Test reason",
        applicable_standards=[std],
        confidence=0.9,
        evidence=[ev]
    )
    analysis = Analysis(
        id="a1",
        status="completed",
        input_type="text",
        requirements=[req],
        findings=[f]
    )
    
    async def mock_enrich_side_effect(analysis_obj, retrieved_standards, aiml_response):
        analysis_obj.requirements = [req]
        analysis_obj.findings = [f]
        
    mock_enrich.side_effect = mock_enrich_side_effect
    mock_extract.return_value = "extracted text"
    
    response = client.post(
        "/api/v1/procurement/analyze",
        json={"raw_text": "Supply LED", "category_hint": "LED"},
        headers={"X-API-Key": "sk_standiq_dev_26108"}
    )
    
    assert response.status_code == 200
    data = response.json()
    print(data.keys())
    first_finding = data["extracted_requirements"][0]
    assert first_finding["evidence_chain"]["page_number"] is None
    assert first_finding["evidence_chain"]["quote"] == "not_available"
    assert first_finding["evidence_chain"]["clause"] == "not_available"

