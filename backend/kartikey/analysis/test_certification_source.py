"""
kartikey/analysis/test_certification_source.py

Per-finding certification is read from the finding's final applicable
standards only — never from a standard the scope check found not to apply, and
never chosen by verdict severity.

Runs against the real reconciled catalogue, whose records carry:
  IS 694                      PVC cables    QCO, ISI mark
  IS 1554 : Part 1            PVC cables    QCO, ISI mark
  IS 7098 : Part 1            XLPE cables   QCO, ISI mark
  IS 7098 : Part 2            XLPE cables   no QCO
  IS 10322 : Part 1           luminaires    no QCO
  IS 10322 : Part 5 : Sec 3   street lights QCO, CRS
"""

from __future__ import annotations

import asyncio

import pytest

from shared.contracts import AimlFinding, AimlResponse
from shared.models import Analysis, InputType, Requirement, StandardStatus

from kartikey.analysis.findings import assemble_findings
from kartikey.orchestration.knowledge_registry import initialize_knowledge_registry

NO_APPLICABLE_NOTE = "No applicable standard identified; certification cannot be determined."

XLPE_CITING_1554 = "1.1 kV XLPE insulated armoured power cables conforming to IS 1554 : Part 1."
LED_TEXT = "LED street light luminaires conforming to IS 10322 (Part 5/Sec 3):2012."


@pytest.fixture(scope="module")
def store():
    return initialize_knowledge_registry().standards_store


@pytest.fixture(scope="module")
def std(store):
    def _get(is_number: str):
        return store.get_by_is_number(is_number)[0]
    return _get


def _assemble(store, text, selected, *, is_reference=None, cited_year=None,
              compliance_only=False, extra=()):
    lookup = {s.id: s for s in store.list_all()}
    for s in extra:
        lookup[s.id] = s
    analysis = Analysis(input_type=InputType.TEXT, raw_text=text)
    req = Requirement(
        analysis_id=analysis.id, text=text,
        is_reference=is_reference, cited_year=cited_year,
    )
    analysis.requirements = [req]
    response = None if compliance_only else AimlResponse(
        analysis_id=analysis.id,
        findings=[AimlFinding(
            finding_id="f", requirement_id=req.id, verdict="justified",
            reason="stub", confidence=0.9,
            applicable_standard_ids=[s.id for s in selected],
        )],
    )
    return assemble_findings(
        analysis=analysis,
        retrieved_standards=list(selected),
        aiml_response=response,
        standards_lookup=lookup,
        evidence_lookup={},
        requirement_candidates={req.id: [s.id for s in selected]},
    )[0]


def _cert(finding) -> dict:
    return finding.dimensions["certification"]


def _assert_currentness_agrees(finding) -> None:
    """G + H: currentness QCO fields come from the same source as the dimension."""
    cert = _cert(finding)
    assert finding.currentness["qco_notified"] == cert["qco_notified"]
    assert finding.currentness["qco_ministry"] == cert["issuing_ministry"]


# ===========================================================================
# A–F
# ===========================================================================

def test_a_cited_and_applicable_is_unchanged(store, std) -> None:
    is694 = std("IS 694")
    f = _assemble(
        store, "PVC insulated cables up to 1100 V conforming to IS 694:2010.",
        [is694], is_reference="IS 694", cited_year=2010,
    )

    assert [s.id for s in f.applicable_standards] == [is694.id]
    cert = _cert(f)
    assert cert["qco_notified"] is True
    assert cert["scheme"] == is694.required_certification_scheme.value
    assert cert["gazette_so_number"] == is694.qco_gazette_so_number
    assert cert["issuing_ministry"] == is694.qco_issuing_ministry
    _assert_currentness_agrees(f)


def test_b_cited_but_not_applicable_is_not_qco(store, std) -> None:
    is1554 = std("IS 1554 : Part 1")
    assert is1554.qco_notified   # the claim below is only meaningful if it is
    f = _assemble(
        store, XLPE_CITING_1554, [is1554], is_reference="IS 1554", cited_year=1988,
    )

    assert f.applicable_standards == []
    assert [s.id for s in f.cited_standards] == [is1554.id]
    cert = _cert(f)
    assert cert["qco_notified"] is False
    assert cert["scheme"] is None
    assert cert["gazette_so_number"] is None
    assert cert["issuing_ministry"] is None
    assert cert["effective_date"] is None
    assert cert["note"] == NO_APPLICABLE_NOTE
    _assert_currentness_agrees(f)


def test_b_compliance_only_path_cited_but_not_applicable(store, std) -> None:
    is1554 = std("IS 1554 : Part 1")
    f = _assemble(
        store, XLPE_CITING_1554, [is1554],
        is_reference="IS 1554 : Part 1", cited_year=1988, compliance_only=True,
    )

    assert f.applicable_standards == []
    assert [s.id for s in f.cited_standards] == [is1554.id]
    assert _cert(f)["qco_notified"] is False
    assert _cert(f)["note"] == NO_APPLICABLE_NOTE
    _assert_currentness_agrees(f)


def test_c_applicable_without_citation_is_considered(store, std) -> None:
    xlpe = std("IS 7098 : Part 1")
    f = _assemble(store, "1.1 kV XLPE insulated armoured power cables.", [xlpe])

    assert [s.id for s in f.applicable_standards] == [xlpe.id]
    cert = _cert(f)
    assert cert["qco_notified"] is True
    assert cert["gazette_so_number"] == xlpe.qco_gazette_so_number
    _assert_currentness_agrees(f)


@pytest.mark.parametrize("applicable_number", ["IS 7098 : Part 2", "IS 7098 : Part 1"])
def test_d_non_applicable_standard_has_zero_influence(store, std, applicable_number) -> None:
    """With and without the scope-mismatched IS 1554, certification is identical."""
    is1554, applicable = std("IS 1554 : Part 1"), std(applicable_number)

    both = _assemble(
        store, XLPE_CITING_1554, [is1554, applicable],
        is_reference="IS 1554", cited_year=1988,
    )
    alone = _assemble(
        store, XLPE_CITING_1554, [applicable],
        is_reference="IS 1554", cited_year=1988,
    )

    assert [s.id for s in both.applicable_standards] == [applicable.id]
    assert [s.id for s in both.cited_standards] == [is1554.id]
    assert _cert(both) == _cert(alone)
    assert _cert(both)["qco_notified"] is applicable.qco_notified
    _assert_currentness_agrees(both)


def test_e_order_of_applicable_standards_does_not_change_certification(store, std) -> None:
    part1, sec3 = std("IS 10322 : Part 1"), std("IS 10322 : Part 5 : Sec 3")
    assert not part1.qco_notified and sec3.qco_notified

    forward = _assemble(store, LED_TEXT, [part1, sec3], is_reference="IS 10322", cited_year=2012)
    reverse = _assemble(store, LED_TEXT, [sec3, part1], is_reference="IS 10322", cited_year=2012)

    assert {s.id for s in forward.applicable_standards} == {part1.id, sec3.id}
    assert _cert(forward)["qco_notified"] is True
    assert _cert(forward) == _cert(reverse)   # one QCO-notified → details identical too
    assert _cert(forward)["scheme"] == "crs"
    assert _cert(forward)["gazette_so_number"] == sec3.qco_gazette_so_number
    _assert_currentness_agrees(forward)
    _assert_currentness_agrees(reverse)


@pytest.mark.parametrize("order", ["qco_first", "superseded_first"])
def test_f_superseded_non_qco_applicable_does_not_hide_active_qco(store, std, order) -> None:
    sec3 = std("IS 10322 : Part 5 : Sec 3")
    superseded = std("IS 10322 : Part 1").model_copy(update={
        "id": "superseded-part1", "status": StandardStatus.SUPERSEDED,
        "superseded_by": "IS 10322 : Part 1",
    })
    selected = [sec3, superseded] if order == "qco_first" else [superseded, sec3]

    f = _assemble(
        store, LED_TEXT.replace(":2012", ":2026"), selected,
        is_reference="IS 10322", cited_year=2026, extra=[superseded],
    )

    # The superseded standard still drives the headline verdict …
    assert f.verdict.value == "outdated_reference"
    # … but not the certification.
    assert _cert(f)["qco_notified"] is True
    assert _cert(f)["scheme"] == "crs"
    _assert_currentness_agrees(f)


def test_multiple_qco_schemes_report_first_qco_notified_details(store, std) -> None:
    """Known limitation: one set of details. qco_notified is order-independent."""
    sec3, xlpe = std("IS 10322 : Part 5 : Sec 3"), std("IS 7098 : Part 1")
    text = "Equipment shall conform to the applicable Indian Standards."

    a = _assemble(store, text, [sec3, xlpe])
    b = _assemble(store, text, [xlpe, sec3])

    assert _cert(a)["qco_notified"] is _cert(b)["qco_notified"] is True
    assert _cert(a)["scheme"] == "crs"
    assert _cert(b)["scheme"] == "isi_mark"


# ===========================================================================
# I. /procurement/analyze agrees with the per-finding certification
# ===========================================================================

def test_i_procurement_qco_matches_finding_certification(store, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from kartikey.api.main import app
    from kartikey.api.routes import procurement

    monkeypatch.setattr("shared.config.settings.enable_bis_sync", False)
    captured = {}
    real_enrich = procurement._step_enrich

    async def _capture(analysis, retrieved, response):
        await real_enrich(analysis, retrieved, response)
        captured["analysis"] = analysis

    monkeypatch.setattr(procurement, "_step_enrich", _capture)

    text = (
        "Supply, installation, testing and commissioning of 90W LED street light "
        "luminaires for the municipal road lighting project. The luminaires shall "
        "conform to IS 10322 (Part 5/Sec 3) : 2012 and shall comply with the "
        "performance requirements of IS 16107. Minimum luminaire efficacy shall be "
        "110 lm/W with IP66 ingress protection."
    )
    from shared.config import settings
    response = TestClient(app).post(
        "/api/v1/procurement/analyze",
        json={"raw_text": text},
        headers={"X-API-Key": settings.api_key},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    by_req = {r["id"]: r for r in body["extracted_requirements"]}
    findings = captured["analysis"].findings
    assert findings and set(by_req) == {f.requirement_id for f in findings}
    for f in findings:
        assert by_req[f.requirement_id]["qco_notified"] == _cert(f)["qco_notified"]

    # The IS 10322 citation. Since the cited part is preserved it resolves to
    # Part 5 : Sec 3 and leads the applicable list; order independence of the
    # QCO flag itself is covered by test_e above.
    cited = next(f for f in findings if any(
        s.is_number == "IS 10322 : Part 5 : Sec 3" for s in f.applicable_standards
    ))
    assert cited.applicable_standards[0].is_number == "IS 10322 : Part 5 : Sec 3"
    assert by_req[cited.requirement_id]["qco_notified"] is True
