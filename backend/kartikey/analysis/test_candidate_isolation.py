"""
kartikey/analysis/test_candidate_isolation.py

The requirement → candidate-standard guardrail in `_assemble_with_aiml`.

The AI/ML component is shown one pooled list of retrieved standards for every
requirement, so the IDs it returns for one requirement can belong to standards
retrieved for another. When `requirement_candidates` is supplied, a standard
may be attached to a requirement only if it was retrieved for that requirement.
"""

from __future__ import annotations

import logging

from shared.contracts import AimlFinding, AimlResponse
from shared.models import (
    Analysis,
    InputType,
    Requirement,
    Standard,
    StandardStatus,
    Verdict,
)

from kartikey.analysis.findings import assemble_findings


# ===========================================================================
# Fixtures
# ===========================================================================

def _std(is_number: str) -> Standard:
    return Standard(
        is_number=is_number,
        year=2020,
        title=f"Standard {is_number}",
        status=StandardStatus.ACTIVE,
    )


def _req(text: str) -> Requirement:
    # No IS citation and no material: keeps compliance/scope checks from
    # reshuffling standards between applicable and cited.
    return Requirement(analysis_id="placeholder", text=text)


def _analysis(*requirements: Requirement) -> Analysis:
    analysis = Analysis(input_type=InputType.TEXT, raw_text="tender")
    for req in requirements:
        req.analysis_id = analysis.id
    analysis.requirements = list(requirements)
    return analysis


def _finding(req: Requirement, standard_ids: list[str]) -> AimlFinding:
    return AimlFinding(
        finding_id=f"f-{req.id[:8]}",
        requirement_id=req.id,
        verdict=Verdict.JUSTIFIED.value,
        reason="stub",
        confidence=0.9,
        applicable_standard_ids=standard_ids,
    )


def _run(analysis, standards, aiml_findings, requirement_candidates):
    return assemble_findings(
        analysis=analysis,
        retrieved_standards=standards,
        aiml_response=AimlResponse(analysis_id=analysis.id, findings=aiml_findings),
        standards_lookup={s.id: s for s in standards},
        evidence_lookup={},
        requirement_candidates=requirement_candidates,
    )


def _attached_ids(finding) -> set[str]:
    """Every standard attached to a finding, applicable or cited."""
    return {s.id for s in finding.applicable_standards} | {
        s.id for s in finding.cited_standards
    }


# Four standards and two requirements with disjoint candidate sets:
#   requirement A → {IS1, IS2}, requirement B → {IS3, IS4}
IS1, IS2, IS3, IS4 = (_std(n) for n in ("IS 1001", "IS 1002", "IS 1003", "IS 1004"))
ALL = [IS1, IS2, IS3, IS4]


def _two_requirements():
    req_a = _req("Requirement A text.")
    req_b = _req("Requirement B text.")
    analysis = _analysis(req_a, req_b)
    candidates = {
        req_a.id: {IS1.id: 1.0, IS2.id: 0.8},
        req_b.id: {IS3.id: 0.9, IS4.id: 0.7},
    }
    return analysis, req_a, req_b, candidates


# ===========================================================================
# Guardrail
# ===========================================================================

class TestCandidateGuardrail:

    def test_accepts_standard_from_own_candidates(self) -> None:
        """1. A candidates {IS1, IS2}; AI returns IS2 → applicable = {IS2}."""
        analysis, req_a, _, candidates = _two_requirements()

        findings = _run(analysis, ALL, [_finding(req_a, [IS2.id])], candidates)

        assert len(findings) == 1
        assert [s.id for s in findings[0].applicable_standards] == [IS2.id]
        assert _attached_ids(findings[0]) == {IS2.id}

    def test_rejects_standard_from_another_requirement(self, caplog) -> None:
        """2. B candidates {IS3, IS4}; AI returns IS2 (A's) → rejected."""
        analysis, _, req_b, candidates = _two_requirements()

        with caplog.at_level(logging.WARNING):
            findings = _run(analysis, ALL, [_finding(req_b, [IS2.id])], candidates)

        assert _attached_ids(findings[0]) == set()
        assert "belongs_to_other_requirement" in caplog.text
        assert IS2.id in caplog.text

    def test_rejects_unknown_standard_id(self, caplog) -> None:
        """3. An ID that resolves to nothing → rejected."""
        analysis, req_a, _, candidates = _two_requirements()

        with caplog.at_level(logging.WARNING):
            findings = _run(
                analysis, ALL, [_finding(req_a, ["no-such-standard"])], candidates,
            )

        assert _attached_ids(findings[0]) == set()
        assert "unknown_standard_id" in caplog.text

    def test_unknown_id_listed_as_candidate_is_still_rejected(self) -> None:
        """A candidate ID the lookup cannot resolve is not invented."""
        analysis, req_a, _, candidates = _two_requirements()
        candidates[req_a.id]["ghost-id"] = 0.5

        findings = _run(analysis, ALL, [_finding(req_a, ["ghost-id", IS1.id])], candidates)

        assert [s.id for s in findings[0].applicable_standards] == [IS1.id]

    def test_empty_ai_result_stays_empty(self) -> None:
        """4. AI returns [] → applicable remains [], never back-filled."""
        analysis, req_a, _, candidates = _two_requirements()

        findings = _run(analysis, ALL, [_finding(req_a, [])], candidates)

        assert findings[0].applicable_standards == []
        assert findings[0].cited_standards == []

    def test_all_rejected_is_not_replaced_with_candidates(self) -> None:
        """A fully rejected proposal must not fall back to the candidate set."""
        analysis, req_a, _, candidates = _two_requirements()

        findings = _run(analysis, ALL, [_finding(req_a, [IS3.id, IS4.id])], candidates)

        assert _attached_ids(findings[0]) == set()

    def test_disjoint_requirements_never_share_ids(self) -> None:
        """5. AI proposes every standard for both → each keeps only its own."""
        analysis, req_a, req_b, candidates = _two_requirements()
        everything = [s.id for s in ALL]

        findings = _run(
            analysis, ALL,
            [_finding(req_a, everything), _finding(req_b, everything)],
            candidates,
        )
        by_req = {f.requirement_id: f for f in findings}

        assert _attached_ids(by_req[req_a.id]) == {IS1.id, IS2.id}
        assert _attached_ids(by_req[req_b.id]) == {IS3.id, IS4.id}
        assert not _attached_ids(by_req[req_a.id]) & _attached_ids(by_req[req_b.id])

    def test_requirement_missing_from_map_gets_nothing(self) -> None:
        """No recorded candidates means nothing may be attached."""
        analysis, req_a, _, candidates = _two_requirements()
        del candidates[req_a.id]

        findings = _run(analysis, ALL, [_finding(req_a, [IS1.id])], candidates)

        assert _attached_ids(findings[0]) == set()

    def test_duplicate_ids_attached_once(self) -> None:
        analysis, req_a, _, candidates = _two_requirements()

        findings = _run(analysis, ALL, [_finding(req_a, [IS1.id, IS1.id])], candidates)

        assert [s.id for s in findings[0].applicable_standards] == [IS1.id]

    def test_candidate_ids_may_be_given_as_list(self) -> None:
        """The map's values only need to be iterables of standard IDs."""
        analysis, req_a, req_b, _ = _two_requirements()
        candidates = {req_a.id: [IS1.id], req_b.id: [IS3.id]}

        findings = _run(analysis, ALL, [_finding(req_a, [IS1.id, IS3.id])], candidates)

        assert _attached_ids(findings[0]) == {IS1.id}


# ===========================================================================
# 6. Behaviour without a candidate map is unchanged
# ===========================================================================

class TestLegacyBehaviourPreserved:

    def test_no_map_keeps_unscoped_resolution(self) -> None:
        """Callers that pass no map (simulator, older analyses) are unaffected."""
        analysis, req_a, _, _ = _two_requirements()

        findings = _run(analysis, ALL, [_finding(req_a, [IS3.id, "nope"])], None)

        assert [s.id for s in findings[0].applicable_standards] == [IS3.id]

    def test_compliance_only_path_ignores_map(self) -> None:
        """Without AI/ML output the candidate map plays no part."""
        req = Requirement(
            analysis_id="placeholder",
            text="Shall conform to IS 1001:2020.",
            is_reference="IS 1001",
            cited_year=2020,
        )
        analysis = _analysis(req)

        findings = assemble_findings(
            analysis=analysis,
            retrieved_standards=ALL,
            aiml_response=None,
            standards_lookup={s.id: s for s in ALL},
            evidence_lookup={},
            requirement_candidates={req.id: set()},
        )

        assert [s.id for s in findings[0].applicable_standards] == [IS1.id]
