"""
kartikey/analysis/test_findings.py

Tests for the findings assembler — specifically the two things the enrichment
work changed.

Step 8 (combine dimensions instead of overwriting):
    `_merge_verdicts` ranks verdicts by severity to pick one headline. But the
    two sources do not answer the same question. The deterministic rules read
    years and status flags, so they can only speak to currentness; whether a
    clause narrows the vendor pool is the AI's axis alone. In
    `_VERDICT_SEVERITY_ORDER`, OUTDATED_REFERENCE outranks
    POTENTIALLY_OVER_RESTRICTIVE — so a clause that was *both* came back as
    merely "outdated". Fix the year and the tender reads clean, while the
    restriction that actually excluded suppliers is never mentioned. That is the
    regression pinned in `TestBothAxesSurvive`.

Step 9 (cross-reference integration):
    A tender that cites IS 16107 but never mentions the IS 10322 it normatively
    references is incomplete even though every citation in it is individually
    valid. The fixtures below use real reference chains from
    ai-engine/data/bis_full_knowledge_base.json rather than invented ones.
"""

from __future__ import annotations

import pytest

from shared.contracts import AimlFinding, AimlResponse
from shared.models import (
    Analysis,
    CertificationScheme,
    Evidence,
    EvidenceSourceType,
    InputType,
    Requirement,
    Standard,
    StandardStatus,
    Verdict,
)

from kartikey.analysis.findings import (
    _build_cross_references,
    _currentness_label,
    _tender_cited_is_numbers,
    assemble_findings,
)
from kartikey.analysis.compliance import run_compliance_checks


# ===========================================================================
# Fixtures
# ===========================================================================

def _analysis(*requirements: Requirement) -> Analysis:
    analysis = Analysis(input_type=InputType.TEXT, raw_text="tender")
    for req in requirements:
        req.analysis_id = analysis.id
    analysis.requirements = list(requirements)
    return analysis


def _req(
    text: str = "Luminaires shall conform to IS 10322:2012.",
    is_reference: str | None = "IS 10322",
    cited_year: int | None = 2012,
) -> Requirement:
    return Requirement(
        analysis_id="placeholder",
        text=text,
        is_reference=is_reference,
        cited_year=cited_year,
    )


def _std(
    is_number: str = "IS 10322",
    year: int | None = 2022,
    title: str = "Luminaires",
    **kwargs,
) -> Standard:
    return Standard(
        is_number=is_number,
        year=year,
        title=title,
        status=kwargs.pop("status", StandardStatus.ACTIVE),
        **kwargs,
    )


def _aiml(
    req: Requirement,
    verdict: str,
    confidence: float = 0.72,
    reason: str = "The 120 lm/W floor excludes most compliant suppliers.",
    standard_ids: list[str] | None = None,
    evidence_ids: list[str] | None = None,
) -> AimlResponse:
    return AimlResponse(
        analysis_id=req.analysis_id,
        findings=[
            AimlFinding(
                finding_id="f1",
                requirement_id=req.id,
                verdict=verdict,
                reason=reason,
                confidence=confidence,
                applicable_standard_ids=standard_ids or [],
                evidence_ids=evidence_ids or [],
            )
        ],
    )


def _run(analysis, standards, aiml=None):
    return assemble_findings(
        analysis=analysis,
        retrieved_standards=standards,
        aiml_response=aiml,
        standards_lookup={s.id: s for s in standards},
        evidence_lookup={},
    )


# ===========================================================================
# Step 8 — both axes survive the merge
# ===========================================================================

class TestBothAxesSurvive:
    """
    The regression: a clause that is both over-restrictive and cites a stale
    edition must report both, not only the more severe one.
    """

    def _both_problems(self):
        req = _req(cited_year=2012)
        analysis = _analysis(req)
        std = _std(year=2022)
        aiml = _aiml(
            req, Verdict.POTENTIALLY_OVER_RESTRICTIVE.value,
            confidence=0.72, standard_ids=[std.id],
        )
        return _run(analysis, [std], aiml)[0]

    def test_headline_is_the_more_severe_verdict(self):
        finding = self._both_problems()

        assert finding.verdict == Verdict.OUTDATED_REFERENCE
        assert finding.dimensions["headline"]["axis"] == "currentness"
        assert finding.dimensions["headline"]["resolution"] == "compliance_override"

    def test_the_ai_verdict_is_not_erased(self):
        """This is the whole point of Step 8."""
        finding = self._both_problems()
        applicability = finding.dimensions["applicability"]

        assert applicability["verdict"] == Verdict.POTENTIALLY_OVER_RESTRICTIVE.value
        assert applicability["confidence"] == 0.72
        assert applicability["source"] == "ai"

    def test_currentness_dimension_carries_the_numbers(self):
        currentness = self._both_problems().dimensions["currentness"]

        assert currentness["label"] == "OUTDATED"
        assert currentness["cited_year"] == 2012
        assert currentness["current_year"] == 2022
        assert currentness["gap_years"] == 10
        assert currentness["is_current"] is False
        assert currentness["source"] == "version_checker"

    def test_headline_note_says_the_other_axis_still_applies(self):
        note = self._both_problems().dimensions["headline"]["note"]

        assert "still applies" in note
        assert Verdict.POTENTIALLY_OVER_RESTRICTIVE.value in note

    def test_the_reason_prose_names_the_surviving_finding(self):
        """
        A reader of the PDF must see both problems. dimensions is JSON the
        frontend reads; the reason is what a procurement officer reads.
        """
        reason = self._both_problems().reason

        assert "[Applicability]" in reason
        assert Verdict.POTENTIALLY_OVER_RESTRICTIVE.value in reason
        assert "[Compliance override]" in reason

    def test_same_axis_override_does_not_claim_a_second_finding(self):
        """
        AI says outdated, compliance says withdrawn — both currentness. There is
        no independent second finding to preserve, so the note must not imply one.
        """
        req = _req(cited_year=2012)
        analysis = _analysis(req)
        std = _std(year=2022, status=StandardStatus.WITHDRAWN)
        aiml = _aiml(
            req, Verdict.OUTDATED_REFERENCE.value, standard_ids=[std.id],
        )

        finding = _run(analysis, [std], aiml)[0]

        assert finding.verdict == Verdict.INCORRECT_STANDARD
        assert "still applies" not in finding.dimensions["headline"]["note"]
        assert "[Applicability]" not in finding.reason

    def test_ai_wins_when_it_is_the_more_severe_axis(self):
        """A current citation with a restrictive clause keeps the AI headline."""
        req = _req(cited_year=2022)
        analysis = _analysis(req)
        std = _std(year=2022)
        aiml = _aiml(
            req, Verdict.WRONG_SCOPE.value, confidence=0.66, standard_ids=[std.id],
        )

        finding = _run(analysis, [std], aiml)[0]

        assert finding.verdict == Verdict.WRONG_SCOPE
        assert finding.dimensions["headline"]["resolution"] == "ai_primary"
        # Currentness is still reported — it was checked and it was clean.
        assert finding.dimensions["currentness"]["label"] == "CURRENT"
        assert "note" not in finding.dimensions["headline"]

    def test_certification_dimension_reports_qco_facts(self):
        req = _req(cited_year=2022)
        analysis = _analysis(req)
        std = _std(
            year=2022,
            qco_notified=True,
            qco_issuing_ministry="DPIIT",
            required_certification_scheme=CertificationScheme.ISI_MARK,
        )
        aiml = _aiml(req, Verdict.JUSTIFIED.value, standard_ids=[std.id])

        certification = _run(analysis, [std], aiml)[0].dimensions["certification"]

        assert certification["qco_notified"] is True
        assert certification["scheme"] == CertificationScheme.ISI_MARK.value
        assert certification["issuing_ministry"] == "DPIIT"


class TestComplianceOnlyPath:
    """
    Without AI, applicability was never assessed. Saying so is different from
    reporting a verdict for it.
    """

    def test_applicability_is_reported_as_not_assessed(self):
        req = _req(cited_year=2012)
        analysis = _analysis(req)
        std = _std(year=2022)

        finding = _run(analysis, [std], aiml=None)[0]
        applicability = finding.dimensions["applicability"]

        assert applicability["source"] == "not_assessed"
        assert applicability["verdict"] is None
        assert "No AI/ML analysis" in applicability["note"]

    def test_currentness_still_decided_and_headline_marked(self):
        req = _req(cited_year=2012)
        analysis = _analysis(req)
        std = _std(year=2022)

        finding = _run(analysis, [std], aiml=None)[0]

        assert finding.verdict == Verdict.OUTDATED_REFERENCE
        assert finding.dimensions["currentness"]["label"] == "OUTDATED"
        assert finding.dimensions["headline"]["resolution"] == "compliance_only"


class TestCurrentnessLabel:
    """
    The label drives a UI badge, so each situation an officer would act on
    differently needs its own value.
    """

    @pytest.mark.parametrize(
        "cited,current,status,expected",
        [
            (2022, 2022, StandardStatus.ACTIVE, "CURRENT"),
            (2012, 2022, StandardStatus.ACTIVE, "OUTDATED"),
            (None, 2022, StandardStatus.ACTIVE, "YEAR_OMITTED"),
            (2012, None, StandardStatus.ACTIVE, "UNVERIFIABLE"),
            (2035, 2022, StandardStatus.ACTIVE, "AHEAD_OF_CATALOGUE"),
            (2022, 2022, StandardStatus.WITHDRAWN, "WITHDRAWN"),
            (2022, 2022, StandardStatus.SUPERSEDED, "SUPERSEDED"),
        ],
    )
    def test_label(self, cited, current, status, expected):
        result = run_compliance_checks(
            _req(cited_year=cited), _std(year=current, status=status),
        )

        assert _currentness_label(result) == expected

    def test_no_compliance_result_is_unknown_not_current(self):
        assert _currentness_label(None) == "UNKNOWN"


# ===========================================================================
# Step 9 — cross-reference integration
# ===========================================================================

# Real chains from ai-engine/data/bis_full_knowledge_base.json. Invented ones
# would prove the code runs; these prove it runs on the shape the catalogue uses.
_IS_16107 = {
    "is_number": "IS 16107",
    "title": "Luminaire performance",
    "year": 2012,
    "normative_references": ["IS 10322 : Part 1", "IS 1944"],
}
_IS_1554_P1 = {
    "is_number": "IS 1554",
    "part": "Part 1",
    "title": "PVC insulated cables",
    "year": 1988,
    "normative_references": ["IS 694", "IS 8130"],
}


class TestTenderCitedIsNumbers:
    def test_prose_citations_count_not_just_the_structured_field(self):
        """
        A dependency named in a requirement's text but not chosen as its primary
        is_reference is still cited. Reporting it missing is a false alarm.
        """
        analysis = _analysis(
            _req(
                text="Poles per IS 2062 and fixtures per IS 10322.",
                is_reference="IS 10322",
            )
        )

        cited = _tender_cited_is_numbers(analysis)

        assert "IS 2062" in cited
        assert "IS 10322" in cited

    def test_no_requirements_gives_an_empty_set(self):
        assert _tender_cited_is_numbers(_analysis()) == set()


class TestBuildCrossReferences:
    def test_normative_references_are_extracted_from_the_catalogue_field(self):
        entries = _build_cross_references([_std(**_IS_16107)], tender_cited={"IS 16107"})

        assert len(entries) == 1
        assert set(entries[0]["references"]) == {"IS 10322", "IS 1944"}
        assert entries[0]["source"] == "IS 16107"

    def test_uncited_dependencies_are_reported_unmet(self):
        entries = _build_cross_references([_std(**_IS_16107)], tender_cited={"IS 16107"})

        assert set(entries[0]["unmet"]) == {"IS 10322", "IS 1944"}
        assert "does not cite" in entries[0]["note"]

    def test_a_dependency_the_tender_cites_is_not_unmet(self):
        entries = _build_cross_references(
            [_std(**_IS_16107)],
            tender_cited={"IS 16107", "IS 10322", "IS 1944"},
        )

        assert entries[0]["unmet"] == []
        assert "All of these are cited" in entries[0]["note"]

    def test_partial_coverage_reports_only_what_is_missing(self):
        entries = _build_cross_references(
            [_std(**_IS_1554_P1)], tender_cited={"IS 1554", "IS 694"},
        )

        assert entries[0]["unmet"] == ["IS 8130"]

    def test_scope_prose_is_scanned_too(self):
        std = _std(
            is_number="IS 15885",
            scope="Tests shall be as per IS 10322 and IEC 62031.",
        )

        entries = _build_cross_references([std], tender_cited=set())

        assert entries[0]["references"] == ["IS 10322"]

    def test_a_sibling_part_is_not_a_dependency(self):
        """
        Extraction is at base-IS-number granularity, so IS 10322 Part 5 citing
        IS 10322 Part 1 normalises to a self-reference. Citing any part of
        IS 10322 does cite IS 10322 — reporting it unmet would be wrong.
        """
        std = _std(
            is_number="IS 10322",
            part="Part 5",
            section="Sec 3",
            normative_references=["IS 10322 : Part 1"],
        )

        assert _build_cross_references([std], tender_cited=set()) == []

    def test_a_standard_with_no_references_produces_no_entry(self):
        assert _build_cross_references([_std()], tender_cited=set()) == []

    def test_inputs_are_not_mutated(self):
        std = _std(**_IS_16107)
        before = std.model_dump()

        _build_cross_references([std], tender_cited=set())

        assert std.model_dump() == before


class TestCrossReferencesReachTheFinding:
    """
    Wiring, not extraction: the dependency must appear on the Finding, in the
    prose, and in the action — not only in the helper's return value.
    """

    def _finding_with_unmet_dependency(self, aiml_verdict=Verdict.JUSTIFIED.value):
        req = _req(
            text="Luminaires shall conform to IS 16107:2012.",
            is_reference="IS 16107",
            cited_year=2012,
        )
        analysis = _analysis(req)
        std = _std(**_IS_16107)
        aiml = _aiml(req, aiml_verdict, standard_ids=[std.id])
        return _run(analysis, [std], aiml)[0]

    def test_cross_references_are_attached_to_the_finding(self):
        finding = self._finding_with_unmet_dependency()

        assert len(finding.cross_references) == 1
        assert set(finding.cross_references[0]["unmet"]) == {"IS 10322", "IS 1944"}

    def test_unmet_dependencies_appear_in_the_reason(self):
        reason = self._finding_with_unmet_dependency().reason

        assert "[Dependencies]" in reason
        assert "IS 10322" in reason
        assert "IS 1944" in reason

    def test_unmet_dependencies_appear_in_the_recommended_action(self):
        """
        A JUSTIFIED citation whose prerequisites are missing still leaves the
        spec incomplete, so the action must not be "no action required" alone.
        """
        action = self._finding_with_unmet_dependency().recommended_action

        assert "No action required" in action
        assert "Also add the referenced standard(s)" in action
        assert "IS 10322" in action

    def test_the_compliance_only_path_reports_them_too(self):
        req = _req(
            text="Luminaires shall conform to IS 16107:2012.",
            is_reference="IS 16107",
            cited_year=2012,
        )
        analysis = _analysis(req)
        std = _std(**_IS_16107)

        finding = _run(analysis, [std], aiml=None)[0]

        assert set(finding.cross_references[0]["unmet"]) == {"IS 10322", "IS 1944"}
        assert "[Dependencies]" in finding.reason

    def test_a_tender_that_cites_its_dependencies_gets_no_dependency_noise(self):
        reqs = [
            _req(
                text="Luminaires shall conform to IS 16107:2012.",
                is_reference="IS 16107", cited_year=2012,
            ),
            _req(text="Photometry per IS 10322.", is_reference="IS 10322"),
            _req(text="Galvanizing per IS 1944.", is_reference="IS 1944"),
        ]
        analysis = _analysis(*reqs)
        std = _std(**_IS_16107)
        aiml = _aiml(reqs[0], Verdict.JUSTIFIED.value, standard_ids=[std.id])

        finding = _run(analysis, [std], aiml)[0]

        assert finding.cross_references[0]["unmet"] == []
        assert "[Dependencies]" not in finding.reason
        assert "Also add the referenced" not in finding.recommended_action


# ===========================================================================
# Regression guards on the existing contract
# ===========================================================================

class TestExistingBehaviourPreserved:
    def test_no_requirements_produces_no_findings(self):
        assert _run(_analysis(), [], aiml=None) == []

    def test_ai_evidence_ids_that_resolve_to_nothing_are_dropped(self):
        """The anti-hallucination guardrail: unresolvable IDs must not invent evidence."""
        req = _req(cited_year=2022)
        analysis = _analysis(req)
        std = _std(year=2022)
        aiml = _aiml(
            req, Verdict.JUSTIFIED.value,
            standard_ids=[std.id], evidence_ids=["does-not-exist"],
        )

        finding = _run(analysis, [std], aiml)[0]

        assert all(ev.id != "does-not-exist" for ev in finding.evidence)

    def test_resolved_evidence_is_carried_through(self):
        req = _req(cited_year=2022)
        analysis = _analysis(req)
        std = _std(year=2022)
        ev = Evidence(
            source_type=EvidenceSourceType.BIS_STANDARD,
            source_name="IS 10322 (Part 5/Sec 3):2022",
            excerpt="Clause 5.2 — photometric requirements.",
        )
        aiml = _aiml(
            req, Verdict.JUSTIFIED.value,
            standard_ids=[std.id], evidence_ids=[ev.id],
        )

        finding = assemble_findings(
            analysis=analysis,
            retrieved_standards=[std],
            aiml_response=aiml,
            standards_lookup={std.id: std},
            evidence_lookup={ev.id: ev},
        )[0]

        assert ev.id in {e.id for e in finding.evidence}

    def test_requirement_without_an_is_reference_is_undeterminable(self):
        analysis = _analysis(
            _req(text="Warranty shall be 5 years.", is_reference=None, cited_year=None)
        )

        finding = _run(analysis, [], aiml=None)[0]

        assert finding.verdict == Verdict.UNABLE_TO_DETERMINE
        assert finding.requires_human_verification is True
        assert finding.cross_references == []

    def test_cited_standard_absent_from_the_knowledge_base(self):
        analysis = _analysis(_req(is_reference="IS 99999", cited_year=2012))

        finding = _run(analysis, [], aiml=None)[0]

        assert finding.verdict == Verdict.UNABLE_TO_DETERMINE
        assert "not found" in finding.reason
        assert finding.dimensions is None   # no compliance result to build from

    def test_unknown_verdict_from_the_ai_degrades_instead_of_raising(self):
        req = _req(cited_year=2022)
        analysis = _analysis(req)
        std = _std(year=2022)
        aiml = _aiml(req, "not_a_real_verdict", standard_ids=[std.id])

        finding = _run(analysis, [std], aiml)[0]

        assert finding.dimensions["applicability"]["verdict"] == (
            Verdict.UNABLE_TO_DETERMINE.value
        )

    def test_aiml_finding_for_an_unknown_requirement_is_skipped(self):
        req = _req()
        analysis = _analysis(req)
        stray = AimlResponse(
            analysis_id=analysis.id,
            findings=[
                AimlFinding(
                    finding_id="f1", requirement_id="no-such-requirement",
                    verdict=Verdict.JUSTIFIED.value, reason="x", confidence=0.9,
                )
            ],
        )

        assert _run(analysis, [], stray) == []

    def test_compliance_override_populates_applicable_standards(self):
        req = _req(is_reference="IS 16107")
        analysis = _analysis(req)
        
        std = Standard(
            id="s1",
            is_number="IS 16107",
            year=2013,
            designation="IS 16107:2013",
            title="LED Street Lighting",
            status=StandardStatus.ACTIVE,
        )
        
        resp = AimlResponse(
            analysis_id=analysis.id,
            findings=[
                AimlFinding(
                    finding_id="f1",
                    requirement_id=req.id,
                    verdict=Verdict.UNABLE_TO_DETERMINE.value,
                    reason="Low confidence",
                    confidence=0.43,
                    applicable_standard_ids=["s1"],
                    evidence_ids=[],
                )
            ]
        )
        
        findings = _run(analysis, [std], resp)
        
        assert len(findings) == 1
        f = findings[0]
        
        assert f.verdict == Verdict.OUTDATED_REFERENCE
        assert "[Compliance override]" in f.reason
        
        assert len(f.applicable_standards) == 1
        assert f.applicable_standards[0].id == "s1"
        assert "s1" not in [s.id for s in f.cited_standards]

    def test_compliance_override_picks_most_severe_standard(self):
        req = _req(is_reference="IS 1")
        analysis = _analysis(req)
        
        # 1. AMBIGUOUS (index 8)
        std_ambiguous = Standard(
            id="s_ambig",
            is_number="IS 1",
            year=None,
            designation="IS 1",
            title="S1",
            status=StandardStatus.ACTIVE,
        )
        
        # 2. OUTDATED_REFERENCE (index 2)
        std_outdated = Standard(
            id="s_outdated",
            is_number="IS 2",
            year=2000,
            designation="IS 2:2000",
            title="S2",
            status=StandardStatus.SUPERSEDED,
        )
        
        resp = AimlResponse(
            analysis_id=analysis.id,
            findings=[
                AimlFinding(
                    finding_id="f1",
                    requirement_id=req.id,
                    verdict=Verdict.UNABLE_TO_DETERMINE.value,
                    reason="Low confidence",
                    confidence=0.43,
                    applicable_standard_ids=["s_ambig", "s_outdated"],
                    evidence_ids=[],
                )
            ]
        )
        
        findings = _run(analysis, [std_ambiguous, std_outdated], resp)
        
        assert len(findings) == 1
        f = findings[0]
        
        assert f.verdict == Verdict.OUTDATED_REFERENCE
        assert "[Compliance override]" in f.reason
        
        # The override should be driven by the superseded standard
        assert "superseded" in f.reason.lower() or "withdrawn" in f.reason.lower()

    def test_low_ml_confidence_clears_standards_and_sets_wrong_scope(self):
        req = _req(is_reference="IS 9999")
        analysis = _analysis(req)
        
        std1 = Standard(
            id="s1",
            is_number="IS 12824",
            year=1989,
            designation="IS 12824:1989",
            title="Irrelevant Standard",
            status=StandardStatus.ACTIVE,
        )
        std2 = Standard(
            id="s2",
            is_number="SP 62",
            year=1997,
            designation="SP 62:1997",
            title="Another Irrelevant Standard",
            status=StandardStatus.ACTIVE,
        )
        
        resp = AimlResponse(
            analysis_id=analysis.id,
            findings=[
                AimlFinding(
                    finding_id="f1",
                    requirement_id=req.id,
                    verdict=Verdict.AMBIGUOUS.value,
                    reason="I guess maybe one of these?",
                    confidence=0.1,  # < 0.2
                    applicable_standard_ids=["s1", "s2"],
                    evidence_ids=[],
                )
            ]
        )
        
        findings = _run(analysis, [std1, std2], resp)
        
        assert len(findings) == 1
        f = findings[0]
        
        assert f.verdict == Verdict.WRONG_SCOPE
        assert f.applicable_standards == []
        assert f.cited_standards == []
        assert "No relevant standard could be confidently matched" in f.reason

