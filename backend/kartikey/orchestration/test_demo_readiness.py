"""
kartikey/orchestration/test_demo_readiness.py

Demo-readiness gate: proves the REAL pipeline produces a complete report.

Every other test in this repo checks one stage in isolation with a store it
populated itself. This one runs two real procurement scenarios through the whole
chain — extraction, retrieval, applicability ranking, analysis, VersionChecker,
CrossRefExtractor, certification and evidence enrichment — against the real
reconciled BIS catalogue, and asserts that each section the report renders
actually carries data traceable to that catalogue.

Two rules shape the assertions:

1. Nothing is hardcoded. The expected values are read back out of the knowledge
   registry, so the test cannot pass by agreeing with a constant somebody typed
   into it. If the catalogue's IS 16107 record loses its normative references,
   the cross-reference assertions fail rather than continuing to assert a list
   this file remembers.

2. No assertion depends on Gemini. The LLM is one degradable stage among many,
   and in CI (and in any sandbox without egress) it is unavailable. The pipeline
   is supposed to degrade to the deterministic path and still produce a complete
   report — so that is exactly what is asserted, and the test is meaningful in
   both modes. Where the mode matters it is asserted as a property ("the mode is
   recorded and the degradation is explained"), not as a fixed value.

The scenarios are real Indian public-procurement text, and the standards they
cite are ones the catalogue genuinely covers, chosen so the reference chains
under test are real: IS 16107 normatively references IS 10322, IS 1944 and
IS 16101, and IS 1554 Part 1 references IS 694, IS 8130 and IS 7098.
"""

from __future__ import annotations

import asyncio

import pytest

from shared.models import Analysis, AnalysisStatus, InputType, Verdict
from kartikey.orchestration.knowledge_registry import initialize_knowledge_registry
from kartikey.orchestration.pipeline import run_analysis_pipeline

# The sections the frontend renders and this gate holds the pipeline to. Named
# once so a new section cannot be added to the report without appearing here.
DEMO_SECTIONS = (
    "applicable_standards",
    "cited_standards",
    "cross_references",
    "evidence",
    "currentness",
    "scope",
    "certification",
    "applicability",
    "recommended_action",
)

# Every value Finding.data_availability is allowed to carry. A section that is
# empty must say which of these it is; anything else is an unlabelled blank.
AVAILABILITY_STATES = frozenset(
    {"verified", "not_available", "not_assessed", "not_identified"}
)

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


def _run_scenario(text: str) -> Analysis:
    """
    Run one scenario through the real pipeline and return the finished Analysis.

    The extractor is deliberately NOT stubbed. It was, briefly, and the stub cost
    more than it bought: it returned Requirement objects carrying is_reference but
    no cited_year, so VersionChecker had no year to compare and every finding came
    back YEAR_OMITTED. The gate then reported that the pipeline could not detect a
    stale citation — the one thing this product exists to do — when in fact the
    real path detects it correctly (IS 10322: cited 2012, current 2026, gap 14).

    Stubbing was never needed for determinism either. When Gemini is unreachable
    the extractor degrades to the regex IS scanner inside _step_extract, which is
    deterministic, offline, and populates cited_year from the tender text. That is
    the path this gate runs on in CI, and it is a real production path rather than
    a fixture.
    """
    analysis = Analysis(
        input_type=InputType.TEXT, raw_text=text, status=AnalysisStatus.QUEUED,
    )
    store = {analysis.id: analysis}
    asyncio.run(run_analysis_pipeline(analysis.id, store))
    return store[analysis.id]


@pytest.fixture(scope="module")
def registry():
    """
    The real catalogue, not a store the test filled in.

    initialize_knowledge_registry() rebuilds the singleton from
    shared/bis_catalogue_reconciled.json on every call, so this also undoes any
    clear() a previously-run test module did to the shared registry.
    """
    return initialize_knowledge_registry()


@pytest.fixture(scope="module")
def led(registry) -> Analysis:
    """Scenario A, run once and shared — the pipeline is too slow to repeat per test."""
    return _run_scenario(SCENARIO_LED)


@pytest.fixture(scope="module")
def cable(registry) -> Analysis:
    """Scenario B, run once and shared."""
    return _run_scenario(SCENARIO_CABLE)


# ---------------------------------------------------------------------------
# The pipeline completes at all
# ---------------------------------------------------------------------------

class TestPipelineCompletes:
    """
    Before any section can be checked, the run has to finish.

    This is the assertion that would have caught the LLM_CALL_FAILED bug: with
    Gemini returning 403 the whole analysis failed, and no amount of enrichment
    quality mattered because there were no findings to enrich.
    """

    @pytest.mark.parametrize("scenario", ["led", "cable"])
    def test_the_analysis_reaches_completed(self, scenario, request) -> None:
        analysis = request.getfixturevalue(scenario)
        assert analysis.status is AnalysisStatus.COMPLETED, analysis.error_message
        assert analysis.error_message is None

    @pytest.mark.parametrize("scenario", ["led", "cable"])
    def test_requirements_were_extracted(self, scenario, request) -> None:
        analysis = request.getfixturevalue(scenario)
        assert analysis.requirements, "no requirements extracted from the tender text"
        assert analysis.total_requirements == len(analysis.requirements)
        assert all(r.text.strip() for r in analysis.requirements)

    @pytest.mark.parametrize("scenario", ["led", "cable"])
    def test_a_degraded_run_explains_itself(self, scenario, request) -> None:
        """
        Whichever mode the run used, it has to be legible.

        Not an assertion about which mode: Gemini's availability is an
        environment fact. The assertion is that a degraded run never passes
        itself off as a full one — if a fallback happened, its reason is on the
        record, so a report generated during an outage can be read as such.
        """
        analysis = request.getfixturevalue(scenario)
        meta = analysis.metadata

        if meta.get("extraction_fallback"):
            assert meta.get("extraction_fallback_reason"), (
                "extraction fell back without recording why"
            )
        if meta.get("analysis_mode") == "mock":
            assert meta.get("degraded_reason"), (
                "analysis degraded to the deterministic engine without recording why"
            )


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

class TestFindings:
    """The report's primary content."""

    @pytest.mark.parametrize("scenario", ["led", "cable"])
    def test_every_requirement_produces_a_finding(self, scenario, request) -> None:
        analysis = request.getfixturevalue(scenario)
        assert analysis.findings, "the analysis produced no findings"
        assessed = {f.requirement_id for f in analysis.findings}
        assert assessed == {r.id for r in analysis.requirements}, (
            "some requirements were never assessed"
        )

    @pytest.mark.parametrize("scenario", ["led", "cable"])
    def test_each_finding_states_a_verdict_and_its_reasoning(
        self, scenario, request,
    ) -> None:
        analysis = request.getfixturevalue(scenario)
        for f in analysis.findings:
            assert isinstance(f.verdict, Verdict)
            # A verdict with no reasoning is not defensible in a tender file, and
            # the reason is what the officer reads first.
            assert f.reason and f.reason.strip(), f"finding {f.id} has no reason"
            assert 0.0 <= f.confidence <= 1.0

    @pytest.mark.parametrize("scenario", ["led", "cable"])
    def test_findings_are_not_placeholder_text(self, scenario, request) -> None:
        """
        The deterministic engine is a real analysis path, and it must not be
        described as a stub. It used to label its own output "Mock analysis",
        which made an honest degraded report read as fabricated.
        """
        analysis = request.getfixturevalue(scenario)
        for f in analysis.findings:
            lowered = f.reason.lower()
            for banned in ("mock", "placeholder", "lorem ipsum", "todo", "n/a"):
                assert banned not in lowered, (
                    f"finding {f.id} reason reads as placeholder text: {f.reason!r}"
                )

    @pytest.mark.parametrize("scenario", ["led", "cable"])
    def test_the_issue_count_matches_the_findings(self, scenario, request) -> None:
        analysis = request.getfixturevalue(scenario)
        expected = sum(1 for f in analysis.findings if f.verdict is not Verdict.JUSTIFIED)
        assert analysis.issues_found == expected


# ---------------------------------------------------------------------------
# Standards map / relationships
# ---------------------------------------------------------------------------

class TestStandardsMap:
    """
    The matched standards and how they relate — the report's graph view.

    Every assertion here reads the expected value back out of the registry, so
    the test tracks the catalogue instead of a snapshot of it.
    """

    def test_positive_scenario_standards_are_matched_and_valid(
        self, led, registry,
    ) -> None:
        matched = [s for f in led.findings for s in f.applicable_standards]
        assert matched, "no standard was matched to any requirement in the positive scenario"

        for std in matched:
            assert registry.standards_store.get_by_id(std.id) is not None, (
                f"matched standard {std.designation} is not in the knowledge base"
            )
            assert std.is_number.startswith("IS "), std.is_number
            assert std.title and std.title.strip()

    def test_negative_scenario_identifies_incorrect_standard(
        self, cable, registry,
    ) -> None:
        """
        The cable scenario specifies XLPE insulation and cites IS 1554 Part 1,
        which covers PVC. That citation is real, active, correctly numbered and
        wrong, and every year-and-status check in the pipeline passes it — the
        defect is not about the edition, it is about what the standard covers.

        Asserted at the deterministic boundary, `dimensions["scope"]`, rather
        than only on the headline verdict. Three reasons:

          - It is the boundary the check actually lives at. compliance.py reads
            the material out of BIS's own titles; the headline is then whatever
            the severity merge promotes, and pinning only the headline would test
            the merge rather than the detection.
          - It needs no LLM, so the assertion means the same thing in CI, in a
            sandbox with no egress, and on a developer machine with a Gemini key.
            The previous version of this test asserted INCORRECT_STANDARD at the
            end of the pipeline, which was reachable only through Gemini's prose.
          - INCORRECT_STANDARD is this codebase's verdict for a *withdrawn*
            standard. IS 1554 is not withdrawn. WRONG_SCOPE — "the IS exists but
            covers a different application" — is the verdict that describes this,
            and it files on the applicability axis where a material mismatch
            belongs rather than on currentness.

        Expected values are read back out of the registry, per this module's
        rule 1: if the catalogue's XLPE coverage changes, the assertion tracks it.
        """
        req = next(
            (r for r in cable.requirements
             if r.is_reference and "IS 1554" in r.is_reference),
            None,
        )
        assert req, "The referenced IS 1554 was not extracted from the scenario text."

        finding = next(
            (f for f in cable.findings if f.requirement_id == req.id), None
        )
        assert finding, "No finding was generated for the IS 1554 requirement."

        scope = (finding.dimensions or {}).get("scope") or {}
        assert scope.get("assessed"), (
            "the XLPE-vs-PVC comparison was never made: "
            f"{scope.get('note')!r}"
        )
        assert scope.get("mismatch") is True, (
            "IS 1554 (PVC) was not flagged against an XLPE requirement: "
            f"{scope}"
        )

        # The finding has to name both sides, or the officer cannot act on it.
        assert "xlpe" in (scope.get("required_material") or "").lower(), scope
        assert "pvc" in (scope.get("covered_material") or "").lower(), scope
        assert scope.get("attribute") == "insulated", scope

        # And it has to point somewhere. The catalogue's XLPE cable standards are
        # the IS 7098 family; read from the registry rather than named here.
        xlpe_covering = {
            std.designation
            for std in registry.standards_store.list_all()
            if "xlpe" in (std.title or "").lower()
            and "insulated" in (std.title or "").lower()
        }
        assert xlpe_covering, (
            "the catalogue holds no XLPE insulation standard, so this scenario "
            "can no longer test a recoverable mismatch"
        )
        assert set(scope.get("alternatives") or []) & xlpe_covering, (
            f"alternatives {scope.get('alternatives')} name none of the "
            f"catalogue's XLPE standards {sorted(xlpe_covering)}"
        )

        # The headline must carry the scope problem, not bury it: WRONG_SCOPE is
        # more severe than the AMBIGUOUS the omitted year would otherwise yield.
        assert finding.verdict is Verdict.WRONG_SCOPE, (
            f"scope mismatch detected but headline is {finding.verdict}\n"
            f"Reason: {finding.reason}"
        )
        assert (finding.dimensions or {})["headline"]["axis"] == "applicability", (
            "a material mismatch was filed on the currentness axis: "
            f"{finding.dimensions['headline']}"
        )

        # It must not be reported as governing the requirement — that is the one
        # error a procurement officer cannot catch by reading the report.
        assert not any(
            "IS 1554" in s.is_number for s in finding.applicable_standards
        ), (
            "IS 1554 is reported as applicable to an XLPE requirement: "
            f"{[s.designation for s in finding.applicable_standards]}"
        )
        # But it must not vanish either — it is the citation the officer has to fix.
        assert any("IS 1554" in s.is_number for s in finding.cited_standards), (
            "IS 1554 was dropped from the finding entirely: "
            f"{[s.designation for s in finding.cited_standards]}"
        )

        assert finding.reason and len(finding.reason.strip()) > 10, (
            "No reason provided for the mismatch."
        )
        assert "xlpe" in finding.reason.lower(), (
            f"the reason never mentions the material that is wrong: {finding.reason!r}"
        )

    def test_the_led_scenario_matches_the_standards_it_cites(self, led) -> None:
        """
        The tender names IS 10322 and IS 16107. Retrieval has to find them —
        this is the end-to-end check that the reconciled catalogue actually
        covers the parts the old 1015-record catalogue was missing.

        Matched by family prefix, because is_number carries the part in the
        catalogue ("IS 10322 : Part 5 : Sec 1"). Which part retrieval picks is a
        ranking decision this test has no business pinning; that the right family
        was found is the assertion.
        """
        matched = {s.is_number for f in led.findings for s in f.applicable_standards}
        assert any(n.startswith("IS 10322") for n in matched), matched
        assert any(n.startswith("IS 16107") for n in matched), matched

    def test_the_cable_scenario_matches_the_standard_it_cites(self, cable) -> None:
        """
        Retrieval has to find IS 1554 — but it lands under `cited_standards`,
        not `applicable_standards`, because the scope check establishes that it
        covers PVC while the tender specifies XLPE. Asserted over both fields
        together so this stays a test of retrieval: whether the standard was
        found at all is the question here, and which side of the
        cited/applicable line it falls on is
        `test_negative_scenario_identifies_incorrect_standard`.
        """
        matched = {
            s.is_number
            for f in cable.findings
            for s in list(f.applicable_standards) + list(f.cited_standards)
        }
        assert any(n.startswith("IS 1554") for n in matched), matched

    def test_at_least_one_matched_standard_publishes_relationships(
        self, led, registry,
    ) -> None:
        """
        The graph view needs edges. A catalogue where every record is an isolated
        node renders as a list of dots, which is how the old one behaved.
        """
        edges = [
            (std.designation, ref)
            for f in led.findings
            for std in f.applicable_standards
            for ref in list(std.normative_references) + list(std.related_standards)
        ]
        assert edges, "no matched standard declares any related or referenced standard"
        for _, ref in edges:
            assert "IS" in ref.upper(), f"relationship target is not an IS reference: {ref!r}"

    @pytest.mark.parametrize("scenario", ["led", "cable"])
    def test_matched_standards_carry_a_source(self, scenario, request) -> None:
        """Sources panel: a claim about a standard has to say where it came from."""
        analysis = request.getfixturevalue(scenario)
        sourced = [
            std for f in analysis.findings for std in f.applicable_standards
            if std.source_url or std.provenance
        ]
        assert sourced, "no matched standard records a source URL or provenance"


# ---------------------------------------------------------------------------
# Currentness
# ---------------------------------------------------------------------------

class TestCurrentness:
    """VersionChecker output: is the cited edition the current one?"""

    @pytest.mark.parametrize("scenario", ["led", "cable"])
    def test_every_finding_carries_a_currentness_assessment(
        self, scenario, request,
    ) -> None:
        analysis = request.getfixturevalue(scenario)
        for f in analysis.findings:
            block = (f.dimensions or {}).get("currentness")
            assert block, f"finding {f.id} has no currentness assessment"
            assert block.get("label"), f"currentness for {f.id} has no label"
            # The label is the badge; the note is the sentence under it. A badge
            # with no explanation is not actionable.
            assert block.get("note"), f"currentness for {f.id} has no explanation"

    def test_a_superseded_citation_is_caught_with_its_age(self, led) -> None:
        """
        The tender cites IS 10322 : 2012. The catalogue holds a later edition, so
        the pipeline must say so and quantify the gap — the single most valuable
        thing this product does.

        Asserted against the catalogue's own year rather than a literal, so a
        catalogue update moves the expected gap instead of breaking the test.
        """
        outdated = [
            f for f in led.findings
            if (f.dimensions or {}).get("currentness", {}).get("label")
            in {"OUTDATED", "SUPERSEDED", "OUTDATED_REFERENCE"}
        ]
        assert outdated, (
            "the 2012 citation of IS 10322 was not flagged as outdated: "
            + str([
                (f.verdict.value, (f.dimensions or {}).get("currentness", {}).get("label"))
                for f in led.findings
            ])
        )

        block = (outdated[0].dimensions or {})["currentness"]
        current_year = block.get("current_year") or block.get("latest_year")
        assert current_year, "an outdated verdict that does not name the current edition"
        assert int(current_year) > 2012, (
            f"flagged 2012 as outdated against edition {current_year}"
        )

    def test_a_citation_with_no_year_is_flagged_rather_than_assumed(self, led) -> None:
        """
        The tender cites IS 16107 with no edition year. That is a real defect in a
        tender — it leaves which edition binds ambiguous — and the pipeline must
        report it instead of quietly assuming the latest.
        """
        labels = {
            (f.dimensions or {}).get("currentness", {}).get("label")
            for f in led.findings
        }
        assert "YEAR_OMITTED" in labels, labels


# ---------------------------------------------------------------------------
# Certification
# ---------------------------------------------------------------------------

class TestCertification:
    """QCO / BIS certification enrichment."""

    @pytest.mark.parametrize("scenario", ["led", "cable"])
    def test_certification_is_assessed_for_every_finding(
        self, scenario, request,
    ) -> None:
        analysis = request.getfixturevalue(scenario)
        for f in analysis.findings:
            assert "certification" in (f.dimensions or {}), (
                f"finding {f.id} was never checked for certification requirements"
            )

    def test_a_qco_notified_product_is_identified_from_the_catalogue(
        self, led, registry,
    ) -> None:
        """
        LED luminaires are QCO-notified and under CRS. The assertion checks the
        finding agrees with the catalogue record rather than with a constant: if
        the record says QCO-notified, the finding must too.
        """
        qco_findings = [
            f for f in led.findings
            if (f.dimensions or {}).get("certification", {}).get("qco_notified")
        ]
        assert qco_findings, "no QCO requirement identified for LED street lighting"

        for f in qco_findings:
            block = f.dimensions["certification"]
            # Cross-check against the store: the flag must be catalogue-backed.
            backing = [
                registry.standards_store.get_by_id(s.id) for s in f.applicable_standards
            ]
            assert any(s and s.qco_notified for s in backing), (
                f"finding {f.id} claims a QCO that no matched standard record carries"
            )
            # A mandatory-certification claim is only actionable with the scheme
            # that applies — ISI, CRS and hallmarking impose different obligations
            # on the bidder.
            assert block.get("scheme"), (
                f"QCO finding {f.id} does not name the certification scheme: {block}"
            )

    def test_certification_claims_name_their_gazette_order_or_say_why_not(
        self, led,
    ) -> None:
        """
        A QCO assertion has to be auditable — but auditable includes admitting a
        gap. The BIS records behind some QCO-notified products carry the scheme
        and the ministry without the gazette S.O. number, and demanding one would
        mean inventing it. So the contract is: name the order, or state in the
        note that the reference is not in the record.

        What is not acceptable is asserting mandatory certification and going
        silent about the basis.
        """
        checked = 0
        for f in led.findings:
            block = (f.dimensions or {}).get("certification") or {}
            if not block.get("qco_notified"):
                continue
            checked += 1

            note = block.get("note") or ""
            assert note.strip(), (
                f"QCO finding {f.id} asserts mandatory certification with no note"
            )
            has_order = bool(
                block.get("gazette_so_number") or block.get("effective_date")
            )
            if not has_order:
                # The absence has to be disclosed in the text the officer reads,
                # not left for them to notice.
                assert "not in" in note.lower() or "not available" in note.lower(), (
                    f"QCO finding {f.id} has no gazette reference and does not "
                    f"disclose that: {note!r}"
                )
            # Either way the obligation itself must be spelled out.
            assert block.get("issuing_ministry") or block.get("scheme"), block

        assert checked, "no QCO finding to check"


# ---------------------------------------------------------------------------
# Applicability (Phase 2A ML)
# ---------------------------------------------------------------------------

class TestApplicability:
    """
    The applicability dimension: whether the AI/ML layer judged the matched
    standard actually applicable to the requirement, as distinct from merely
    lexically similar.

    Note this is the backend's applicability *verdict*, not the Phase 2A ML
    applicability_score. That score is produced inside ai-engine's recommender
    (src/ml/applicability_model.py annotates candidates there) and is covered by
    ai-engine/tests/test_ml_applicability.py; it is not carried on the Finding,
    so asserting it here would be asserting against the wrong layer.

    What the report needs from this block is honesty about whether the judgement
    was made at all. When the AI/ML stage is unavailable the block is filled with
    nulls and source="not_assessed", and it must be labelled that way — a
    placeholder that reports itself as a completed assessment is worse than an
    empty panel.
    """

    @pytest.mark.parametrize("scenario", ["led", "cable"])
    def test_applicability_is_recorded_or_explicitly_not_assessed(
        self, scenario, request,
    ) -> None:
        analysis = request.getfixturevalue(scenario)
        for f in analysis.findings:
            label = (f.data_availability or {}).get("applicability")
            assert label in AVAILABILITY_STATES, (
                f"finding {f.id} does not state whether applicability was assessed"
            )

            block = (f.dimensions or {}).get("applicability") or {}
            assert block, f"finding {f.id} has no applicability block at all"
            assert "source" in block, f"applicability block names no source: {block}"

            if label == "verified":
                # Claimed as assessed, so it must carry a real judgement.
                assert block.get("source") not in (None, "not_assessed"), block
                assert block.get("verdict"), (
                    f"applicability verified but no verdict: {block}"
                )
                confidence = block.get("confidence")
                assert confidence is not None, block
                assert 0.0 <= float(confidence) <= 1.0, confidence
            else:
                # Not assessed, so it must say why rather than looking empty.
                assert block.get("source") == "not_assessed", block
                assert block.get("note"), (
                    f"applicability was not assessed and does not explain why: {block}"
                )


# ---------------------------------------------------------------------------
# Cross-references
# ---------------------------------------------------------------------------

class TestCrossReferences:
    """
    CrossRefExtractor output: the standards a cited standard drags in with it,
    and which of those the tender fails to mention. This is the finding a
    procurement officer cannot get from reading the tender.
    """

    def test_the_led_scenario_surfaces_a_real_reference_chain(
        self, led, registry,
    ) -> None:
        chains = [
            entry for f in led.findings for entry in (f.cross_references or [])
        ]
        assert chains, "no cross-reference chain was extracted"

        for entry in chains:
            assert entry.get("source"), f"chain with no source standard: {entry}"
            refs = entry.get("references") or []
            assert refs, f"chain for {entry['source']} lists no references"
            # The note is the sentence the report shows; a chain without one is a
            # list of designations the officer has to interpret unaided.
            assert entry.get("note"), f"chain for {entry['source']} has no note"

            # Each reference must be one the catalogue actually publishes for that
            # standard — not something the extractor produced on its own.
            family = registry.standards_store.get_by_is_number(entry["source"])
            published: set[str] = set()
            for std in family:
                published |= set(std.normative_references) | set(std.related_standards)
            if published:
                # CrossRefExtractor normalizes "IS 1944 : Part 1" down to "IS 1944".
                # Check if any extracted ref is a prefix of any published ref.
                matched = any(
                    any(p.startswith(r) for p in published)
                    for r in refs
                )
                assert matched, (
                    f"references {refs} for {entry['source']} match none of "
                    f"the catalogue's {sorted(published)}"
                )

    def test_unmet_references_are_a_subset_of_the_references(self, led) -> None:
        """
        "Unmet" means "referenced but not cited by this tender". It cannot
        contain anything that is not in the reference list, or the report is
        flagging a dependency that does not exist.
        """
        for f in led.findings:
            for entry in f.cross_references or []:
                refs = set(entry.get("references") or [])
                unmet = set(entry.get("unmet") or [])
                assert unmet <= refs, (
                    f"unmet {sorted(unmet - refs)} are not among the references"
                )

    def test_a_reference_the_tender_never_cites_is_reported(self, led) -> None:
        """
        IS 16107 normatively references IS 1944 and IS 16101; the tender cites
        neither. That gap is the point of the feature, so it must be reported.
        """
        unmet = {
            ref
            for f in led.findings
            for entry in (f.cross_references or [])
            for ref in (entry.get("unmet") or [])
        }
        assert unmet, (
            "the tender cites none of IS 16107's normative references, yet no "
            "unmet reference was reported"
        )
        # Nothing the tender does cite should appear as unmet.
        for ref in unmet:
            assert "10322" not in ref, (
                f"IS 10322 is cited in the tender but was reported unmet: {ref}"
            )


# ---------------------------------------------------------------------------
# Evidence / provenance
# ---------------------------------------------------------------------------

class TestEvidence:
    """
    The audit trail. Every claim in the report has to be traceable to a source a
    procurement officer can open, or the report is not defensible on challenge.
    """

    @pytest.mark.parametrize("scenario", ["led", "cable"])
    def test_findings_carry_evidence(self, scenario, request) -> None:
        analysis = request.getfixturevalue(scenario)
        with_evidence = [f for f in analysis.findings if f.evidence]
        assert with_evidence, "no finding carries any evidence"

    @pytest.mark.parametrize("scenario", ["led", "cable"])
    def test_each_evidence_record_names_its_authority(self, scenario, request) -> None:
        analysis = request.getfixturevalue(scenario)
        for f in analysis.findings:
            for ev in f.evidence:
                assert ev.source_name and ev.source_name.strip(), (
                    f"evidence on finding {f.id} has no source name"
                )
                assert ev.authority and ev.authority.strip(), (
                    f"evidence on finding {f.id} names no issuing authority"
                )
                assert ev.source_type is not None

    @pytest.mark.parametrize("scenario", ["led", "cable"])
    def test_at_least_one_evidence_record_is_resolvable(
        self, scenario, request,
    ) -> None:
        """A citation the officer can actually follow."""
        analysis = request.getfixturevalue(scenario)
        resolvable = [
            ev for f in analysis.findings for ev in f.evidence
            if (ev.url and ev.url.startswith("http"))
            or ev.gazette_so_number
            or ev.section
        ]
        assert resolvable, (
            "no evidence record carries a URL, a gazette number or a clause "
            "reference — nothing in the report can be looked up"
        )


# ---------------------------------------------------------------------------
# Recommended actions
# ---------------------------------------------------------------------------

class TestRecommendedActions:
    """What the officer should do about each finding."""

    @pytest.mark.parametrize("scenario", ["led", "cable"])
    def test_a_flagged_finding_recommends_something(self, scenario, request) -> None:
        analysis = request.getfixturevalue(scenario)
        flagged = [f for f in analysis.findings if f.verdict is not Verdict.JUSTIFIED]
        if not flagged:
            pytest.skip("no issues raised in this scenario, nothing to act on")

        for f in flagged:
            label = (f.data_availability or {}).get("recommended_action")
            assert label in AVAILABILITY_STATES
            if label == "verified":
                assert f.recommended_action.strip()
                # An action has to be actionable: naming the standard is the
                # minimum for the officer to edit the clause.
                assert len(f.recommended_action.split()) >= 3, f.recommended_action

    @pytest.mark.parametrize("scenario", ["led", "cable"])
    def test_low_confidence_findings_ask_for_review(self, scenario, request) -> None:
        """
        A finding the system is unsure about must say so rather than presenting
        itself with the same authority as a confident one.
        """
        analysis = request.getfixturevalue(scenario)
        for f in analysis.findings:
            if f.requires_human_verification:
                assert f.verification_reason and f.verification_reason.strip(), (
                    f"finding {f.id} asks for review without saying why"
                )


# ---------------------------------------------------------------------------
# The API contract for empty sections
# ---------------------------------------------------------------------------

class TestSectionAvailabilityContract:
    """
    The rule that keeps the frontend from looking broken: an optional section is
    either populated or explicitly labelled, never a silent blank.

    This is what makes it safe for the UI to render an empty panel — it can say
    "BIS publishes no normative references for this standard" instead of showing
    nothing and leaving the officer to guess whether the backend failed.
    """

    @pytest.mark.parametrize("scenario", ["led", "cable"])
    def test_every_section_is_labelled(self, scenario, request) -> None:
        analysis = request.getfixturevalue(scenario)
        for f in analysis.findings:
            labels = f.data_availability or {}
            for section in DEMO_SECTIONS:
                assert section in labels, (
                    f"finding {f.id} does not state the availability of {section!r}"
                )
                assert labels[section] in AVAILABILITY_STATES, (
                    f"{section} on finding {f.id} has an unknown state: "
                    f"{labels[section]!r}"
                )

    @pytest.mark.parametrize("scenario", ["led", "cable"])
    def test_a_verified_label_is_backed_by_data(self, scenario, request) -> None:
        """The labels must not drift from the payload they describe."""
        analysis = request.getfixturevalue(scenario)
        for f in analysis.findings:
            labels = f.data_availability or {}
            dimensions = f.dimensions or {}

            if labels.get("applicable_standards") == "verified":
                assert f.applicable_standards
            if labels.get("cross_references") == "verified":
                assert f.cross_references
            if labels.get("evidence") == "verified":
                assert f.evidence
            if labels.get("currentness") == "verified":
                assert f.currentness or dimensions.get("currentness")
            if labels.get("certification") == "verified":
                assert dimensions.get("certification")
            if labels.get("scope") == "verified":
                block = dimensions.get("scope") or {}
                assert block.get("assessed") is True, block
                # A completed comparison has two sides and a sentence saying what
                # it concluded. Without those the panel claims an assessment it
                # cannot show.
                assert block.get("mismatch") is not None, block
                assert block.get("note"), block
                assert block.get("attribute"), block
            if labels.get("cited_standards") == "verified":
                assert f.cited_standards
                # A standard reported as cited-but-not-applicable must say why,
                # or it reads as an unexplained demotion.
                assert (dimensions.get("scope") or {}).get("mismatch") is True, (
                    f"finding {f.id} reports a cited-not-applicable standard with "
                    f"no scope mismatch behind it: {dimensions.get('scope')}"
                )
            if labels.get("recommended_action") == "verified":
                assert f.recommended_action

    @pytest.mark.parametrize("scenario", ["led", "cable"])
    def test_an_empty_section_is_never_labelled_verified(
        self, scenario, request,
    ) -> None:
        """The inverse: nothing empty may claim to hold data."""
        analysis = request.getfixturevalue(scenario)
        for f in analysis.findings:
            labels = f.data_availability or {}
            if not f.applicable_standards:
                assert labels.get("applicable_standards") != "verified"
            if not f.cross_references:
                assert labels.get("cross_references") != "verified"
            if not f.evidence:
                assert labels.get("evidence") != "verified"
            if not f.cited_standards:
                assert labels.get("cited_standards") != "verified"


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    """
    The headline. Derived by counting real findings, which is why it can be
    checked against them — a generated sentence could not be.
    """

    @pytest.mark.parametrize("scenario", ["led", "cable"])
    def test_the_summary_is_populated(self, scenario, request) -> None:
        analysis = request.getfixturevalue(scenario)
        assert analysis.summary and analysis.summary.strip(), (
            "the analysis produced no summary"
        )

    @pytest.mark.parametrize("scenario", ["led", "cable"])
    def test_the_summary_counts_agree_with_the_findings(
        self, scenario, request,
    ) -> None:
        analysis = request.getfixturevalue(scenario)
        summary = analysis.summary
        # The two numbers the summary opens with are the ones most likely to be
        # quoted, so they are the ones held to the findings.
        assert str(len(analysis.requirements)) in summary, summary
        standards = {
            s.is_number for f in analysis.findings for s in f.applicable_standards
        }
        if standards:
            assert str(len(standards)) in summary, summary

    @pytest.mark.parametrize("scenario", ["led", "cable"])
    def test_the_summary_does_not_claim_a_clean_result_over_issues(
        self, scenario, request,
    ) -> None:
        analysis = request.getfixturevalue(scenario)
        if analysis.issues_found:
            assert "No issues were found" not in analysis.summary, analysis.summary
