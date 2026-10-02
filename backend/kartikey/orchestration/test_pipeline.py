"""
kartikey/orchestration/test_pipeline.py

Unit tests for the integrated analysis pipeline state machine.

Tests cover:
  - Text input analysis pipeline end-to-end execution
  - Document input analysis pipeline execution
  - Requirement extraction integration (_step_extract)
  - Standards retrieval integration (_step_retrieve)
  - Pipeline error handling and state transitions
"""

from __future__ import annotations

import asyncio
import pytest

from shared.models import Analysis, AnalysisStatus, InputType, Standard, StandardStatus
from kartikey.orchestration.knowledge_registry import initialize_knowledge_registry
from kartikey.orchestration.pipeline import run_analysis_pipeline


@pytest.fixture
def registry():
    """Initialize and clear the registry used by the current pipeline."""
    registry = initialize_knowledge_registry()
    registry.standards_store.clear()
    registry.evidence_store.clear()
    return registry


def test_pipeline_text_input_end_to_end(registry) -> None:
    """Verify that a text input analysis runs through all steps to COMPLETED."""
    async def _run() -> None:
        std = Standard(
            is_number="IS 10322",
            title="Specification for Luminaires - Street Lighting",
            status=StandardStatus.ACTIVE,
        )
        registry.standards_store.add(std)

        analysis_store: dict[str, Analysis] = {}
        analysis = Analysis(
            input_type=InputType.TEXT,
            raw_text=(
                "1. The LED street lighting luminaire shall conform to IS 10322 (Part 5):2012.\n"
                "2. The driver shall comply with IS 15885 Part 1."
            ),
            status=AnalysisStatus.QUEUED,
        )
        analysis_store[analysis.id] = analysis

        await run_analysis_pipeline(analysis.id, analysis_store)

        completed_analysis = analysis_store[analysis.id]
        assert completed_analysis.status == AnalysisStatus.COMPLETED
        assert len(completed_analysis.requirements) == 2
        assert completed_analysis.total_requirements == 2
        assert completed_analysis.error_message is None

    asyncio.run(_run())


def test_pipeline_missing_analysis_id(registry) -> None:
    """Verify that running a pipeline for an unknown analysis ID aborts gracefully."""
    async def _run() -> None:
        analysis_store: dict[str, Analysis] = {}
        await run_analysis_pipeline("nonexistent-id", analysis_store)
        assert "nonexistent-id" not in analysis_store

    asyncio.run(_run())


def test_pipeline_text_input_missing_text(registry) -> None:
    """Verify that text input without raw_text transitions to FAILED."""
    async def _run() -> None:
        analysis_store: dict[str, Analysis] = {}
        analysis = Analysis(
            input_type=InputType.TEXT,
            raw_text=None,
            status=AnalysisStatus.QUEUED,
        )
        analysis_store[analysis.id] = analysis

        await run_analysis_pipeline(analysis.id, analysis_store)

        failed_analysis = analysis_store[analysis.id]
        assert failed_analysis.status == AnalysisStatus.FAILED
        assert "MISSING_INPUT" in (failed_analysis.error_message or "")

    asyncio.run(_run())


# ===========================================================================
# Gemini final pass integration (Phase 2, Step 6)
# ===========================================================================

def test_step_analyze_gemini_findings_reach_assembled_findings(registry, monkeypatch) -> None:
    """
    Prove the Gemini final pass is wired end-to-end: its verdict and citation
    survive _step_analyze, the anti-hallucination resolution in _step_enrich,
    and land on a real Finding with a real Standard attached.

    The LLM is stubbed — this tests the wiring, not the model.
    """
    from shared.models import Requirement, Verdict
    from kartikey.orchestration.pipeline import _step_analyze, _step_enrich

    class _StubLlm:
        _working_model = "stub-model"

        def generate_json(self, prompt, system_prompt="", temperature=0.1, response_schema=None):
            # Index 0 is the only candidate standard shown in the prompt.
            return {
                "findings": [
                    {
                        "requirement_index": 0,
                        "verdict": Verdict.JUSTIFIED.value,
                        "reason": "IS 10322 covers street lighting luminaires.",
                        "recommended_action": "No action required.",
                        "applicable_standard_indices": [0],
                        "confidence": 0.87,
                    }
                ]
            }

    monkeypatch.setattr(
        "kartikey.analysis.llm_client.get_llm_client", lambda: _StubLlm()
    )
    # No HTTP service, Gemini enabled — exercise the Gemini branch specifically.
    monkeypatch.setattr("shared.config.settings.aiml_service_url", "")
    monkeypatch.setattr("shared.config.settings.google_api_key", "stub-key")
    monkeypatch.setattr("shared.config.settings.gemini_final_pass_enabled", True)

    async def _run() -> None:
        std = Standard(
            is_number="IS 10322",
            title="Specification for Luminaires - Street Lighting",
            status=StandardStatus.ACTIVE,
        )
        registry.standards_store.add(std)

        analysis = Analysis(
            input_type=InputType.TEXT,
            raw_text="Luminaires shall conform to IS 10322.",
            status=AnalysisStatus.QUEUED,
        )
        analysis.requirements = [
            Requirement(
                analysis_id=analysis.id,
                text="Luminaires shall conform to IS 10322.",
                is_reference="IS 10322",
            )
        ]

        response = await _step_analyze(analysis, analysis.raw_text, [std])

        assert response is not None
        assert response.extraction_metadata["execution_mode"] in ("gemini", "mock")
        assert analysis.metadata["analysis_engine"] in ("gemini", "mock")
        assert analysis.metadata["analysis_mode"] == "remote"
        assert "degraded_reason" not in analysis.metadata
        # The index Gemini returned was resolved back to the real Standard ID.
        assert response.findings[0].applicable_standard_ids == [std.id]

        await _step_enrich(analysis, [std], response)

        assert len(analysis.findings) == 1
        finding = analysis.findings[0]
        assert "IS 10322" in finding.reason
        assert [s.is_number for s in finding.applicable_standards] == ["IS 10322"]

    asyncio.run(_run())


def test_step_analyze_records_deterministic_fallback(registry, monkeypatch) -> None:
    """When Gemini is unusable, the analysis still completes and says so."""
    from shared.models import Requirement
    from shared.utils import AnalysisError
    from kartikey.orchestration.pipeline import _step_analyze

    def _boom():
        raise AnalysisError("no key", code="LLM_NOT_CONFIGURED")

    monkeypatch.setattr("kartikey.analysis.llm_client.get_llm_client", _boom)
    monkeypatch.setattr("shared.config.settings.aiml_service_url", "")
    monkeypatch.setattr("shared.config.settings.google_api_key", "stub-key")
    monkeypatch.setattr("shared.config.settings.gemini_final_pass_enabled", True)

    async def _run() -> None:
        std = Standard(
            is_number="IS 10322", title="Luminaires", status=StandardStatus.ACTIVE
        )
        analysis = Analysis(
            input_type=InputType.TEXT,
            raw_text="Luminaires shall conform to IS 10322.",
            status=AnalysisStatus.QUEUED,
        )
        analysis.requirements = [
            Requirement(analysis_id=analysis.id, text="Luminaires shall conform to IS 10322.")
        ]

        response = await _step_analyze(analysis, analysis.raw_text, [std])

        assert response is None
        assert analysis.metadata["analysis_engine"] == "none"
        assert analysis.metadata["analysis_mode"] == "fallback"
        assert "Gemini final pass unavailable" in analysis.metadata["degraded_reason"]

    asyncio.run(_run())


# ===========================================================================
# Enrichment integration (Phase 3, Steps 7-9)
# ===========================================================================

def test_step_enrich_records_currentness_and_dependency_summary(registry) -> None:
    """
    Prove the enrichment subsystems are actually invoked by the pipeline, not
    merely importable. Before Phase 3, VersionChecker and CrossRefExtractor were
    fully built and fully tested modules that nothing in _step_enrich called.

    The tender below cites a 2012 edition of a 2022 standard (currentness), and
    that standard normatively references IS 10322 and IS 1944, neither of which
    the tender mentions (dependencies). Both must reach the audit trail.
    """
    from shared.models import Requirement, Verdict
    from kartikey.orchestration.pipeline import _step_enrich

    async def _run() -> None:
        std = Standard(
            is_number="IS 16107",
            year=2022,
            title="Luminaire performance requirements",
            status=StandardStatus.ACTIVE,
            normative_references=["IS 10322 : Part 1", "IS 1944"],
        )
        registry.standards_store.add(std)

        analysis = Analysis(
            input_type=InputType.TEXT,
            raw_text="Luminaires shall conform to IS 16107:2012.",
            status=AnalysisStatus.QUEUED,
        )
        analysis.requirements = [
            Requirement(
                analysis_id=analysis.id,
                text="Luminaires shall conform to IS 16107:2012.",
                is_reference="IS 16107",
                cited_year=2012,
            )
        ]

        # aiml_response=None exercises the deterministic-only path, so this test
        # needs no LLM at all.
        await _step_enrich(analysis, [std], None)

        assert len(analysis.findings) == 1
        finding = analysis.findings[0]

        # Step 7: the version decision came from VersionChecker.
        assert finding.verdict == Verdict.OUTDATED_REFERENCE
        assert finding.dimensions["currentness"]["gap_years"] == 10

        # Step 9: the uncited dependencies were extracted and reported.
        assert set(finding.cross_references[0]["unmet"]) == {"IS 10322", "IS 1944"}

        # Both reach the audit trail the report header reads.
        enrichment = analysis.metadata["enrichment"]
        assert enrichment["findings"] == 1
        assert enrichment["currentness_flagged"] == 1
        assert enrichment["unmet_dependencies"] == ["IS 10322", "IS 1944"]

    asyncio.run(_run())
