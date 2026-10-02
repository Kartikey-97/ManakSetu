"""
kartikey/orchestration/test_query_gate_integration.py

Tests that the quality gate is actually wired in, at both entry points, and that
it runs BEFORE anything expensive.

The unit tests in kartikey/analysis/test_query_quality.py prove the gate decides
correctly. These prove it is reached: a gate that is correct but bypassed costs
exactly as much as no gate at all. So the assertions here are mostly about what
does *not* happen — no extractor call, no retrieval, no Gemini.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from kartikey.analysis.query_quality import REJECTION_MESSAGE
from kartikey.orchestration.pipeline import run_analysis_pipeline
from shared.models import Analysis, AnalysisStatus, InputType

GIBBERISH = "fhbjfbdjvfdjhbdf"
VALID = "IS 10322 for LED street lighting"


# ===========================================================================
# The route (TEXT input)
# ===========================================================================

class TestRouteGate:
    """
    Rejecting at the route means no analysis row is created and the user gets a
    synchronous answer, rather than a job that fails a few seconds later and has
    to be polled to discover it.
    """

    @pytest.fixture
    def create(self):
        from kartikey.api.routes.analyses import create_analysis
        return create_analysis

    @pytest.mark.asyncio
    async def test_gibberish_text_is_rejected_with_the_exact_message(self, create):
        from fastapi import BackgroundTasks

        from shared.contracts import CreateAnalysisRequest

        with pytest.raises(HTTPException) as caught:
            await create(
                CreateAnalysisRequest(input_type=InputType.TEXT, text=GIBBERISH),
                BackgroundTasks(),
            )

        assert caught.value.status_code == 422
        assert caught.value.detail["error"] == "NOT_A_PROCUREMENT_REQUIREMENT"
        assert caught.value.detail["message"] == (
            "Unable to identify a meaningful procurement requirement. "
            "Please enter a product, specification, or tender requirement."
        )

    @pytest.mark.asyncio
    async def test_the_rejection_carries_its_reasoning(self, create):
        from fastapi import BackgroundTasks

        from shared.contracts import CreateAnalysisRequest

        with pytest.raises(HTTPException) as caught:
            await create(
                CreateAnalysisRequest(input_type=InputType.TEXT, text=GIBBERISH),
                BackgroundTasks(),
            )

        assert caught.value.detail["signals"]["reason"] == "no_meaningful_content"

    @pytest.mark.asyncio
    async def test_no_analysis_is_created_for_rejected_input(self, create, monkeypatch):
        """
        Nothing is persisted. Asserted by watching the repository's save rather
        than by counting rows, so the test proves the route never even tried —
        a save that failed for an unrelated reason would otherwise look like a
        pass.
        """
        from fastapi import BackgroundTasks

        from kartikey.api.routes import analyses as routes
        from shared.contracts import CreateAnalysisRequest

        saved: list[object] = []

        async def _record(analysis):
            saved.append(analysis)

        monkeypatch.setattr(routes.repository, "save", _record)

        with pytest.raises(HTTPException):
            await create(
                CreateAnalysisRequest(input_type=InputType.TEXT, text=GIBBERISH),
                BackgroundTasks(),
            )

        assert saved == []

    @pytest.mark.asyncio
    async def test_no_background_task_is_queued_for_rejected_input(self, create):
        """The pipeline must never be dispatched — that is the whole saving."""
        from fastapi import BackgroundTasks

        from shared.contracts import CreateAnalysisRequest

        tasks = BackgroundTasks()
        with pytest.raises(HTTPException):
            await create(
                CreateAnalysisRequest(input_type=InputType.TEXT, text=GIBBERISH),
                tasks,
            )

        assert tasks.tasks == []

    @pytest.mark.asyncio
    async def test_empty_text_still_reports_the_missing_field(self, create):
        """
        The gate must not swallow the pre-existing 400. An empty body is a
        client bug; gibberish is a user typo. They are different answers.
        """
        from fastapi import BackgroundTasks

        from shared.contracts import CreateAnalysisRequest

        with pytest.raises(HTTPException) as caught:
            await create(
                CreateAnalysisRequest(input_type=InputType.TEXT, text="   "),
                BackgroundTasks(),
            )

        assert caught.value.status_code == 400
        assert caught.value.detail["error"] == "MISSING_TEXT"

    @pytest.mark.asyncio
    async def test_a_valid_query_is_accepted_and_dispatched(self, create, monkeypatch):
        from fastapi import BackgroundTasks

        import kartikey.api.routes.analyses as routes
        from shared.contracts import CreateAnalysisRequest

        async def _noop(analysis):
            return None

        monkeypatch.setattr(routes.repository, "save", _noop)

        tasks = BackgroundTasks()
        response = await create(
            CreateAnalysisRequest(input_type=InputType.TEXT, text=VALID), tasks
        )

        assert response["status"] == AnalysisStatus.QUEUED
        assert len(tasks.tasks) == 1


# ===========================================================================
# The pipeline (DOCUMENT input, and defence in depth for TEXT)
# ===========================================================================

class TestPipelineGate:
    """
    The route only sees typed text. A scanned tender whose OCR produced noise
    arrives as a document_id, so the gate has to exist here too — and this is
    also the layer that proves the gate precedes the expensive work.
    """

    @staticmethod
    def _analysis(text: str) -> Analysis:
        return Analysis(
            input_type=InputType.TEXT,
            raw_text=text,
            status=AnalysisStatus.QUEUED,
        )

    @pytest.mark.asyncio
    async def test_gibberish_fails_the_analysis_with_the_exact_message(self):
        analysis = self._analysis(GIBBERISH)

        await run_analysis_pipeline(analysis.id, {analysis.id: analysis})

        assert analysis.status is AnalysisStatus.FAILED
        assert analysis.error_message == REJECTION_MESSAGE

    @pytest.mark.asyncio
    async def test_the_message_carries_no_error_code_prefix(self):
        """
        The generic AnalysisError handler writes "[CODE] message". This text is
        addressed to a procurement officer, so it must arrive as written.
        """
        analysis = self._analysis(GIBBERISH)

        await run_analysis_pipeline(analysis.id, {analysis.id: analysis})

        assert not analysis.error_message.startswith("[")

    @pytest.mark.asyncio
    async def test_nothing_expensive_runs_for_rejected_input(self, monkeypatch):
        """
        The point of the gate. If any of these three is reached, the gate has
        been placed too late to save anything.
        """
        import kartikey.analysis.requirement_extractor as extractor
        import kartikey.orchestration.pipeline as pipeline

        def _forbidden(*args, **kwargs):
            raise AssertionError("expensive stage ran on rejected input")

        monkeypatch.setattr(extractor, "extract_requirements", _forbidden)
        monkeypatch.setattr(pipeline, "_step_retrieve", _forbidden)
        monkeypatch.setattr(pipeline, "_step_analyze", _forbidden)

        analysis = self._analysis(GIBBERISH)
        await run_analysis_pipeline(analysis.id, {analysis.id: analysis})

        assert analysis.status is AnalysisStatus.FAILED

    @pytest.mark.asyncio
    async def test_a_document_that_ocred_into_noise_is_rejected(self, monkeypatch):
        import kartikey.document_processing.storage as storage

        monkeypatch.setattr(
            storage, "get_extracted_text",
            lambda doc_id: "dhdjfhfjfjfjfjsjsjshshsjsjs kjhgfdsa zxcvbnmqwe",
        )

        analysis = Analysis(
            input_type=InputType.DOCUMENT,
            document_id="doc-noise",
            status=AnalysisStatus.QUEUED,
        )
        await run_analysis_pipeline(analysis.id, {analysis.id: analysis})

        assert analysis.status is AnalysisStatus.FAILED
        assert analysis.error_message == REJECTION_MESSAGE

    @pytest.mark.asyncio
    async def test_the_verdict_is_recorded_for_inspection(self):
        """
        A rejected job has to be explainable after the fact, not just at the
        moment it was refused.
        """
        analysis = self._analysis(GIBBERISH)

        await run_analysis_pipeline(analysis.id, {analysis.id: analysis})

        assert analysis.metadata["input_quality"]["reason"] == "no_meaningful_content"

    @pytest.mark.asyncio
    async def test_a_rejected_analysis_is_persisted(self):
        """The frontend polls for the outcome; it has to be saved to be read."""
        saved: list[Analysis] = []

        async def _persist(analysis):
            saved.append(analysis)

        analysis = self._analysis(GIBBERISH)
        await run_analysis_pipeline(analysis.id, {analysis.id: analysis}, persist=_persist)

        assert saved and saved[-1].status is AnalysisStatus.FAILED

    @pytest.mark.asyncio
    async def test_a_valid_query_passes_the_gate(self, monkeypatch):
        """
        Reached by proving the stage *after* the gate is entered. Stopping there
        keeps this test off the network — the extractor makes a live LLM call,
        which is not what is under test here.
        """
        import kartikey.orchestration.pipeline as pipeline

        reached: list[str] = []

        async def _spy(analysis, *args, **kwargs):
            reached.append("retrieve")
            raise RuntimeError("stop here")

        monkeypatch.setattr(pipeline, "_step_retrieve", _spy)

        async def _extract_only(analysis):
            # Re-run the real gate, then hand back text without calling the LLM.
            from kartikey.analysis.query_quality import assess_query
            quality = assess_query(analysis.raw_text)
            assert quality.is_meaningful
            return analysis.raw_text

        monkeypatch.setattr(pipeline, "_step_extract", _extract_only)

        analysis = self._analysis(VALID)
        await run_analysis_pipeline(analysis.id, {analysis.id: analysis})

        assert reached == ["retrieve"]
