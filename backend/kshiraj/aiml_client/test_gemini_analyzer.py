"""
kshiraj/aiml_client/test_gemini_analyzer.py

Tests for the Gemini final pass (Phase 2, Step 6).

The LLM is stubbed throughout — no network calls. What is being tested is the
contract around the model, which is where the risk lives:

  - the prompt never leaks a Standard ID, so a hallucinated citation cannot
    round-trip into an AimlFinding
  - every index Gemini returns is bounds-checked against what it was shown
  - every requirement gets exactly one finding, even when Gemini skips it
  - a Gemini failure degrades to deterministic analysis instead of failing
"""

from __future__ import annotations

import pytest

from shared.contracts import AimlRequest, AimlResponse
from shared.models import Requirement, Standard, StandardStatus, Verdict
from shared.utils import AnalysisError

from kshiraj.aiml_client import gemini_analyzer as ga
from kshiraj.aiml_client.client import AimlClient
from kshiraj.aiml_client.schemas import AimlResponseError


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def standards() -> list[Standard]:
    return [
        Standard(
            id="std-aaa",
            is_number="IS 10322 : Part 5 : Sec 3",
            title="Luminaires - Road and Street Lighting",
            scope="Covers luminaires for road and street lighting.",
            status=StandardStatus.ACTIVE,
            year=2012,
        ),
        Standard(
            id="std-bbb",
            is_number="IS 694",
            title="PVC Insulated Cables",
            scope="Covers PVC insulated cables up to 1100 V.",
            status=StandardStatus.ACTIVE,
            year=2010,
        ),
    ]


@pytest.fixture
def requirements() -> list[Requirement]:
    return [
        Requirement(
            id="req-11111111-a",
            analysis_id="an-1",
            text="Street light luminaires shall conform to IS 10322.",
            is_reference="IS 10322 : Part 5 : Sec 3",
        ),
        Requirement(
            id="req-22222222-b",
            analysis_id="an-1",
            text="Cables shall be copper conductor, PVC insulated.",
        ),
    ]


@pytest.fixture
def request_obj(requirements, standards) -> AimlRequest:
    return AimlRequest(
        analysis_id="an-1",
        extracted_text="Supply and installation of LED street lighting.",
        requirements=requirements,
        retrieved_standards=standards,
    )


class _StubClient:
    """Stands in for GeminiClient — records prompts, replays canned payloads."""

    def __init__(self, payloads: list[object] | object) -> None:
        self._payloads = payloads if isinstance(payloads, list) else [payloads]
        self._call = 0
        self.prompts: list[str] = []
        self.system_prompts: list[str] = []
        self.schemas: list[object] = []
        self._working_model = "stub-model"

    def generate_json(self, prompt, system_prompt="", temperature=0.1, response_schema=None):
        self.prompts.append(prompt)
        self.system_prompts.append(system_prompt)
        self.schemas.append(response_schema)
        payload = self._payloads[min(self._call, len(self._payloads) - 1)]
        self._call += 1
        if isinstance(payload, Exception):
            raise payload
        return payload


@pytest.fixture
def stub_llm(monkeypatch):
    """Install a stub LLM client and hand the stub back to the test."""

    def _install(payloads) -> _StubClient:
        stub = _StubClient(payloads)
        monkeypatch.setattr(
            "kartikey.analysis.llm_client.get_llm_client", lambda: stub
        )
        return stub

    return _install


def _payload(**overrides) -> dict:
    """A well-formed single finding, with fields overridable per test."""
    finding = {
        "requirement_index": 0,
        "verdict": "justified",
        "reason": "IS 10322 covers road and street lighting luminaires.",
        "recommended_action": "No action required.",
        "applicable_standard_indices": [0],
        "confidence": 0.88,
    }
    finding.update(overrides)
    return {"findings": [finding]}


# ===========================================================================
# 1. Prompt construction — the anti-hallucination invariant
# ===========================================================================

class TestPromptConstruction:

    def test_prompt_never_contains_standard_ids(self, requirements, standards, request_obj):
        """Gemini cannot cite an ID it was never shown."""
        prompt = ga._build_prompt(requirements, standards, request_obj.extracted_text)
        for std in standards:
            assert std.id not in prompt

    def test_prompt_never_contains_requirement_ids(self, requirements, standards):
        prompt = ga._build_prompt(requirements, standards, "")
        for req in requirements:
            assert req.id not in prompt

    def test_prompt_indexes_standards_and_requirements(self, requirements, standards):
        prompt = ga._build_prompt(requirements, standards, "")
        assert "[0] IS 10322 : Part 5 : Sec 3" in prompt
        assert "[1] IS 694" in prompt
        assert "[0] Street light luminaires shall conform to IS 10322." in prompt
        assert "[1] Cables shall be copper conductor, PVC insulated." in prompt

    def test_prompt_states_expected_finding_count(self, requirements, standards):
        prompt = ga._build_prompt(requirements, standards, "")
        assert "Return exactly 2 findings" in prompt

    def test_prompt_handles_no_candidate_standards(self, requirements):
        prompt = ga._build_prompt(requirements, [], "")
        assert "retrieval returned no candidates" in prompt

    def test_long_fields_are_truncated(self, standards):
        req = Requirement(id="r", analysis_id="a", text="x" * 5000)
        prompt = ga._build_prompt([req], standards, "y" * 9000)
        assert "[…]" in prompt
        assert "x" * 5000 not in prompt
        assert "y" * 9000 not in prompt

    def test_system_prompt_frames_tender_text_as_untrusted(self):
        assert "untrusted data" in ga._SYSTEM_PROMPT
        # Every Verdict value must be documented, or the model cannot use it.
        for verdict in Verdict:
            assert verdict.value in ga._SYSTEM_PROMPT


# ===========================================================================
# 2. Response schema
# ===========================================================================

class TestResponseSchema:

    def test_schema_is_passed_to_gemini(self, stub_llm, request_obj):
        stub = stub_llm(_payload())
        ga.run_gemini_analysis(request_obj)
        assert stub.schemas == [ga._GeminiAnalysis]

    def test_schema_constrains_verdict_to_enum(self):
        """The verdict enum must reach Gemini, or nothing enforces it server-side."""
        json_schema = ga._GeminiAnalysis.model_json_schema()
        finding_schema = json_schema["$defs"]["_GeminiFinding"]
        assert set(finding_schema["properties"]["verdict"]["enum"]) == {
            v.value for v in Verdict
        }

    def test_schema_converts_for_the_genai_sdk(self):
        """
        Guards against an SDK upgrade silently dropping the enum: google-genai
        rebuilds the Pydantic model into its own Schema type.
        """
        pytest.importorskip("google.genai")
        from google.genai import _transformers

        schema = _transformers.t_schema(None, ga._GeminiAnalysis)
        verdict = schema.properties["findings"].items.properties["verdict"]
        assert set(verdict.enum) == {v.value for v in Verdict}


# ===========================================================================
# 3. Happy path
# ===========================================================================

class TestHappyPath:

    def test_indices_resolve_to_real_standard_ids(self, stub_llm, request_obj):
        stub_llm({
            "findings": [
                {
                    "requirement_index": 0,
                    "verdict": "justified",
                    "reason": "IS 10322 applies.",
                    "recommended_action": None,
                    "applicable_standard_indices": [0],
                    "confidence": 0.9,
                },
                {
                    "requirement_index": 1,
                    "verdict": "missing_requirement",
                    "reason": "IS 694 covers PVC cables but is not cited.",
                    "recommended_action": "Cite IS 694.",
                    "applicable_standard_indices": [1, 0],
                    "confidence": 0.7,
                },
            ]
        })
        response = ga.run_gemini_analysis(request_obj)

        assert isinstance(response, AimlResponse)
        assert response.analysis_id == "an-1"
        assert len(response.findings) == 2

        first, second = response.findings
        assert first.requirement_id == "req-11111111-a"
        assert first.verdict == Verdict.JUSTIFIED.value
        assert first.applicable_standard_ids == ["std-aaa"]
        assert first.recommended_action is None
        assert first.confidence == 0.9

        assert second.requirement_id == "req-22222222-b"
        assert second.verdict == Verdict.MISSING_REQUIREMENT.value
        assert second.applicable_standard_ids == ["std-bbb", "std-aaa"]
        assert second.recommended_action == "Cite IS 694."

    def test_findings_are_ordered_by_requirement(self, stub_llm, request_obj):
        """Order follows the requirements, not the order Gemini replied in."""
        stub_llm({
            "findings": [
                {"requirement_index": 1, "verdict": "ambiguous", "reason": "b",
                 "applicable_standard_indices": [], "confidence": 0.4},
                {"requirement_index": 0, "verdict": "justified", "reason": "a",
                 "applicable_standard_indices": [0], "confidence": 0.9},
            ]
        })
        response = ga.run_gemini_analysis(request_obj)
        assert [f.requirement_id for f in response.findings] == [
            "req-11111111-a", "req-22222222-b"
        ]

    def test_findings_never_carry_evidence_ids(self, stub_llm, request_obj):
        """Evidence is resolved deterministically by the backend, not by the LLM."""
        stub_llm(_payload())
        response = ga.run_gemini_analysis(request_obj)
        assert all(f.evidence_ids == [] for f in response.findings)

    def test_finding_ids_are_unique(self, stub_llm, request_obj):
        stub_llm(_payload())
        response = ga.run_gemini_analysis(request_obj)
        ids = [f.finding_id for f in response.findings]
        assert len(set(ids)) == len(ids)

    def test_metadata_records_the_gemini_run(self, stub_llm, request_obj):
        stub_llm(_payload())
        response = ga.run_gemini_analysis(request_obj)
        meta = response.extraction_metadata
        assert meta["execution_mode"] == "gemini"
        assert meta["model"] == "stub-model"
        assert meta["gemini_calls"] == 1
        assert meta["gemini_batch_failures"] == 0
        assert meta["requirements_count"] == 2
        assert meta["retrieved_standards_count"] == 2

    def test_empty_standard_indices_is_a_valid_answer(self, stub_llm, request_obj):
        """"None of these apply" must survive as an empty list, not be back-filled."""
        stub_llm(_payload(verdict="unsupported", applicable_standard_indices=[]))
        response = ga.run_gemini_analysis(request_obj)
        assert response.findings[0].applicable_standard_ids == []
        assert response.findings[0].verdict == Verdict.UNSUPPORTED.value


# ===========================================================================
# 4. Hostile / malformed payloads
# ===========================================================================

class TestPayloadValidation:

    @pytest.mark.parametrize("bad_index", [2, 99, -1])
    def test_out_of_range_standard_index_is_dropped(self, stub_llm, request_obj, bad_index):
        """Only 2 standards were shown — index 2+ or negative cannot be cited."""
        stub_llm(_payload(applicable_standard_indices=[bad_index, 0]))
        response = ga.run_gemini_analysis(request_obj)
        assert response.findings[0].applicable_standard_ids == ["std-aaa"]

    def test_duplicate_standard_indices_are_deduplicated(self, stub_llm, request_obj):
        stub_llm(_payload(applicable_standard_indices=[0, 0, 1, 1]))
        response = ga.run_gemini_analysis(request_obj)
        assert response.findings[0].applicable_standard_ids == ["std-aaa", "std-bbb"]

    def test_out_of_range_requirement_index_is_dropped(self, stub_llm, request_obj):
        stub_llm({
            "findings": [
                {"requirement_index": 7, "verdict": "justified", "reason": "ghost",
                 "applicable_standard_indices": [0], "confidence": 0.9},
            ]
        })
        response = ga.run_gemini_analysis(request_obj)
        # Two requirements in, two findings out — both unable_to_determine.
        assert len(response.findings) == 2
        assert all(
            f.verdict == Verdict.UNABLE_TO_DETERMINE.value for f in response.findings
        )
        assert all("ghost" not in f.reason for f in response.findings)

    def test_skipped_requirement_gets_unable_to_determine(self, stub_llm, request_obj):
        """A requirement Gemini ignores must not vanish from the report."""
        stub_llm(_payload(requirement_index=0))
        response = ga.run_gemini_analysis(request_obj)
        assert len(response.findings) == 2
        assert response.findings[0].verdict == Verdict.JUSTIFIED.value
        second = response.findings[1]
        assert second.requirement_id == "req-22222222-b"
        assert second.verdict == Verdict.UNABLE_TO_DETERMINE.value
        assert second.confidence == 0.0
        assert second.applicable_standard_ids == []

    def test_duplicate_requirement_index_keeps_the_first(self, stub_llm, request_obj):
        stub_llm({
            "findings": [
                {"requirement_index": 0, "verdict": "justified", "reason": "first",
                 "applicable_standard_indices": [0], "confidence": 0.9},
                {"requirement_index": 0, "verdict": "unsupported", "reason": "second",
                 "applicable_standard_indices": [], "confidence": 0.1},
            ]
        })
        response = ga.run_gemini_analysis(request_obj)
        assert len(response.findings) == 2
        assert response.findings[0].reason == "first"

    def test_unknown_verdict_falls_back_without_losing_the_batch(self, stub_llm, request_obj):
        """Verdict drift must cost one verdict, not the whole call."""
        stub_llm({
            "findings": [
                {"requirement_index": 0, "verdict": "TOTALLY_MADE_UP", "reason": "a",
                 "applicable_standard_indices": [0], "confidence": 0.9},
                {"requirement_index": 1, "verdict": "justified", "reason": "b",
                 "applicable_standard_indices": [1], "confidence": 0.8},
            ]
        })
        response = ga.run_gemini_analysis(request_obj)
        assert response.findings[0].verdict == Verdict.UNABLE_TO_DETERMINE.value
        # The cited standard survives — only the verdict label was unusable.
        assert response.findings[0].applicable_standard_ids == ["std-aaa"]
        assert response.findings[1].verdict == Verdict.JUSTIFIED.value

    @pytest.mark.parametrize(
        "raw,expected", [(1.7, 1.0), (-0.5, 0.0), (0.63, 0.63)]
    )
    def test_confidence_is_clamped(self, stub_llm, request_obj, raw, expected):
        stub_llm(_payload(confidence=raw))
        response = ga.run_gemini_analysis(request_obj)
        assert response.findings[0].confidence == expected

    def test_blank_reason_gets_a_placeholder(self, stub_llm, request_obj):
        """AimlFinding.reason is surfaced to the user; it must never be empty."""
        stub_llm(_payload(reason="   "))
        response = ga.run_gemini_analysis(request_obj)
        assert response.findings[0].reason.strip()

    def test_blank_recommended_action_becomes_none(self, stub_llm, request_obj):
        stub_llm(_payload(recommended_action="  "))
        response = ga.run_gemini_analysis(request_obj)
        assert response.findings[0].recommended_action is None

    @pytest.mark.parametrize("payload", [
        {"not_findings": []},
        {"findings": "a string"},
        {"findings": [{"verdict": "justified"}]},          # missing required fields
        [{"requirement_index": 0}],                        # list, not object
    ])
    def test_structurally_invalid_payload_raises(self, stub_llm, request_obj, payload):
        stub_llm(payload)
        with pytest.raises(AimlResponseError) as exc:
            ga.run_gemini_analysis(request_obj)
        assert exc.value.code == "GEMINI_ALL_BATCHES_FAILED"

    def test_zero_findings_yields_unable_to_determine_for_all(self, stub_llm, request_obj):
        stub_llm({"findings": []})
        response = ga.run_gemini_analysis(request_obj)
        assert len(response.findings) == 2
        assert all(
            f.verdict == Verdict.UNABLE_TO_DETERMINE.value for f in response.findings
        )


# ===========================================================================
# 5. Batching and partial failure
# ===========================================================================

def _many_requirements(count: int) -> list[Requirement]:
    return [
        Requirement(id=f"req-{i:08d}", analysis_id="an-1", text=f"Requirement {i}.")
        for i in range(count)
    ]


class TestBatching:

    def test_requirements_are_split_across_calls(self, stub_llm, standards):
        count = ga._MAX_REQUIREMENTS_PER_CALL + 3
        request = AimlRequest(
            analysis_id="an-1",
            extracted_text="",
            requirements=_many_requirements(count),
            retrieved_standards=standards,
        )
        # Each batch is answered for its own local index 0 only; the rest fill in.
        stub = stub_llm(_payload())
        response = ga.run_gemini_analysis(request)

        assert len(stub.prompts) == 2
        assert len(response.findings) == count
        assert response.extraction_metadata["gemini_calls"] == 2
        # Requirement IDs are still 1:1 and in order across the batch boundary.
        assert [f.requirement_id for f in response.findings] == [
            r.id for r in request.requirements
        ]

    def test_batch_indices_are_local_to_the_batch(self, stub_llm, standards):
        """
        Batch 2's requirement_index 0 must resolve to the 13th requirement,
        not the 1st — an off-by-batch error here would attach verdicts to the
        wrong requirements.
        """
        count = ga._MAX_REQUIREMENTS_PER_CALL + 1
        request = AimlRequest(
            analysis_id="an-1",
            extracted_text="",
            requirements=_many_requirements(count),
            retrieved_standards=standards,
        )
        stub_llm(_payload(reason="answered"))
        response = ga.run_gemini_analysis(request)

        last = response.findings[ga._MAX_REQUIREMENTS_PER_CALL]
        assert last.requirement_id == request.requirements[-1].id
        assert last.reason == "answered"

    def test_one_failed_batch_does_not_discard_the_others(self, stub_llm, standards):
        count = ga._MAX_REQUIREMENTS_PER_CALL + 1
        request = AimlRequest(
            analysis_id="an-1",
            extracted_text="",
            requirements=_many_requirements(count),
            retrieved_standards=standards,
        )
        stub_llm([
            _payload(reason="batch one answered"),
            AnalysisError("quota gone", code="LLM_QUOTA_EXHAUSTED"),
        ])
        response = ga.run_gemini_analysis(request)

        assert len(response.findings) == count
        assert response.findings[0].reason == "batch one answered"
        assert response.extraction_metadata["gemini_batch_failures"] == 1
        # The failed batch's requirement is present, flagged, not dropped.
        assert response.findings[-1].verdict == Verdict.UNABLE_TO_DETERMINE.value

    def test_all_batches_failing_raises(self, stub_llm, request_obj):
        stub_llm(AnalysisError("gemini down", code="LLM_CALL_FAILED"))
        with pytest.raises(AimlResponseError) as exc:
            ga.run_gemini_analysis(request_obj)
        assert exc.value.code == "GEMINI_ALL_BATCHES_FAILED"

    def test_unconfigured_llm_raises(self, monkeypatch, request_obj):
        def _boom():
            raise AnalysisError("no key", code="LLM_NOT_CONFIGURED")

        monkeypatch.setattr("kartikey.analysis.llm_client.get_llm_client", _boom)
        with pytest.raises(AimlResponseError) as exc:
            ga.run_gemini_analysis(request_obj)
        assert exc.value.code == "LLM_NOT_CONFIGURED"

    def test_no_requirements_makes_no_calls(self, stub_llm, standards):
        request = AimlRequest(
            analysis_id="an-1", extracted_text="", requirements=[],
            retrieved_standards=standards,
        )
        stub = stub_llm(_payload())
        response = ga.run_gemini_analysis(request)
        assert stub.prompts == []
        assert response.findings == []


# ===========================================================================
# 6. AimlClient wiring
# ===========================================================================

class TestClientModeSelection:

    def test_http_url_wins_over_gemini(self):
        client = AimlClient(service_url="http://localhost:9000/analyze", use_gemini=True)
        assert client.execution_mode == "http"
        assert client.is_mock is False

    def test_gemini_used_when_no_http_url(self):
        client = AimlClient(service_url="", use_gemini=True)
        assert client.execution_mode == "gemini"
        assert client.is_mock is False

    def test_mock_when_gemini_disabled_and_no_url(self):
        client = AimlClient(service_url="", use_gemini=False)
        assert client.execution_mode == "mock"
        assert client.is_mock is True

    def test_force_mock_overrides_everything(self):
        client = AimlClient(
            service_url="http://localhost:9000/analyze", use_gemini=True, force_mock=True
        )
        assert client.execution_mode == "mock"
        assert client.is_mock is True

    @pytest.mark.asyncio
    async def test_gemini_mode_runs_the_final_pass(self, stub_llm, request_obj):
        stub_llm(_payload())
        client = AimlClient(service_url="", use_gemini=True)
        response = await client.run_analysis(request_obj)
        assert response.extraction_metadata["execution_mode"] == "gemini"
        assert client.last_engine == "gemini"
        assert client.fallback_reason is None

    @pytest.mark.asyncio
    async def test_gemini_failure_raises_error(self, stub_llm, request_obj):
        """Without a mock fallback, Gemini outage fails the analysis."""
        stub_llm(AnalysisError("gemini down", code="LLM_CALL_FAILED"))
        client = AimlClient(service_url="", use_gemini=True)
        with pytest.raises(AimlResponseError) as exc:
            await client.run_analysis(request_obj)
        assert exc.value.code == "GEMINI_ALL_BATCHES_FAILED"


class TestEngineLadder:
    """http → gemini → mock, with each step down recorded."""

    @pytest.mark.asyncio
    async def test_unreachable_http_service_falls_back_to_gemini(
        self, stub_llm, request_obj
    ):
        """
        AIML_SERVICE_URL usually points at a localhost ai-engine. When that is
        not running, a valid Gemini key should still produce a real analysis.
        """
        stub_llm(_payload())
        # Port 1 is reserved and never listening, so the connection genuinely fails.
        client = AimlClient(
            service_url="http://127.0.0.1:1/analyze", use_gemini=True, timeout=2.0
        )
        assert client.execution_mode == "http"

        response = await client.run_analysis(request_obj)

        assert client.last_engine == "gemini"
        assert response.extraction_metadata["execution_mode"] == "gemini"
        assert "127.0.0.1:1" in client.fallback_reason

    @pytest.mark.asyncio
    async def test_unreachable_http_service_still_raises_without_gemini(
        self, request_obj
    ):
        """Without a fallback engine the caller must still see the failure."""
        client = AimlClient(
            service_url="http://127.0.0.1:1/analyze", use_gemini=False, timeout=2.0
        )
        with pytest.raises(AimlResponseError):
            await client.run_analysis(request_obj)

    @pytest.mark.asyncio
    async def test_both_http_and_gemini_failing_raises_error(
        self, stub_llm, request_obj
    ):
        stub_llm(AnalysisError("gemini down", code="LLM_CALL_FAILED"))
        client = AimlClient(
            service_url="http://127.0.0.1:1/analyze", use_gemini=True, timeout=2.0
        )
        with pytest.raises(AimlResponseError) as exc:
            await client.run_analysis(request_obj)
        assert exc.value.code == "GEMINI_ALL_BATCHES_FAILED"

    @pytest.mark.asyncio
    async def test_state_is_reset_between_runs(self, stub_llm, request_obj):
        """A stale fallback_reason would mislabel a later successful run."""
        stub_llm([AnalysisError("gemini down", code="LLM_CALL_FAILED"), _payload()])
        client = AimlClient(service_url="", use_gemini=True)

        with pytest.raises(AimlResponseError):
            await client.run_analysis(request_obj)

        await client.run_analysis(request_obj)
        # We did not fall back, so there's no fallback reason
        assert client.fallback_reason is None
        assert client.last_engine == "gemini"


class TestPromptEconomy:
    """The prompt should carry signal, not defaults."""

    def test_default_category_is_omitted(self, standards):
        from shared.models import RequirementCategory

        default = Requirement(id="r1", analysis_id="a", text="Default category.")
        assert default.category == RequirementCategory.OTHER
        prompt = ga._build_prompt([default], standards, "")
        assert "Category:" not in prompt

    def test_assigned_category_is_included(self, standards):
        from shared.models import RequirementCategory

        typed = Requirement(
            id="r1", analysis_id="a", text="Typed category.",
            category=RequirementCategory.SAFETY,
        )
        prompt = ga._build_prompt([typed], standards, "")
        assert "Category: safety" in prompt


class TestTransportFailures:
    """
    Failures below the SDK's error types must degrade, not crash.

    GeminiClient wraps API errors in AnalysisError, but a proxy 403, a DNS
    failure or a TLS reset surfaces raw from httpx through the model-resolution
    probe. Observed for real: httpx.ProxyError("403 Forbidden").
    """

    @pytest.mark.parametrize("exc", [
        __import__("httpx").ProxyError("403 Forbidden"),
        __import__("httpx").ConnectError("nodename nor servname provided"),
        RuntimeError("SDK changed shape"),
        ValueError("unexpected"),
    ])
    def test_raw_transport_exception_becomes_a_degraded_batch(
        self, stub_llm, request_obj, exc
    ):
        stub_llm(exc)
        with pytest.raises(AimlResponseError) as caught:
            ga.run_gemini_analysis(request_obj)
        assert caught.value.code == "GEMINI_ALL_BATCHES_FAILED"

    def test_raw_transport_exception_spares_the_other_batches(
        self, stub_llm, standards
    ):
        import httpx

        count = ga._MAX_REQUIREMENTS_PER_CALL + 1
        request = AimlRequest(
            analysis_id="an-1", extracted_text="",
            requirements=_many_requirements(count), retrieved_standards=standards,
        )
        stub_llm([_payload(reason="batch one answered"), httpx.ProxyError("403 Forbidden")])
        response = ga.run_gemini_analysis(request)

        assert len(response.findings) == count
        assert response.findings[0].reason == "batch one answered"
        assert response.extraction_metadata["gemini_batch_failures"] == 1

    @pytest.mark.asyncio
    async def test_transport_failure_raises_error(
        self, stub_llm, request_obj
    ):
        """End of the chain: the pipeline raises if Gemini transport fails."""
        import httpx

        stub_llm(httpx.ProxyError("403 Forbidden"))
        client = AimlClient(service_url="", use_gemini=True)
        with pytest.raises(AimlResponseError) as exc:
            await client.run_analysis(request_obj)
        assert exc.value.code == "GEMINI_ALL_BATCHES_FAILED"

    def test_client_construction_failure_is_wrapped(self, monkeypatch, request_obj):
        """A non-AnalysisError from get_llm_client must not escape raw."""
        def _boom():
            raise RuntimeError("SDK import exploded")

        monkeypatch.setattr("kartikey.analysis.llm_client.get_llm_client", _boom)
        with pytest.raises(AimlResponseError) as caught:
            ga.run_gemini_analysis(request_obj)
        assert caught.value.code == "GEMINI_CLIENT_UNAVAILABLE"

    def test_bug_in_our_own_validation_is_not_swallowed(
        self, stub_llm, request_obj, monkeypatch
    ):
        """
        The broad catch covers the LLM call only. A defect in our validation
        must fail loudly in tests rather than masquerade as an LLM outage.
        """
        stub_llm(_payload())

        def _broken(*args, **kwargs):
            raise TypeError("regression in _findings_from_payload")

        monkeypatch.setattr(ga, "_findings_from_payload", _broken)
        with pytest.raises(TypeError):
            ga.run_gemini_analysis(request_obj)
