"""
kartikey/analysis/test_llm_client.py

Tests for the Gemini transport wrapper.

These exist because of a real failure: a live run of the Gemini final pass
degraded to deterministic analysis in under a second with the message

    "This model is currently experiencing high demand. Spikes in demand are
     usually temporary. Please try again later."

That is a 503 ServerError. It escaped raw from the old eager `_resolve_model()`
probe, which ran *before* the retry loop and only caught ClientError — so a
transient overload on the first model took the whole LLM path down without
retrying and without ever trying gemini-3.6-flash-lite.

The invariants below pin that down: one sweep tries every candidate model, a
sweep never sleeps between models, 404 rules a model out permanently, quota and
overload fail over, and no SDK exception ever escapes un-wrapped.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors as genai_errors

from shared.config import settings
from shared.utils import AnalysisError

from kartikey.analysis import llm_client as L
from kartikey.analysis.llm_client import GeminiClient

OK = '{"ok": true}'

# The four names are read off the live priority chain rather than written out.
# Nothing here is a test of *which* models ship — that list changes whenever
# Google deprecates a version or the team points GEMINI_MODEL somewhere else,
# and hardcoding it made every such edit look like a failover regression. What
# the tests below actually pin is the order the chain is walked in and what
# each error class does to it, so position is the only property they need.
assert len(L._MODEL_PRIORITY) == 4, (
    "These tests script one response per position in the fallback chain. "
    f"The chain is now {L._MODEL_PRIORITY!r} — add or remove positions below "
    "to match, rather than letting the extra models go unscripted."
)
FLASH, FLASH_LITE, LATEST, LITE_LATEST = L._MODEL_PRIORITY


# ===========================================================================
# Stubs
# ===========================================================================

def _server_error(
    message: str = (
        "This model is currently experiencing high demand. Spikes in demand are "
        "usually temporary. Please try again later."
    ),
) -> genai_errors.ServerError:
    """The exact error class and payload shape that broke the live run."""
    return genai_errors.ServerError(
        503, {"error": {"code": 503, "status": "UNAVAILABLE", "message": message}}
    )


def _client_error(code: int, status: str, message: str) -> genai_errors.ClientError:
    return genai_errors.ClientError(
        code, {"error": {"code": code, "status": status, "message": message}}
    )


def _not_found(model: str) -> genai_errors.ClientError:
    return _client_error(
        404, "NOT_FOUND", f"models/{model} is not found for API version v1beta"
    )


def _rate_limited() -> genai_errors.ClientError:
    return _client_error(429, "RESOURCE_EXHAUSTED", "Quota exceeded for this model.")


class _StubModels:
    """
    Replays a scripted outcome per model name.

    `script` maps a model name to a list of outcomes consumed in order; the last
    outcome repeats once the list runs dry. An outcome is either a `str` (the
    response text) or an `Exception` instance (raised). Models absent from the
    script get `default`.
    """

    def __init__(self, script: dict[str, list], default=None) -> None:
        self.script = {model: list(outcomes) for model, outcomes in script.items()}
        self.default = default if default is not None else _not_found("unscripted")
        self.calls: list[SimpleNamespace] = []
        self.current_key: str | None = None

    def generate_content(self, *, model, contents, config=None):
        self.calls.append(
            SimpleNamespace(
                model=model, contents=contents, config=config, key=self.current_key
            )
        )
        outcomes = self.script.get(model)
        if not outcomes:
            outcome = self.default
        else:
            outcome = outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(text=outcome)

    @property
    def models_tried(self) -> list[str]:
        return [c.model for c in self.calls]


@pytest.fixture
def make_client(monkeypatch):
    """Build a GeminiClient whose transport is a scripted stub."""

    sleeps: list[float] = []
    monkeypatch.setattr(L, "time", SimpleNamespace(sleep=sleeps.append))

    def _make(script: dict[str, list], default=None, keys: tuple[str, ...] = ("key-1",)):
        monkeypatch.setattr(settings, "google_api_key", keys[0])
        monkeypatch.setattr(settings, "google_api_key_2", keys[1] if len(keys) > 1 else "")
        monkeypatch.setattr(settings, "google_api_key_3", keys[2] if len(keys) > 2 else "")

        client = GeminiClient()
        stub = _StubModels(script, default)

        def _get_client(api_key: str | None = None):
            stub.current_key = api_key or client._api_keys[0]
            return SimpleNamespace(models=stub)

        monkeypatch.setattr(client, "_get_client", _get_client)
        return client, stub

    _make.sleeps = sleeps
    return _make


# ===========================================================================
# 1. The regression that prompted this file
# ===========================================================================

class TestOverloadedModel:
    """503 "high demand" must fail over, not abort the LLM path."""

    def test_transient_503_fails_over_to_the_next_model(self, make_client):
        client, stub = make_client({FLASH: [_server_error()], FLASH_LITE: [OK]})

        assert client.generate_json(prompt="p") == {"ok": True}
        assert stub.models_tried == [FLASH, FLASH_LITE]
        # The whole point: failover is immediate. Sleeping on a model that is
        # overloaded right now only adds latency to a user-facing request.
        assert make_client.sleeps == []

    def test_503_everywhere_raises_analysis_error_not_a_raw_sdk_error(self, make_client):
        """
        The live failure. `AimlClient` and the pipeline both key off
        AnalysisError; a raw ServerError escaping here is what made the final
        pass give up in one second.
        """
        client, stub = make_client(
            {m: [_server_error()] for m in (FLASH, FLASH_LITE, LATEST, LITE_LATEST)}
        )

        with pytest.raises(AnalysisError) as exc_info:
            client.generate_json(prompt="p")

        assert exc_info.value.code == "LLM_CALL_FAILED"
        assert "high demand" in str(exc_info.value)
        # 3 sweeps x 4 models, and a sleep between sweeps but not within one.
        assert len(stub.calls) == 12
        assert make_client.sleeps == [2.0, 4.0]

    def test_503_recovers_on_a_later_sweep(self, make_client):
        client, stub = make_client(
            {
                FLASH: [_server_error(), _server_error(), OK],
                FLASH_LITE: [_server_error()],
                LATEST: [_server_error()],
                LITE_LATEST: [_server_error()],
            }
        )

        assert client.generate_json(prompt="p") == {"ok": True}
        assert client._working_model == FLASH


# ===========================================================================
# 2. Model selection
# ===========================================================================

class TestModelSelection:

    def test_no_throwaway_probe_call_is_made(self, make_client):
        """
        Model selection happens against the real request. The old code spent an
        extra API call per process pinging, and made every call depend on that
        unrelated ping succeeding.
        """
        client, stub = make_client({FLASH: [OK]})

        client.generate_json(prompt="the real prompt")

        assert len(stub.calls) == 1
        assert stub.calls[0].contents == "the real prompt"
        assert "ping" not in stub.models_tried

    def test_404_rules_a_model_out_for_the_rest_of_the_call(self, make_client):
        client, stub = make_client(
            {
                FLASH: [_not_found(FLASH)],
                FLASH_LITE: [_server_error()],
                LATEST: [OK],
            }
        )

        assert client.generate_json(prompt="p") == {"ok": True}
        # flash asked once and never again; flash-lite retried on the same sweep.
        assert stub.models_tried.count(FLASH) == 1

    def test_all_models_404_reports_no_model_available(self, make_client):
        client, stub = make_client(
            {m: [_not_found(m)] for m in (FLASH, FLASH_LITE, LATEST, LITE_LATEST)}
        )

        with pytest.raises(AnalysisError) as exc_info:
            client.generate_json(prompt="p")

        assert exc_info.value.code == "LLM_NO_MODEL_AVAILABLE"
        # One sweep only — a 404 will not become a 200 in two seconds.
        assert len(stub.calls) == 4
        assert make_client.sleeps == []

    def test_working_model_is_cached_but_the_chain_stays_available(self, make_client):
        client, stub = make_client({FLASH: [OK, _server_error()], FLASH_LITE: [OK]})

        client.generate_json(prompt="first")
        assert client._working_model == FLASH

        # flash now 503s. A cached model must not become a single point of failure.
        assert client.generate_json(prompt="second") == {"ok": True}
        assert client._working_model == FLASH_LITE

    def test_cached_model_is_tried_first(self, make_client):
        client, stub = make_client({FLASH: [OK], LITE_LATEST: [OK]})
        client._working_model = LITE_LATEST

        client.generate_json(prompt="p")

        assert stub.models_tried == [LITE_LATEST]


# ===========================================================================
# 3. Quota
# ===========================================================================

class TestQuota:

    def test_429_on_one_model_fails_over_to_the_next(self, make_client):
        """Free-tier quota is per model; a rate-limited flash says nothing
        about flash-lite."""
        client, stub = make_client({FLASH: [_rate_limited()], FLASH_LITE: [OK]})

        assert client.generate_json(prompt="p") == {"ok": True}
        assert stub.models_tried == [FLASH, FLASH_LITE]

    def test_429_everywhere_reports_quota_exhausted(self, make_client):
        client, stub = make_client(
            {m: [_rate_limited()] for m in (FLASH, FLASH_LITE, LATEST, LITE_LATEST)}
        )

        with pytest.raises(AnalysisError) as exc_info:
            client.generate_json(prompt="p")

        assert exc_info.value.code == "LLM_QUOTA_EXHAUSTED"

    def test_429_is_retried_with_the_next_key(self, make_client):
        """Quota is per key, so a retry must not reuse the key that just 429'd."""
        client, stub = make_client(
            {m: [_rate_limited()] for m in (FLASH, FLASH_LITE, LATEST, LITE_LATEST)},
            keys=("key-1", "key-2"),
        )

        with pytest.raises(AnalysisError):
            client.generate_json(prompt="p")

        assert {"key-1", "key-2"} == {c.key for c in stub.calls}

    def test_extra_keys_come_from_settings(self, make_client):
        """
        Regression: these were read with os.getenv, but pydantic-settings parses
        .env without exporting it to os.environ — so a configured
        GOOGLE_API_KEY_2 was silently ignored and rotation never happened.
        """
        client, _ = make_client({FLASH: [OK]}, keys=("a", "b", "c"))
        assert client._api_keys == ["a", "b", "c"]

    def test_missing_key_is_reported_as_not_configured(self, monkeypatch):
        monkeypatch.setattr(settings, "google_api_key", "")
        monkeypatch.setattr(settings, "google_api_key_2", "")
        monkeypatch.setattr(settings, "google_api_key_3", "")

        with pytest.raises(AnalysisError) as exc_info:
            GeminiClient()

        assert exc_info.value.code == "LLM_NOT_CONFIGURED"


# ===========================================================================
# 4. Non-retryable errors
# ===========================================================================

class TestNonRetryableErrors:

    def test_other_4xx_fails_immediately_without_burning_quota(self, make_client):
        """A 400 is a problem with our request. Trying three more models would
        cost quota and hide the cause."""
        client, stub = make_client(
            {FLASH: [_client_error(400, "INVALID_ARGUMENT", "response_schema is invalid")]}
        )

        with pytest.raises(AnalysisError) as exc_info:
            client.generate_json(prompt="p")

        assert exc_info.value.code == "LLM_CALL_FAILED"
        assert "response_schema is invalid" in str(exc_info.value)
        assert len(stub.calls) == 1

    def test_transport_failure_is_retried_then_wrapped(self, make_client):
        """
        A proxy 403 / DNS / TLS failure is an httpx exception, not an SDK one.
        These must be retried and then wrapped — callers downstream only handle
        AnalysisError.
        """
        client, stub = make_client(
            {m: [httpx.ProxyError("403 Forbidden")] for m in
             (FLASH, FLASH_LITE, LATEST, LITE_LATEST)}
        )

        with pytest.raises(AnalysisError) as exc_info:
            client.generate_json(prompt="p")

        assert exc_info.value.code == "LLM_CALL_FAILED"
        assert "403 Forbidden" in str(exc_info.value)
        # One call per sweep, not four: an unreachable network is not a property
        # of the model, so walking the chain would repeat the same failure.
        assert stub.models_tried == [FLASH, FLASH, FLASH]

    def test_transport_failure_recovers_on_a_later_sweep(self, make_client):
        client, stub = make_client(
            {
                FLASH: [httpx.ConnectError("connection reset"), OK],
                FLASH_LITE: [httpx.ConnectError("connection reset")],
                LATEST: [httpx.ConnectError("connection reset")],
                LITE_LATEST: [httpx.ConnectError("connection reset")],
            }
        )

        assert client.generate_json(prompt="p") == {"ok": True}


# ===========================================================================
# 5. Response handling
# ===========================================================================

class TestResponseHandling:

    def test_markdown_fenced_json_is_parsed(self, make_client):
        client, _ = make_client({FLASH: ['```json\n{"ok": true}\n```']})
        assert client.generate_json(prompt="p") == {"ok": True}

    def test_non_json_output_keeps_its_own_error_code(self, make_client):
        """
        LLM_PARSE_ERROR is documented but the old code flattened it to
        LLM_CALL_FAILED, so callers could not tell "the model rambled" from
        "the API is down".
        """
        client, stub = make_client({m: ["I cannot help with that."] for m in
                                    (FLASH, FLASH_LITE, LATEST, LITE_LATEST)})

        with pytest.raises(AnalysisError) as exc_info:
            client.generate_json(prompt="p")

        assert exc_info.value.code == "LLM_PARSE_ERROR"
        # Unlike a transport failure, non-JSON output *is* model-specific, so the
        # rest of the chain is worth trying.
        assert set(stub.models_tried) == {FLASH, FLASH_LITE, LATEST, LITE_LATEST}

    def test_empty_response_is_retried(self, make_client):
        """An empty response usually means MAX_TOKENS or a safety block — worth
        one more try before giving up."""
        client, stub = make_client({FLASH: ["", OK]})

        assert client.generate_json(prompt="p") == {"ok": True}
        assert stub.models_tried.count(FLASH) >= 2


# ===========================================================================
# 6. Request configuration
# ===========================================================================

class TestRequestConfig:

    def test_json_mode_and_temperature_are_forwarded(self, make_client):
        client, stub = make_client({FLASH: [OK]})

        client.generate_json(prompt="p", system_prompt="sys", temperature=0.2)

        config = stub.calls[0].config
        assert config.response_mime_type == "application/json"
        assert config.temperature == 0.2
        assert config.system_instruction == "sys"

    def test_response_schema_is_forwarded_when_given(self, make_client):
        from pydantic import BaseModel

        class _Schema(BaseModel):
            ok: bool

        client, stub = make_client({FLASH: [OK]})
        client.generate_json(prompt="p", response_schema=_Schema)

        assert stub.calls[0].config.response_schema is _Schema

    def test_no_schema_means_no_schema(self, make_client):
        client, stub = make_client({FLASH: [OK]})
        client.generate_json(prompt="p")
        assert stub.calls[0].config.response_schema is None

    def test_automatic_function_calling_is_disabled(self, make_client):
        """We pass no tools, so the SDK's AFC loop is dead weight — and it logs a
        warning on every cold start."""
        client, stub = make_client({FLASH: [OK]})
        client.generate_json(prompt="p")

        afc = stub.calls[0].config.automatic_function_calling
        assert afc is not None and afc.disable is True


# ===========================================================================
# 7. _resolve_model (the image-OCR route needs a model name up front)
# ===========================================================================

class TestResolveModel:

    def test_probe_survives_an_overloaded_first_model(self, make_client):
        """Same bug class as the 503 above: the probe caught only ClientError,
        so an overloaded flash aborted OCR entirely."""
        client, stub = make_client({FLASH: [_server_error()], FLASH_LITE: [OK]})

        assert client._resolve_model() == FLASH_LITE

    def test_probe_survives_a_transport_failure(self, make_client):
        client, _ = make_client(
            {FLASH: [httpx.ProxyError("403 Forbidden")], FLASH_LITE: [OK]}
        )

        assert client._resolve_model() == FLASH_LITE

    def test_probe_skips_missing_models(self, make_client):
        client, stub = make_client({FLASH: [_not_found(FLASH)], FLASH_LITE: [OK]})

        assert client._resolve_model() == FLASH_LITE
        assert stub.calls[0].contents == "ping"

    def test_probe_reuses_a_model_generate_json_already_proved(self, make_client):
        client, stub = make_client({FLASH: [OK]})
        client.generate_json(prompt="p")

        assert client._resolve_model() == FLASH
        assert len(stub.calls) == 1  # no probe needed

    def test_probe_reports_quota_exhaustion(self, make_client):
        client, _ = make_client(
            {m: [_rate_limited()] for m in (FLASH, FLASH_LITE, LATEST, LITE_LATEST)}
        )

        with pytest.raises(AnalysisError) as exc_info:
            client._resolve_model()

        assert exc_info.value.code == "LLM_QUOTA_EXHAUSTED"

    def test_probe_reports_no_model_available(self, make_client):
        client, _ = make_client(
            {m: [_not_found(m)] for m in (FLASH, FLASH_LITE, LATEST, LITE_LATEST)}
        )

        with pytest.raises(AnalysisError) as exc_info:
            client._resolve_model()

        assert exc_info.value.code == "LLM_NO_MODEL_AVAILABLE"
