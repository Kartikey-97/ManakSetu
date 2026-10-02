"""
Requirement-specific candidate isolation in Analyzer.process.

Each requirement must be reasoned against, and mapped to, only the standards
retrieved for it (request.requirement_candidates), never the pooled
request.retrieved_standards.
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.main import AimlRequest
from src.reasoning.analyzer import Analyzer
from src.reasoning.providers.mock import MockReasoner


class RecordingProvider:
    """Records the standards each requirement saw; returns a scripted result."""

    def __init__(self, results=None):
        self.results = results or {}
        self.seen = {}

    def analyze(self, req_text, req_type, standards, is_reference=None, cited_year=None):
        self.seen[req_text] = [(s.id, s.relevance_score) for s in standards]
        return dict(self.results.get(req_text, {
            "verdict": "requires_human_verification",
            "reason": "stub",
            "confidence": 0.5,
            "applicable_standard_ids": [],
        }))


def _std(sid, is_number, score=None, status="active"):
    return {"id": sid, "is_number": is_number, "title": is_number,
            "status": status, "relevance_score": score}


def _req(rid, text, is_reference=None):
    return {"id": rid, "analysis_id": "a1", "text": text, "is_reference": is_reference}


def _analyzer(provider, monkeypatch):
    monkeypatch.setenv("AI_MODE", "mock")
    analyzer = Analyzer()
    analyzer.provider = provider
    return analyzer


def _request(requirements, candidates, pool=None):
    return AimlRequest.model_validate({
        "analysis_id": "a1",
        "extracted_text": "tender",
        "requirements": requirements,
        "retrieved_standards": pool if pool is not None else [
            s for stds in candidates.values() for s in stds
        ],
        "requirement_candidates": candidates,
    })


def _ids(response, rid):
    return next(
        f["applicable_standard_ids"] for f in response["findings"] if f["requirement_id"] == rid
    )


# A: IS1 (cited), IS2, shared IS3 at 0.2.  B: shared IS3 at 0.9, IS4.
A_CANDS = [_std("s1", "IS 1001", 1.0), _std("s2", "IS 1002", 0.6), _std("s3", "IS 1003", 0.2)]
B_CANDS = [_std("s3", "IS 1003", 0.9), _std("s4", "IS 1004", 0.5)]
REQS = [_req("rA", "req A", "IS 1001"), _req("rB", "req B")]


def test_a_each_requirement_sees_only_its_own_candidates(monkeypatch):
    """A + C: disjoint inputs; A-only standards never reach B's reasoning."""
    provider = RecordingProvider()
    _analyzer(provider, monkeypatch).process(
        _request(REQS, {"rA": A_CANDS, "rB": B_CANDS})
    )

    assert [sid for sid, _ in provider.seen["req A"]] == ["s1", "s2", "s3"]
    assert [sid for sid, _ in provider.seen["req B"]] == ["s3", "s4"]
    assert not {"s1", "s2"} & {sid for sid, _ in provider.seen["req B"]}


def test_b_shared_standard_keeps_per_requirement_score(monkeypatch):
    provider = RecordingProvider()
    _analyzer(provider, monkeypatch).process(
        _request(REQS, {"rA": A_CANDS, "rB": B_CANDS})
    )

    assert dict(provider.seen["req A"])["s3"] == 0.2
    assert dict(provider.seen["req B"])["s3"] == 0.9


def test_d_explicit_empty_ids_stay_empty_despite_citation(monkeypatch):
    """A justified verdict with [] is not overridden by the cited IS 1001."""
    provider = RecordingProvider({"req A": {
        "verdict": "justified", "reason": "r", "confidence": 0.9,
        "applicable_standard_ids": [],
    }})
    response = _analyzer(provider, monkeypatch).process(
        _request(REQS, {"rA": A_CANDS, "rB": B_CANDS})
    )

    assert _ids(response, "rA") == []


def test_d_explicit_null_matched_is_number_stays_empty(monkeypatch):
    """Gemini-style "none applies" (matched_is_number null) stays empty."""
    provider = RecordingProvider({"req A": {
        "verdict": "justified", "reason": "r", "confidence": 0.9,
        "matched_is_number": None,
    }})
    response = _analyzer(provider, monkeypatch).process(
        _request(REQS, {"rA": A_CANDS, "rB": B_CANDS})
    )

    assert _ids(response, "rA") == []


def test_e_non_candidate_ids_are_dropped_not_widened(monkeypatch):
    provider = RecordingProvider({
        "req A": {"verdict": "justified", "reason": "r", "confidence": 0.9,
                  "applicable_standard_ids": ["s4", "unknown", "s2", "s2"]},
        "req B": {"verdict": "justified", "reason": "r", "confidence": 0.9,
                  "applicable_standard_ids": ["s1", "unknown"]},
    })
    response = _analyzer(provider, monkeypatch).process(
        _request(REQS, {"rA": A_CANDS, "rB": B_CANDS})
    )

    assert _ids(response, "rA") == ["s2"]
    # Every selection invalid → empty, never "all candidates".
    assert _ids(response, "rB") == []


def test_e_matched_is_number_outside_candidates_selects_nothing(monkeypatch):
    provider = RecordingProvider({"req B": {
        "verdict": "justified", "reason": "r", "confidence": 0.9,
        "matched_is_number": "IS 1001",   # A's standard, not B's
    }})
    response = _analyzer(provider, monkeypatch).process(
        _request(REQS, {"rA": A_CANDS, "rB": B_CANDS})
    )

    assert _ids(response, "rB") == []


def test_f_no_candidates_gets_no_pool_fallback(monkeypatch):
    """The pool holds the cited standard, but the requirement has no candidates."""
    provider = RecordingProvider({"req A": {
        "verdict": "justified", "reason": "r", "confidence": 0.9,
    }})
    response = _analyzer(provider, monkeypatch).process(
        _request(REQS, {}, pool=A_CANDS + B_CANDS)
    )

    assert provider.seen["req A"] == []
    assert provider.seen["req B"] == []
    assert _ids(response, "rA") == []
    assert _ids(response, "rB") == []


def test_missing_map_field_means_no_candidates(monkeypatch):
    """An old-shape payload validates, but is never reasoned against the pool."""
    provider = RecordingProvider()
    request = AimlRequest.model_validate({
        "analysis_id": "a1", "extracted_text": "t",
        "requirements": REQS, "retrieved_standards": A_CANDS,
    })
    _analyzer(provider, monkeypatch).process(request)

    assert provider.seen == {"req A": [], "req B": []}


def test_mock_reasoner_cites_only_among_own_candidates(monkeypatch):
    """The citation fallback applies only when the provider expresses nothing."""
    response = _analyzer(MockReasoner(), monkeypatch).process(
        _request(REQS, {"rA": A_CANDS, "rB": B_CANDS})
    )

    assert _ids(response, "rA") == ["s1"]
    assert _ids(response, "rB") == []


def test_citation_is_matched_by_base_number_not_substring(monkeypatch):
    """IS 1554 must not select IS 15544; a part of IS 1554 still matches."""
    reqs = [_req("rC", "req C", "IS 1554")]
    cands = {"rC": [_std("x1", "IS 15544", 0.9), _std("x2", "IS 1554 : Part 1", 1.0)]}
    response = _analyzer(MockReasoner(), monkeypatch).process(_request(reqs, cands))

    assert _ids(response, "rC") == ["x2"]


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_MODEL = os.path.join(_ROOT, "standiq_applicability_model_v2.joblib")
_META = os.path.join(_ROOT, "standiq_applicability_model_v2_metadata.json")


@pytest.mark.skipif(not os.path.exists(_MODEL), reason="applicability model not present")
def test_production_ml_reasoner_maps_only_within_own_candidates(monkeypatch):
    """AI_MODE=ml, as start.sh runs it: selections stay inside each requirement's set."""
    from src.ml.applicability_model import load_applicability_model
    load_applicability_model(model_path=_MODEL, metadata_path=_META)
    monkeypatch.setenv("AI_MODE", "ml")
    analyzer = Analyzer()
    assert analyzer._mode == "ml"

    a = [_std("a1", "IS 694", 1.0), _std("a2", "IS 1554 : Part 1", 0.6)]
    b = [_std("b1", "IS 10322 : Part 5 : Sec 3", 0.9), _std("b2", "IS 16107 : Part 1", 0.5)]
    a[0]["title"] = "PVC insulated cables for working voltages up to 1100 V"
    b[0]["title"] = "Luminaires for road and street lighting"
    reqs = [
        _req("rA", "Cables shall be PVC insulated conforming to IS 694", "IS 694"),
        _req("rB", "LED street light luminaire with IP66 enclosure"),
    ]
    response = analyzer.process(_request(reqs, {"rA": a, "rB": b}))

    assert set(_ids(response, "rA")) <= {"a1", "a2"}
    assert set(_ids(response, "rB")) <= {"b1", "b2"}


def test_outdated_check_only_uses_selected_candidates(monkeypatch):
    """A superseded standard in the pool but not selected cannot flag the finding."""
    provider = RecordingProvider({"req B": {
        "verdict": "justified", "reason": "r", "confidence": 0.9,
        "applicable_standard_ids": ["s4"],
    }})
    pool = A_CANDS + B_CANDS + [_std("s9", "IS 1009", 0.9, status="superseded")]
    response = _analyzer(provider, monkeypatch).process(
        _request(REQS, {"rA": A_CANDS, "rB": B_CANDS}, pool=pool)
    )

    finding = next(f for f in response["findings"] if f["requirement_id"] == "rB")
    assert finding["verdict"] == "justified"
