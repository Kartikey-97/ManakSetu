"""
ai-engine/tests/test_ml_applicability.py

Pytest tests for the ML applicability feature builder and model wrapper.
"""

import math
import os

import pandas as pd
import pytest

MODEL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "standiq_applicability_model_v2.joblib"
    )
)
METADATA_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "standiq_applicability_model_v2_metadata.json",
    )
)

EXPECTED_FEATURES = [
    "embedding_cosine_sim",
    "embedding_cosine_sim_rank",
    "bm25_score",
    "bm25_score_rank",
    "tfidf_cosine_sim",
    "token_jaccard",
    "technical_keyword_overlap_ratio",
    "numeric_param_match_ratio",
    "numeric_param_count_requirement",
    "numeric_param_count_standard",
    "unit_overlap_ratio",
    "rrf_score",
    "rrf_rank",
    "requirement_length_ratio",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_candidate():
    return {
        "is_number": "IS 10322",
        "title": "Luminaires for Road and Street Lighting",
        "scope": "Specifications for LED luminaires IP66 street lighting",
        "search_text": "LED luminaire street road IP66 90W",
        "semantic_score": 0.82,
        "bm25_score": 12.5,
        "rrf_score": 0.031,
    }


@pytest.fixture
def sample_candidates(sample_candidate):
    return [
        sample_candidate,
        {
            "is_number": "IS 2062",
            "title": "Steel",
            "scope": "Structural steel",
            "search_text": "steel",
            "semantic_score": 0.3,
            "bm25_score": 2.0,
            "rrf_score": 0.01,
        },
    ]


# ---------------------------------------------------------------------------
# Feature builder tests
# ---------------------------------------------------------------------------
def test_feature_builder_returns_correct_columns(sample_candidate, sample_candidates):
    from src.ml.applicability_features import build_applicability_features

    req = "LED street light luminaire IP66 90W"
    df = build_applicability_features(req, sample_candidate, sample_candidates)
    assert list(df.columns) == EXPECTED_FEATURES
    assert len(df) == 1


def test_feature_builder_uses_nan_for_missing():
    from src.ml.applicability_features import build_applicability_features

    candidate = {"is_number": "IS 999", "title": "Test", "scope": "", "search_text": ""}
    df = build_applicability_features("some requirement", candidate, [candidate])
    # bm25_score and rrf_score should be NaN since not in candidate
    assert math.isnan(df["bm25_score"].iloc[0])
    assert math.isnan(df["rrf_score"].iloc[0])


def test_existing_retrieval_fields_preserved(sample_candidate, sample_candidates):
    from src.ml.applicability_features import build_applicability_features

    orig_bm25 = sample_candidate["bm25_score"]
    orig_sem = sample_candidate["semantic_score"]
    build_applicability_features("LED street light", sample_candidate, sample_candidates)
    assert sample_candidate["bm25_score"] == orig_bm25
    assert sample_candidate["semantic_score"] == orig_sem


def test_rank_features_are_one_based(sample_candidate, sample_candidates):
    """The best candidate by semantic_score should have rank 1."""
    from src.ml.applicability_features import build_applicability_features

    df = build_applicability_features("LED", sample_candidate, sample_candidates)
    # sample_candidate has semantic_score=0.82 which is highest -> rank 1
    assert df["embedding_cosine_sim_rank"].iloc[0] == 1.0


def test_empty_requirement_text():
    """Empty requirement should not crash; some features become NaN."""
    from src.ml.applicability_features import build_applicability_features

    candidate = {
        "is_number": "IS 1",
        "title": "Some Title",
        "scope": "Some scope text",
        "search_text": "some text",
        "semantic_score": 0.5,
        "bm25_score": 1.0,
        "rrf_score": 0.02,
    }
    df = build_applicability_features("", candidate, [candidate])
    assert list(df.columns) == EXPECTED_FEATURES
    assert len(df) == 1


# ---------------------------------------------------------------------------
# Model tests (skip when model file absent)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model file not found")
def test_model_loads_and_predicts(sample_candidate, sample_candidates):
    from src.ml.applicability_model import load_applicability_model
    from src.ml.applicability_features import build_applicability_features

    model = load_applicability_model(MODEL_PATH, METADATA_PATH)
    assert model is not None

    feat_df = build_applicability_features(
        "LED street light luminaire IP66", sample_candidate, sample_candidates
    )
    result = model.predict(feat_df)

    assert "applicability_score" in result
    assert "applicability_class" in result
    assert result["applicability_class"] in ("APPLICABLE", "NOT_APPLICABLE")
    assert 0.0 <= result["applicability_score"] <= 1.0
    assert 0.0 <= result["feature_coverage"] <= 1.0


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model file not found")
def test_model_fallback_on_empty_df():
    from src.ml.applicability_model import load_applicability_model

    model = load_applicability_model(MODEL_PATH, METADATA_PATH)
    empty_df = pd.DataFrame([[float("nan")] * 14], columns=EXPECTED_FEATURES)
    result = model.predict(empty_df)
    # Must not raise -- fallback is acceptable
    assert "applicability_class" in result


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model file not found")
def test_model_version_populated():
    from src.ml.applicability_model import load_applicability_model

    model = load_applicability_model(MODEL_PATH, METADATA_PATH)
    assert model is not None
    assert model.model_version != ""


def test_load_model_returns_none_on_bad_path():
    from src.ml.applicability_model import load_applicability_model

    result = load_applicability_model(
        "/nonexistent/model.joblib", "/nonexistent/meta.json"
    )
    assert result is None


# ---------------------------------------------------------------------------
# G-1: ML applicability reranking must not discard an explicitly cited standard
# ---------------------------------------------------------------------------

def _candidate(is_number, applicability_score, relevance_score=0.5):
    return {
        "is_number": is_number,
        "title": f"Title for {is_number}",
        "applicability_score": applicability_score,
        "relevance_score": relevance_score,
    }


def test_select_top_k_preserves_explicitly_cited_standard():
    """A cited standard the reranker demoted below top_k must still survive."""
    from src.recommender import Recommender

    ranked = [
        _candidate("IS 9999", 0.90),
        _candidate("IS 8888", 0.80),
        _candidate("IS 7777", 0.70),
        _candidate("IS 6134 : Part 2", 0.10),  # cited, but demoted to last
    ]
    selected = Recommender._select_top_k(None, ranked, "IS 6134 : Part 2", 3)

    assert len(selected) == 3
    assert "IS 6134 : Part 2" in [c["is_number"] for c in selected]
    # Ordering stays reranker-driven; the cited standard is last, not promoted.
    assert [c["is_number"] for c in selected][-1] == "IS 6134 : Part 2"
    assert [c["rank"] for c in selected] == [1, 2, 3]


def test_select_top_k_matches_citation_despite_punctuation():
    """'IS 6134 Part 2' and 'IS 6134 : Part 2' denote the same standard."""
    from src.recommender import Recommender

    ranked = [_candidate("IS 9999", 0.9), _candidate("IS 6134 : Part 2", 0.1)]
    selected = Recommender._select_top_k(None, ranked, "IS 6134 Part 2", 1)

    assert [c["is_number"] for c in selected] == ["IS 6134 : Part 2"]


def test_select_top_k_is_plain_truncation_for_prose_queries():
    """No citation in the query means the reranker's top_k is returned as-is."""
    from src.recommender import Recommender

    ranked = [
        _candidate("IS 9999", 0.90),
        _candidate("IS 8888", 0.80),
        _candidate("IS 7777", 0.70),
    ]
    selected = Recommender._select_top_k(
        None, ranked, "LED street lighting luminaire", 2
    )

    assert [c["is_number"] for c in selected] == ["IS 9999", "IS 8888"]


def test_select_top_k_does_not_invent_candidates():
    """A citation that retrieval never surfaced must not appear in the output."""
    from src.recommender import Recommender

    ranked = [_candidate("IS 9999", 0.9), _candidate("IS 8888", 0.8)]
    selected = Recommender._select_top_k(None, ranked, "IS 16107", 1)

    assert [c["is_number"] for c in selected] == ["IS 9999"]
