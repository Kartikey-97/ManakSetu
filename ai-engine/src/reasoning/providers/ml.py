"""
ai-engine/src/reasoning/providers/ml.py

ML-based reasoning provider. Uses the trained applicability model to score
every retrieved candidate standard against a requirement and returns the top-N
most applicable ones (up to 3) rather than just the single best match.

Key design notes:
- `scored_candidates` is initialised before the loop that populates it.
- Rank-based features (embedding_cosine_sim_rank, bm25_score_rank, rrf_rank)
  are computed against the COMPLETE candidate universe so relative ordering
  is preserved. Passing a single candidate would make every candidate rank 1
  and destroy the signals the model was trained on.
- Threshold is 0.3 (inclusive) to allow moderately applicable standards
  to surface; the downstream assembler decides what to show the user.
"""

import logging
from src.reasoning.providers.base import ReasoningProvider
from src.ml.applicability_model import get_applicability_model
from src.ml.applicability_features import build_applicability_features

logger = logging.getLogger(__name__)

# Maximum number of standards to return per requirement.
_TOP_N = 5
# Minimum applicability score to be included in the result.
_MIN_SCORE = 0.3


class MLReasoner(ReasoningProvider):
    def analyze(self, req_text, req_type, standards, is_reference=None, cited_year=None):
        if not standards:
            return {
                "verdict": "unsupported",
                "reason": "No standard in the knowledge base covers this requirement's domain.",
                "action": "Review requirement for compliance with alternative domains.",
                "confidence": 0.0,
                "matched_is_number": None,
                "applicable_standard_ids": [],
            }

        model = get_applicability_model()
        if not model:
            raise RuntimeError("ApplicabilityModel is unavailable")

        # Build (std_object, candidate_dict) pairs.
        # candidate_dict contains the scalar fields the feature builder needs.
        candidate_pairs = []
        for std in standards:
            candidate_pairs.append(
                (
                    std,
                    {
                        "is_number": getattr(std, "is_number", ""),
                        "title": getattr(std, "title", ""),
                        "summary": "",
                        "scope": getattr(std, "scope", ""),
                        "search_text": getattr(std, "search_text", ""),
                        # The retrieval contract exposes semantic_score and
                        # relevance_score; raw BM25 is not available so we
                        # leave it as NaN and map relevance_score → rrf_score.
                        "bm25_score": getattr(std, "bm25_score", float("nan")),
                        "semantic_score": getattr(std, "semantic_score", float("nan")),
                        "rrf_score": getattr(std, "relevance_score", float("nan")),
                    },
                )
            )

        all_candidates = [c for _, c in candidate_pairs]

        # --- Feature extraction (one row per candidate) ---
        features_list = []
        valid_candidates = []

        for std, candidate_dict in candidate_pairs:
            try:
                features_df = build_applicability_features(
                    requirement_text=req_text,
                    candidate=candidate_dict,
                    all_candidates=all_candidates,
                )
                features_list.append(features_df)
                valid_candidates.append(std)
                logger.debug(
                    "ML features for %s: %s",
                    getattr(std, "is_number", ""),
                    features_df.to_dict(orient="records"),
                )
            except Exception as exc:
                logger.warning(
                    "ML feature extraction failed for standard %s: %s",
                    getattr(std, "is_number", ""),
                    exc,
                )

        # --- Batch prediction ---
        # scored_candidates is initialised here so it is always defined,
        # even if the feature list is empty or the predict call raises.
        scored_candidates = []

        if features_list:
            import pandas as pd
            batch_df = pd.concat(features_list, ignore_index=True)
            try:
                preds = model.predict(batch_df)
                if isinstance(preds, dict):
                    preds = [preds]

                for std, pred in zip(valid_candidates, preds):
                    score = pred.get("applicability_score")
                    logger.debug(
                        "ML prediction for %s: score=%s",
                        getattr(std, "is_number", ""),
                        score,
                    )
                    if score is not None:
                        scored_candidates.append((float(score), std))

            except Exception as exc:
                logger.warning("Batch ML prediction failed: %s", exc)

        # --- Select top-N candidates above threshold ---
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = [
            (score, std)
            for score, std in scored_candidates
            if score >= _MIN_SCORE
        ][:_TOP_N]

        if top_candidates:
            best_score, best_std = top_candidates[0]
            ids = [
                std.id if hasattr(std, "id") else std.get("id")
                for _, std in top_candidates
            ]
            logger.info(
                "ML matched requirement to %d standard(s): %s (top score=%.2f)",
                len(ids),
                ", ".join(
                    getattr(std, "is_number", "?") for _, std in top_candidates
                ),
                best_score,
            )
            return {
                "verdict": "justified",
                "reason": (
                    f"The ML applicability model (v2) matched this requirement to "
                    f"{best_std.is_number} (score={best_score:.2f})."
                ),
                "action": "Requirement verified against applicable standard.",
                "confidence": best_score,
                "matched_is_number": best_std.is_number,
                "applicable_standard_ids": ids,
            }

        elif scored_candidates:
            # All candidates scored below threshold — return the best one as a
            # low-confidence hint but do not commit to a standard.
            best_score, best_std = scored_candidates[0]
            return {
                "verdict": "requires_human_verification",
                "reason": (
                    f"The ML applicability model (v2) could not confidently match "
                    f"this requirement (best match: {best_std.is_number} at {best_score:.2f})."
                ),
                "action": "Manually verify specification against applicable standards.",
                "confidence": max(0.1, best_score),
                "matched_is_number": None,
                "applicable_standard_ids": [],
            }

        else:
            return {
                "verdict": "requires_human_verification",
                "reason": "The ML applicability model could not score the candidates.",
                "action": "Manually verify specification against applicable standards.",
                "confidence": 0.0,
                "matched_is_number": None,
                "applicable_standard_ids": [],
            }
