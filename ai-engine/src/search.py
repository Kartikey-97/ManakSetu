"""
ai-engine/src/search.py

Semantic search via Qdrant Cloud.
Falls back to [] on ANY failure — never crashes the pipeline.
"""
from __future__ import annotations

import os
import logging
from typing import Any

logger = logging.getLogger(__name__)

COLLECTION_NAME = "bis_standards_v1"


def _get_client():
    """Lazy Qdrant client. Returns None if URL missing or connection fails."""
    url = os.environ.get("QDRANT_URL", "").strip()
    if not url:
        logger.warning("QDRANT_URL not set — semantic search disabled")
        return None
    try:
        from qdrant_client import QdrantClient
        return QdrantClient(
            url=url,
            api_key=os.environ.get("QDRANT_API_KEY"),
            timeout=10,
        )
    except Exception as exc:
        logger.error("Qdrant connect failed: %s", exc)
        return None


def semantic_search(
    query: str,
    top_k: int = 10,
    score_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Returns top_k standards matching the query.
    Returns [] on any error — NEVER raises.
    """
    if not query or not query.strip():
        return []

    client = _get_client()
    if client is None:
        return []

    try:
        from src.embedding import get_embedding
        vec = get_embedding(query.strip(), task_type="RETRIEVAL_QUERY")
    except Exception as exc:
        logger.warning("Embedding failed: %s", exc)
        return []

    try:
        hits = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vec,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        ).points
    except Exception as exc:
        logger.warning("Qdrant search failed: %s", exc)
        return []

    return [
        {
            "is_number": h.payload.get("is_number", ""),
            "title": h.payload.get("title", ""),
            "scope": h.payload.get("scope"),
            "score": h.score,
            "payload": h.payload,
        }
        for h in hits
        if h.payload
    ]
