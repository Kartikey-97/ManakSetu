"""
ai-engine/src/embedding.py

Embedding generation via Google Gemini text-embedding-004.
Zero RAM on Render — no model loaded into process memory.
Free within Gemini API limits (1500 RPM on free tier).
768-dimensional output (better than 384-dim local models).
"""
from __future__ import annotations

import os
import time
import logging
from typing import Sequence

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 3072
_GEMINI_MODEL = "gemini-embedding-001"


def get_embedding(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """
    Get a single embedding. Raises RuntimeError if GOOGLE_API_KEY not set.
    Retries 3 times with exponential backoff.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")

    import google.genai as genai
    client = genai.Client(api_key=api_key)
    text = text[:8000]  # Safe truncation

    for attempt in range(3):
        try:
            result = client.models.embed_content(
                model=_GEMINI_MODEL,
                contents=text,
                config={"task_type": task_type},
            )
            return result.embeddings[0].values
        except Exception as exc:
            if attempt == 2:
                raise RuntimeError(f"Gemini embedding failed: {exc}") from exc
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError("Unreachable")


def get_embeddings_batch(
    texts: Sequence[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    """Batch embedding with rate-limiting. Logs progress every 10 items."""
    result = []
    for i, text in enumerate(texts):
        result.append(get_embedding(text, task_type=task_type))
        if i > 0 and i % 10 == 0:
            time.sleep(0.5)
            logger.info("Embedded %d/%d", i, len(texts))
    return result
