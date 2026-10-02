"""
kshiraj/knowledge/embedding_service.py

Embedding service for semantic vector generation using Gemini API via direct REST.
Default model: gemini-embedding-001 (3072-dimensional dense vectors).

Uses `curl` subprocess to completely bypass Python's macOS IPv6 socket timeouts 
and asyncio/httpx deadlocks that happen with urllib or google-genai.
"""

from __future__ import annotations

import json
import math
import subprocess
from typing import Any, List, Optional

from shared.utils import AppError, get_logger
from shared.config import settings

logger = get_logger(__name__)


class EmbeddingServiceError(AppError):
    """Raised when embedding generation or model loading fails."""
    def __init__(self, message: str, code: str = "EMBEDDING_ERROR") -> None:
        super().__init__(message, code=code)


class EmbeddingService:
    """
    Service wrapper around Gemini REST API for semantic embedding generation.
    """

    DEFAULT_MODEL_NAME = "gemini-embedding-001"
    DEFAULT_DIMENSION = 3072
    API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent?key={key}"
    BATCH_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents?key={key}"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "auto",
        batch_size: int = 32,
        model: Optional[Any] = None,
    ) -> None:
        if not model_name or "BAAI/" in model_name or model_name == "text-embedding-004":
            model_name = self.DEFAULT_MODEL_NAME
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._dimension = self.DEFAULT_DIMENSION

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_api_key(self) -> str:
        api_key = settings.google_api_key_2 or settings.google_api_key
        if not api_key:
            raise EmbeddingServiceError("No GOOGLE_API_KEY available.")
        return api_key

    def _normalize_vector(self, vec: List[float]) -> List[float]:
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0:
            return vec
        return [x / norm for x in vec]

    def _curl_request(self, url: str, payload: dict) -> dict:
        """Helper to run request. Uses curl on macOS to bypass IPv6 DNS hangs, and native urllib on Linux/Production."""
        import sys
        if sys.platform == "darwin":
            try:
                # -4 forces IPv4, which immediately fixes macOS Python 75-second DNS hangs!
                cmd = ["curl", "-4", "-s", "-X", "POST", "-H", "Content-Type: application/json", "-d", json.dumps(payload), url]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                data = json.loads(result.stdout)
                if "error" in data:
                    raise Exception(data["error"])
                return data
            except subprocess.CalledProcessError as e:
                raise Exception(f"Curl failed with output: {e.stderr or e.stdout}")
            except Exception as e:
                raise Exception(f"Request failed: {e}")
        else:
            import urllib.request
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=30.0) as response:
                    data = json.loads(response.read().decode("utf-8"))
                    if "error" in data:
                        raise Exception(data["error"])
                    return data
            except Exception as e:
                raise Exception(f"Native request failed: {e}")


    def encode_text(self, text: str) -> List[float]:
        """Generate vector for a single query string using RETRIEVAL_QUERY."""
        if not text or not text.strip():
            return [0.0] * self.dimension

        url = self.API_URL_TEMPLATE.format(model=self.model_name, key=self._get_api_key())
        payload = {
            "model": f"models/{self.model_name}",
            "content": {"parts": [{"text": text[:8000]}]},
            "taskType": "RETRIEVAL_QUERY"
        }
        
        try:
            data = self._curl_request(url, payload)
            vec = data["embedding"]["values"]
            return self._normalize_vector(vec)
        except Exception as exc:
            logger.error("REST Embedding failed: %s", exc)
            raise EmbeddingServiceError(f"Embedding generation failed: {exc}") from exc

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate vectors for documents using RETRIEVAL_DOCUMENT."""
        if not texts:
            return []

        url = self.BATCH_API_URL_TEMPLATE.format(model=self.model_name, key=self._get_api_key())
        all_embeddings = []
        zero_vec = [0.0] * self.dimension

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            requests_list = []
            valid_indices = []
            
            for idx, t in enumerate(batch):
                if not t.strip():
                    continue
                valid_indices.append(idx)
                requests_list.append({
                    "model": f"models/{self.model_name}",
                    "content": {"parts": [{"text": t[:8000]}]},
                    "taskType": "RETRIEVAL_DOCUMENT"
                })
                
            if not requests_list:
                all_embeddings.extend([zero_vec] * len(batch))
                continue
                
            payload = {"requests": requests_list}
            try:
                data = self._curl_request(url, payload)
                batch_vectors = [zero_vec] * len(batch)
                for res_idx, valid_idx in enumerate(valid_indices):
                    vec = data["embeddings"][res_idx]["values"]
                    batch_vectors[valid_idx] = self._normalize_vector(vec)
                    
                all_embeddings.extend(batch_vectors)
            except Exception as exc:
                logger.error("REST Batch Embedding failed: %s", exc)
                raise EmbeddingServiceError(f"Batch embedding generation failed: {exc}") from exc

        return all_embeddings
