"""
kshiraj/knowledge/test_embedding_service.py

Unit tests for kshiraj.knowledge.embedding_service.EmbeddingService.
All tests use mock models and do NOT require live Gemini API or internet access.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from kshiraj.knowledge.embedding_service import EmbeddingService, EmbeddingServiceError


@pytest.fixture
def mock_gemini_client():
    """Mock google.genai.Client object returning deterministic vectors."""
    client = MagicMock()

    def mock_embed_content(model, contents):
        response = MagicMock()
        response.embeddings = []
        
        # If single string
        if isinstance(contents, str):
            contents = [contents]
            
        for i, text in enumerate(contents):
            emb = MagicMock()
            if text.strip():
                # Mock a 3072 dimension vector, e.g. [0.1, 0.1, ...]
                emb.values = [0.1 * (i + 1)] * 3072
            else:
                emb.values = [0.0] * 3072
            response.embeddings.append(emb)
            
        return response

    client.models.embed_content.side_effect = mock_embed_content
    return client


class TestEmbeddingService:
    """Test suite for EmbeddingService."""

    def test_single_embedding(self, mock_gemini_client):
        service = EmbeddingService(model=mock_gemini_client)
        vec = service.encode_text("Luminaires standard IS 10322")

        assert isinstance(vec, list)
        assert len(vec) == 3072
        # L2 norm check
        norm = sum(x * x for x in vec) ** 0.5
        assert pytest.approx(norm, 1e-4) == 1.0

    def test_batch_embedding(self, mock_gemini_client):
        service = EmbeddingService(model=mock_gemini_client)
        texts = ["Text 1", "Text 2", "Text 3"]
        vecs = service.encode_batch(texts)

        assert isinstance(vecs, list)
        assert len(vecs) == 3
        for v in vecs:
            assert len(v) == 3072
            norm = sum(x * x for x in v) ** 0.5
            assert pytest.approx(norm, 1e-4) == 1.0

    def test_empty_text_handling(self, mock_gemini_client):
        service = EmbeddingService(model=mock_gemini_client)
        empty_vec = service.encode_text("")
        ws_vec = service.encode_text("   ")

        assert empty_vec == [0.0] * 3072
        assert ws_vec == [0.0] * 3072

    def test_empty_batch_handling(self, mock_gemini_client):
        service = EmbeddingService(model=mock_gemini_client)
        vecs = service.encode_batch([])
        assert vecs == []

    def test_batch_with_empty_strings(self, mock_gemini_client):
        service = EmbeddingService(model=mock_gemini_client)
        vecs = service.encode_batch(["Valid text", "", "   "])
        assert len(vecs) == 3
        assert vecs[1] == [0.0] * 3072
        assert vecs[2] == [0.0] * 3072

    def test_dimension_property(self, mock_gemini_client):
        service = EmbeddingService(model=mock_gemini_client)
        assert service.dimension == 3072

    def test_model_reuse(self, mock_gemini_client):
        service = EmbeddingService(model=mock_gemini_client)
        m1 = service._get_model()
        m2 = service._get_model()
        assert m1 is m2
