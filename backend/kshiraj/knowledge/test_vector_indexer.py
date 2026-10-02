"""
kshiraj/knowledge/test_vector_indexer.py

Unit tests for VectorIndexer, realistic semantic search, and Qdrant failure fallback.
All tests run in-memory and isolated without live external server dependencies.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from shared.models import Evidence, EvidenceSourceType, Standard, StandardStatus

from kshiraj.knowledge.embedding_service import EmbeddingService
from kshiraj.knowledge.evidence_store import EvidenceStore
from kshiraj.knowledge.hybrid_retrieval import HybridRetrievalService
from kshiraj.knowledge.retrieval_service import RetrievalResult, RetrievalService
from kshiraj.knowledge.standards_store import StandardsStore
from kshiraj.knowledge.vector_indexer import VectorIndexer
from kshiraj.knowledge.vector_store import VectorStore


@pytest.fixture
def mock_embedding_service():
    service = MagicMock()
    service.dimension = 3072
    service.encode_text.return_value = [0.15] * 3072
    service.encode_batch.side_effect = lambda texts: [[0.1 * (i + 1)] * 3072 for i in range(len(texts))]
    return service


@pytest.fixture
def in_memory_vector_store(mock_embedding_service):
    """VectorStore using mock qdrant client."""
    client = MagicMock()
    collections = {}

    def create_collection(collection_name, vectors_config=None, **kwargs):
        if collection_name not in collections:
            collections[collection_name] = []

    def upsert(collection_name, points, **kwargs):
        if collection_name not in collections:
            collections[collection_name] = []
        collections[collection_name].extend(points)

    def count(collection_name, **kwargs):
        res = MagicMock()
        res.count = len(collections.get(collection_name, []))
        return res

    def query_points(collection_name, query=None, query_filter=None, limit=10, **kwargs):
        points = collections.get(collection_name, [])
        hits = []
        for p in points[:limit]:
            hit = MagicMock()
            if isinstance(p, dict):
                hit.id = p.get("id", "1")
                hit.payload = p.get("payload", {})
            else:
                hit.id = getattr(p, "id", "1")
                hit.payload = getattr(p, "payload", {})
            hit.score = 0.88
            hits.append(hit)
        res = MagicMock()
        res.points = hits
        return res

    client.create_collection.side_effect = create_collection
    client.upsert.side_effect = upsert
    client.count.side_effect = count
    client.query_points.side_effect = query_points
    client.collection_exists.return_value = False

    return VectorStore(client=client)


class TestVectorIndexer:
    """Test suite for VectorIndexer."""

    def test_index_single_standard(self, mock_embedding_service, in_memory_vector_store):
        indexer = VectorIndexer(embedding_service=mock_embedding_service, vector_store=in_memory_vector_store)
        std = Standard(id="s1", is_number="IS 2062", title="Steel Plates", year=2011, status=StandardStatus.ACTIVE)

        count = indexer.index_standards([std])
        assert count == 1
        assert in_memory_vector_store.count_standards() == 1

    def test_index_multiple_standards(self, mock_embedding_service, in_memory_vector_store):
        indexer = VectorIndexer(embedding_service=mock_embedding_service, vector_store=in_memory_vector_store)
        s1 = Standard(id="s1", is_number="IS 2062", title="Steel Plates", status=StandardStatus.ACTIVE)
        s2 = Standard(id="s2", is_number="IS 10322", title="Luminaires", status=StandardStatus.ACTIVE)

        count = indexer.index_standards([s1, s2])
        assert count == 2
        assert in_memory_vector_store.count_standards() == 2

    def test_index_empty_input(self, mock_embedding_service, in_memory_vector_store):
        indexer = VectorIndexer(embedding_service=mock_embedding_service, vector_store=in_memory_vector_store)
        assert indexer.index_standards([]) == 0
        assert indexer.index_evidence([]) == 0

    def test_index_standards_from_store(self, mock_embedding_service, in_memory_vector_store):
        store = StandardsStore()
        store.add(Standard(id="s1", is_number="IS 2062", title="Steel Plates", status=StandardStatus.ACTIVE))
        store.add(Standard(id="s2", is_number="IS 10322", title="Luminaires", status=StandardStatus.ACTIVE))

        indexer = VectorIndexer(embedding_service=mock_embedding_service, vector_store=in_memory_vector_store)
        count = indexer.index_standards_from_store(store)
        assert count == 2

    def test_index_evidence_and_from_store(self, mock_embedding_service, in_memory_vector_store):
        ev_store = EvidenceStore()
        ev = Evidence(id="e1", source_type=EvidenceSourceType.BIS_STANDARD, source_name="BIS Standard", excerpt="Excerpt text")
        ev_store.add(ev)

        indexer = VectorIndexer(embedding_service=mock_embedding_service, vector_store=in_memory_vector_store)
        count = indexer.index_evidence_from_store(ev_store)
        assert count == 1
        assert in_memory_vector_store.count_evidence() == 1


class TestRealisticSemanticRetrievalAndFallback:
    """Test realistic semantic search and Qdrant failure fallback."""

    def test_realistic_semantic_search_structural_steel(self):
        """
        Verify that semantic search identifies IS 2062 for a conceptual query
        'steel requirements for structural fabrication' without explicit IS number.
        """
        std_store = StandardsStore()
        ev_store = EvidenceStore()

        steel_std = Standard(
            id="steel-2062",
            is_number="IS 2062",
            year=2011,
            title="Hot Rolled Medium and High Tensile Structural Steel",
            scope="Covers requirements for structural steel for construction and general engineering purposes.",
            status=StandardStatus.ACTIVE,
        )
        cable_std = Standard(
            id="cable-694",
            is_number="IS 694",
            year=2010,
            title="PVC Insulated Cables",
            scope="Covers flexible cables for general electrical wiring.",
            status=StandardStatus.ACTIVE,
        )
        std_store.add(steel_std)
        std_store.add(cable_std)

        mock_embedding = MagicMock()
        mock_embedding.encode_text.return_value = [0.2] * 3072

        mock_vec_store = MagicMock()
        mock_vec_store.search_standards.return_value = [
            {"id": "steel-2062", "score": 0.92, "payload": {"is_number": "IS 2062"}, "type": "standard"}
        ]

        hybrid = HybridRetrievalService(
            standards_store=std_store,
            evidence_store=ev_store,
            embedding_service=mock_embedding,
            vector_store=mock_vec_store,
            lexical_weight=0.3,
            vector_weight=0.7,
        )

        res = hybrid.search("steel requirements for structural fabrication")

        assert isinstance(res, RetrievalResult)
        assert res.total_candidates >= 1
        top_match = res.candidates[0]
        assert top_match.standard.is_number == "IS 2062"

    def test_qdrant_failure_graceful_fallback(self):
        """
        Verify that when vector search fails or Qdrant raises an exception,
        HybridRetrievalService falls back cleanly to lexical search without crashing.
        """
        std_store = StandardsStore()
        ev_store = EvidenceStore()

        std = Standard(id="s1", is_number="IS 10322", title="Luminaires", status=StandardStatus.ACTIVE)
        std_store.add(std)

        mock_embedding = MagicMock()
        mock_embedding.encode_text.return_value = [0.1] * 3072

        failing_vec_store = MagicMock()
        failing_vec_store.search_standards.side_effect = Exception("Qdrant connection refused: 6333 unreachable")

        hybrid = HybridRetrievalService(
            standards_store=std_store,
            evidence_store=ev_store,
            embedding_service=mock_embedding,
            vector_store=failing_vec_store,
        )

        # Query should not crash; lexical search should still succeed
        res = hybrid.search("IS 10322 Luminaires")

        assert isinstance(res, RetrievalResult)
        assert res.total_candidates == 1
        assert res.candidates[0].standard.is_number == "IS 10322"
