"""
kshiraj/knowledge/vector_store.py

Vector store layer backed by Qdrant vector database.
Supports local in-memory mode (QdrantClient(":memory:")) for isolated development/testing,
local persistent Qdrant server, and remote Qdrant Cloud deployments.

Maintains vector collections for Standard models and Evidence models using Cosine similarity.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union
import uuid

from shared.models import Evidence, Standard, StandardStatus
from shared.utils import AppError, get_logger

logger = get_logger(__name__)


class VectorStoreError(AppError):
    """Raised when vector storage, collection creation, or search operations fail."""

    def __init__(self, message: str, code: str = "VECTOR_STORE_ERROR") -> None:
        super().__init__(message, code=code)


def _to_uuid(id_str: str) -> str:
    """Convert an arbitrary string ID into a deterministic UUID string for Qdrant compatibility."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, id_str))


class VectorStore:
    """
    Qdrant vector store abstraction for indexing and searching Standard and Evidence vectors.
    """

    STANDARDS_COLLECTION = os.getenv("QDRANT_STANDARDS_COLLECTION", "bis_standards_v1")
    EVIDENCE_COLLECTION = os.getenv("QDRANT_EVIDENCE_COLLECTION", "evidence_collection")

    def __init__(
        self,
        dimension: int = 3072,
        client: Optional[Any] = None,
        location: Optional[str] = ":memory:",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        standards_collection: str = STANDARDS_COLLECTION,
        evidence_collection: str = EVIDENCE_COLLECTION,
    ) -> None:
        """
        Initialize Qdrant VectorStore.

        Parameters
        ----------
        dimension:
            Vector embedding dimension size (default 384 for bge-small-en-v1.5).
        client:
            Optional pre-initialized QdrantClient or mock object.
        location:
            Qdrant storage location (default ":memory:" for isolated in-memory DB).
        url:
            Optional Qdrant Cloud or server URL (e.g. "http://localhost:6333").
            If None, falls back to env var QDRANT_URL.
        api_key:
            Optional Qdrant API key. If None, falls back to env var QDRANT_API_KEY.
        standards_collection:
            Collection name for Standard vectors.
        evidence_collection:
            Collection name for Evidence vectors.
        """
        self.dimension = dimension
        self.standards_collection = standards_collection
        self.evidence_collection = evidence_collection
        self._client = client
        self._location = location
        self._url = url or os.getenv("QDRANT_URL")
        self._api_key = api_key or os.getenv("QDRANT_API_KEY")

    def _get_client(self) -> Any:
        """Lazy load or return the Qdrant client."""
        if self._client is not None:
            return self._client

        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            logger.error("qdrant-client package is not installed.")
            raise VectorStoreError(
                "The 'qdrant-client' package is required for vector storage. "
                "Install it using `pip install qdrant-client`.",
                code="DEPENDENCY_MISSING",
            ) from exc

        try:
            # If Qdrant URL is set but fails, we shouldn't wipe data.
            # We'll default to a persistent local path instead of :memory:
            if self._url:
                try:
                    logger.info("Connecting to Qdrant server at %s...", self._url)
                    self._client = QdrantClient(url=self._url, api_key=self._api_key, timeout=5.0)
                    self._client.get_collections()
                except Exception as e:
                    logger.warning("Cloud Qdrant unreachable (%s), falling back to local persistent storage.", e)
                    self._client = QdrantClient(path="./data/qdrant_local")
            else:
                fallback_path = "./data/qdrant_local" if self._location == ":memory:" else self._location
                logger.info("Initializing persistent local Qdrant client (%s)...", fallback_path)
                self._client = QdrantClient(path=fallback_path)
            return self._client
        except Exception as exc:
            logger.error("Failed to initialize QdrantClient: %s", exc)
            raise VectorStoreError(f"Failed to connect to Qdrant: {exc}") from exc

    def create_collections_if_needed(self) -> None:
        """Create standards and evidence vector collections if they do not exist."""
        client = self._get_client()

        try:
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            Distance = None
            VectorParams = None

        for coll_name in [self.standards_collection, self.evidence_collection]:
            try:
                exists = False
                if hasattr(client, "collection_exists"):
                    exists = client.collection_exists(collection_name=coll_name)
                elif hasattr(client, "get_collections"):
                    colls = client.get_collections().collections
                    exists = any(c.name == coll_name for c in colls)

                if not exists:
                    logger.info("Creating Qdrant collection '%s' (dim=%s, metric=COSINE)...", coll_name, self.dimension)
                    if hasattr(client, "create_collection"):
                        params = VectorParams(size=self.dimension, distance=Distance.COSINE) if VectorParams else {"size": self.dimension, "distance": "Cosine"}
                        client.create_collection(
                            collection_name=coll_name,
                            vectors_config=params,
                        )
            except Exception as exc:
                logger.warning("Could not verify/create collection '%s': %s", coll_name, exc)

    def upsert_standards(self, standards: List[Standard], embeddings: List[List[float]]) -> None:
        """
        Upsert Standard models with corresponding embedding vectors into Qdrant.

        Parameters
        ----------
        standards:
            List of Standard objects.
        embeddings:
            List of normalized float vector lists.
        """
        if not standards or not embeddings or len(standards) != len(embeddings):
            return

        self.create_collections_if_needed()
        client = self._get_client()

        try:
            from qdrant_client.models import PointStruct
        except ImportError:
            PointStruct = None

        points = []
        for std, vec in zip(standards, embeddings):
            payload = {
                "id": std.id,
                "designation": std.designation,
                "is_number": std.is_number,
                "year": std.year,
                "status": std.status.value,
                "title": std.title,
                "scope": std.scope,
                "division_council": std.division_council,
                "technical_committee": std.technical_committee,
                "source_url": std.source_url,
            }
            point_id = _to_uuid(std.id)
            if PointStruct:
                points.append(PointStruct(id=point_id, vector=vec, payload=payload))
            else:
                points.append({"id": point_id, "vector": vec, "payload": payload})

        try:
            client.upsert(collection_name=self.standards_collection, points=points)
            logger.info("Upserted %s standards into Qdrant '%s'.", len(points), self.standards_collection)
        except Exception as exc:
            logger.error("Error upserting standards to Qdrant: %s", exc)
            raise VectorStoreError(f"Failed to upsert standards to vector store: {exc}") from exc

    def upsert_evidence(self, evidence_items: List[Evidence], embeddings: List[List[float]]) -> None:
        """
        Upsert Evidence models with corresponding embedding vectors into Qdrant.

        Parameters
        ----------
        evidence_items:
            List of Evidence objects.
        embeddings:
            List of normalized float vector lists.
        """
        if not evidence_items or not embeddings or len(evidence_items) != len(embeddings):
            return

        self.create_collections_if_needed()
        client = self._get_client()

        try:
            from qdrant_client.models import PointStruct
        except ImportError:
            PointStruct = None

        points = []
        for ev, vec in zip(evidence_items, embeddings):
            payload = {
                "id": ev.id,
                "source_type": ev.source_type.value,
                "source_name": ev.source_name,
                "authority": ev.authority,
                "url": ev.url,
                "section": ev.section,
                "excerpt": ev.excerpt,
                "retrieval_date": ev.retrieval_date.isoformat() if ev.retrieval_date else None,
            }
            point_id = _to_uuid(ev.id)
            if PointStruct:
                points.append(PointStruct(id=point_id, vector=vec, payload=payload))
            else:
                points.append({"id": point_id, "vector": vec, "payload": payload})

        try:
            client.upsert(collection_name=self.evidence_collection, points=points)
            logger.info("Upserted %s evidence records into Qdrant '%s'.", len(points), self.evidence_collection)
        except Exception as exc:
            logger.error("Error upserting evidence to Qdrant: %s", exc)
            raise VectorStoreError(f"Failed to upsert evidence to vector store: {exc}") from exc

    def search_standards(
        self,
        query_vector: List[float],
        top_k: int = 10,
        status_filter: Optional[StandardStatus] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform cosine similarity vector search for standards.

        Parameters
        ----------
        query_vector:
            Normalized query embedding vector.
        top_k:
            Maximum number of candidates to return.
        status_filter:
            Optional StandardStatus enum filter.

        Returns
        -------
        List[Dict[str, Any]]
            List of candidate dictionaries: [{'id': str, 'score': float, 'payload': dict}]
        """
        if not query_vector:
            return []

        client = self._get_client()
        self.create_collections_if_needed()

        query_filter = None
        if status_filter is not None:
            try:
                from qdrant_client.models import FieldCondition, Filter, MatchValue
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="status",
                            match=MatchValue(value=status_filter.value),
                        )
                    ]
                )
            except ImportError:
                query_filter = None

        try:
            if hasattr(client, "query_points"):
                results = client.query_points(
                    collection_name=self.standards_collection,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=top_k,
                ).points
            elif hasattr(client, "search"):
                results = client.search(
                    collection_name=self.standards_collection,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=top_k,
                )
            else:
                return []

            output = []
            for hit in results:
                payload = getattr(hit, "payload", {}) or {}
                score = float(getattr(hit, "score", 0.0))
                std_id = payload.get("id") or str(getattr(hit, "id", ""))
                output.append({"id": std_id, "score": score, "payload": payload, "type": "standard"})
            return output
        except Exception as exc:
            logger.error("Error searching standards vector store: %s", exc)
            return []

    def search_evidence(self, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform cosine similarity vector search for evidence.

        Returns
        -------
        List[Dict[str, Any]]
            List of candidate dictionaries: [{'id': str, 'score': float, 'payload': dict}]
        """
        if not query_vector:
            return []

        client = self._get_client()
        self.create_collections_if_needed()

        try:
            if hasattr(client, "query_points"):
                results = client.query_points(
                    collection_name=self.evidence_collection,
                    query=query_vector,
                    limit=top_k,
                ).points
            elif hasattr(client, "search"):
                results = client.search(
                    collection_name=self.evidence_collection,
                    query_vector=query_vector,
                    limit=top_k,
                )
            else:
                return []

            output = []
            for hit in results:
                payload = getattr(hit, "payload", {}) or {}
                score = float(getattr(hit, "score", 0.0))
                ev_id = payload.get("id") or str(getattr(hit, "id", ""))
                output.append({"id": ev_id, "score": score, "payload": payload, "type": "evidence"})
            return output
        except Exception as exc:
            logger.error("Error searching evidence vector store: %s", exc)
            return []

    def count_standards(self) -> int:
        """Return total number of standard vectors in collection."""
        client = self._get_client()
        try:
            res = client.count(collection_name=self.standards_collection)
            return int(getattr(res, "count", 0))
        except Exception:
            return 0

    def count_evidence(self) -> int:
        """Return total number of evidence vectors in collection."""
        client = self._get_client()
        try:
            res = client.count(collection_name=self.evidence_collection)
            return int(getattr(res, "count", 0))
        except Exception:
            return 0

    def delete_standard(self, standard_id: str) -> None:
        """Delete a standard vector by its standard ID."""
        client = self._get_client()
        point_id = _to_uuid(standard_id)
        try:
            from qdrant_client.models import PointIdsList
            client.delete(collection_name=self.standards_collection, points_selector=PointIdsList(points=[point_id]))
        except Exception as exc:
            logger.warning("Failed to delete standard %s: %s", standard_id, exc)

    def delete_evidence(self, evidence_id: str) -> None:
        """Delete an evidence vector by its evidence ID."""
        client = self._get_client()
        point_id = _to_uuid(evidence_id)
        try:
            from qdrant_client.models import PointIdsList
            client.delete(collection_name=self.evidence_collection, points_selector=PointIdsList(points=[point_id]))
        except Exception as exc:
            logger.warning("Failed to delete evidence %s: %s", evidence_id, exc)

    def clear(self) -> None:
        """Clear all collections in vector store."""
        client = self._get_client()
        for coll in [self.standards_collection, self.evidence_collection]:
            try:
                client.delete_collection(collection_name=coll)
            except Exception:
                pass
        self.create_collections_if_needed()
