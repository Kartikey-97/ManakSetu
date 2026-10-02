"""
ai-engine/build_qdrant_index.py

One-time script to upload the BIS standards knowledge base to Qdrant Cloud.
Run ONCE from the ai-engine directory:
  python build_qdrant_index.py

Requires env vars: QDRANT_URL, QDRANT_API_KEY, GOOGLE_API_KEY
Reads: ai-engine/cleaned_bis_standards.json (1028 records, 3.6 MB)
Creates: Qdrant collection "bis_standards_v1" with 768-dim Cosine vectors
Estimated time: 10-15 minutes (rate-limited Gemini API calls)
"""
from __future__ import annotations

import json
import os
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

COLLECTION_NAME = "bis_standards_v1"
STANDARDS_FILE = os.path.join(os.path.dirname(__file__), "cleaned_bis_standards.json")
BATCH_SIZE = 10  # 10 embeddings per batch, then 0.5s pause


def main() -> None:
    # Validate env vars
    for var in ("QDRANT_URL", "QDRANT_API_KEY", "GOOGLE_API_KEY"):
        if not os.environ.get(var, "").strip():
            print(f"ERROR: {var} environment variable is not set.")
            sys.exit(1)

    # Load standards
    if not os.path.exists(STANDARDS_FILE):
        print(f"ERROR: Standards file not found: {STANDARDS_FILE}")
        sys.exit(1)

    with open(STANDARDS_FILE, "r", encoding="utf-8") as f:
        standards = json.load(f)
    logger.info("Loaded %d standards from %s", len(standards), STANDARDS_FILE)

    # Connect to Qdrant
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from src.embedding import get_embeddings_batch, EMBEDDING_DIM

    client = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ.get("QDRANT_API_KEY"),
        timeout=30,
    )

    # Create or verify collection
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        logger.info("Created collection: %s (%d dims, Cosine)", COLLECTION_NAME, EMBEDDING_DIM)
    else:
        logger.info("Collection %s already exists — will upsert", COLLECTION_NAME)

    # Upload in batches
    total = len(standards)
    for batch_start in range(0, total, BATCH_SIZE):
        batch = standards[batch_start : batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (total - 1) // BATCH_SIZE + 1

        # Build embedding text
        texts = []
        for s in batch:
            is_num = s.get("is_number", "")
            title = s.get("title", "")
            scope_raw = s.get("scope", "") or ""
            scope = scope_raw.get("value", "") if isinstance(scope_raw, dict) else str(scope_raw)
            texts.append(f"{is_num} {title} {scope}".strip())

        # Generate embeddings
        try:
            embeddings = get_embeddings_batch(texts, task_type="RETRIEVAL_DOCUMENT")
        except Exception as exc:
            logger.error("Batch %d/%d embedding failed: %s — skipping", batch_num, total_batches, exc)
            continue

        # Build Qdrant points
        points = []
        for i, std in enumerate(batch):
            point_id = batch_start + i + 1  # 1-indexed integers
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embeddings[i],
                    payload=std,
                )
            )

        # Upsert
        try:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            logger.info("Batch %d/%d (docs %d-%d) upserted OK", batch_num, total_batches, batch_start + 1, batch_start + len(batch))
        except Exception as exc:
            logger.error("Batch %d/%d upsert failed: %s", batch_num, total_batches, exc)
            time.sleep(2.0)
            
    logger.info("Index build complete!")

if __name__ == "__main__":
    main()
