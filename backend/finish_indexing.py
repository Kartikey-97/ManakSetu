import os
import json
import time

# Set env vars
os.environ["SEMANTIC_RETRIEVAL_ENABLED"] = "true"
os.environ["QDRANT_URL"] = "https://79b455c4-f081-4460-8344-317691614c5c.eu-west-2-0.aws.cloud.qdrant.io"
os.environ["QDRANT_API_KEY"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6YTBjYWYyZjktNzQxNC00MzMzLThhNTktMTFiYzJiYzgzZTZlIn0.tnoZDtkavTaQRUd9JbBWtAp8s9-LMQ5dYAxOEumFjXc"
os.environ["QDRANT_STANDARDS_COLLECTION"] = "bis_standards_v1"

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import hashlib

from kshiraj.knowledge.embedding_service import EmbeddingService

def main():
    print("Connecting to Qdrant...")
    # 60s timeout for Qdrant to avoid httpx write timeouts
    q_client = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
        timeout=60.0
    )
    collection = os.environ["QDRANT_STANDARDS_COLLECTION"]

    print("Fetching existing points...")
    existing_is_numbers = set()
    offset = None
    while True:
        points, offset = q_client.scroll(
            collection_name=collection,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        for p in points:
            if p.payload and "is_number" in p.payload:
                existing_is_numbers.add(p.payload["is_number"])
        if offset is None:
            break

    print(f"Found {len(existing_is_numbers)} existing standards in Qdrant.")

    print("Loading local catalog...")
    cat_path = "/Users/kartikeygupta/Desktop/Hackathons/sih_26108/backend/shared/bis_catalogue_reconciled.json"
    with open(cat_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    missing = [s for s in catalog if s.get("is_number") not in existing_is_numbers]
    print(f"Total standards: {len(catalog)}. Missing: {len(missing)}")

    if not missing:
        print("Everything is indexed!")
        return

    emb_service = EmbeddingService(batch_size=10) # Smaller batch to avoid timeouts

    batch_size = 10
    for i in range(0, len(missing), batch_size):
        batch = missing[i:i+batch_size]
        time.sleep(1)
        print(f"Processing {i} to {i+len(batch)} of {len(missing)}...")

        texts = []
        for s in batch:
            is_num = s.get("is_number", "")
            title = s.get("title", "")
            scope_raw = s.get("scope", "") or ""
            scope = scope_raw.get("value", "") if isinstance(scope_raw, dict) else str(scope_raw)
            texts.append(f"{is_num} {title} {scope}".strip())

        try:
            embeddings = emb_service.encode_batch(texts)
        except Exception as e:
            print(f"Embedding failed: {e}")
            continue

        points = []
        for j, std in enumerate(batch):
            point_id_str = f"{std.get('is_number', '')}:{std.get('year', '')}"
            point_id = int(hashlib.md5(point_id_str.encode()).hexdigest()[:15], 16)
            
            payload = std
            payload["normalized_is_number"] = std.get("is_number", "").replace(" ", "")

            points.append(PointStruct(
                id=point_id,
                vector=embeddings[j],
                payload=payload
            ))

        try:
            q_client.upsert(
                collection_name=collection,
                points=points
            )
            print(f"  Upserted {len(points)} points.")
        except Exception as e:
            print(f"  Qdrant upsert failed: {e}")
            time.sleep(2)

    print("Finished uploading!")

if __name__ == "__main__":
    main()
