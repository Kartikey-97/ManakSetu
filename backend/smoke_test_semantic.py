import sys
import os
import asyncio

# Setup environment variables BEFORE importing config
os.environ["SEMANTIC_RETRIEVAL_ENABLED"] = "true"
os.environ["QDRANT_URL"] = "https://79b455c4-f081-4460-8344-317691614c5c.eu-west-2-0.aws.cloud.qdrant.io"
os.environ["QDRANT_API_KEY"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6YTBjYWYyZjktNzQxNC00MzMzLThhNTktMTFiYzJiYzgzZTZlIn0.tnoZDtkavTaQRUd9JbBWtAp8s9-LMQ5dYAxOEumFjXc"

from kartikey.orchestration.knowledge_registry import initialize_knowledge_registry
from kshiraj.knowledge.retrieval_service import RetrievalQuery

async def run_smoke_test():
    print("--- SMOKE TEST: SEMANTIC RETRIEVAL ---")
    print("Initializing knowledge registry (this will connect to Qdrant)...")
    
    registry = initialize_knowledge_registry()
    print(f"Retrieval Mode: {registry.retrieval_mode}")
    print(f"Retrieval Reason: {registry.retrieval_reason}")
    
    if registry.retrieval_mode != "hybrid":
        print("ERROR: Semantic retrieval did not initialize successfully.")
        sys.exit(1)
        
    print("\nExecuting vector search for 'LED Luminaires'...")
    query = RetrievalQuery(query_text="LED Luminaires", top_k=5, include_evidence=False)
    
    # Run the synchronous search_standards in a thread to prevent blocking the asyncio loop!
    # (The google-genai sync client deadlocks if called directly on the main async loop).
    results = await asyncio.to_thread(registry.retrieval_service.search_standards, query)
    
    print(f"\nFound {results.total_candidates} candidates.")
    for c in results.candidates:
        print(f" - {c.standard.is_number}: {c.standard.title} (Score: {c.score:.3f})")
        
    if results.total_candidates > 0:
        print("\nSUCCESS: The pipeline end-to-end integration works!")
    else:
        print("\nWARNING: No results found.")

if __name__ == "__main__":
    asyncio.run(run_smoke_test())
