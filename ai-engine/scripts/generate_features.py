import sys
import os
import json
import asyncio
import numpy as np
import pandas as pd
from pathlib import Path

backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backend'))
ai_engine_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(backend_path)
sys.path.append(ai_engine_path)

from kartikey.orchestration.knowledge_registry import initialize_knowledge_registry
from kshiraj.knowledge.retrieval_service import RetrievalQuery
from src.ml.applicability_features import build_applicability_features

async def main():
    print("Loading services...")
    registry = initialize_knowledge_registry()
    standards_store = registry.standards_store
    hybrid_service = registry.retrieval_service
    lexical_service = hybrid_service.lexical_service
    embedding_service = hybrid_service.embedding_service
    vector_store = hybrid_service.vector_store
    
    # Wait, we need to populate data inside registry
    # initialize_knowledge_registry loads data itself?
    # No, we need to call load on standards_store manually if initialize_knowledge_registry doesn't load catalog.
    
    input_file = os.path.join(ai_engine_path, "data", "training", "labeled_pairs.jsonl")
    output_file = os.path.join(ai_engine_path, "data", "training", "labeled_pairs_featured.jsonl")
    
    rows = []
    with open(input_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            rows.append(json.loads(line))
            
    print(f"Loaded {len(rows)} labeled pairs from {input_file}.")
    
    featured_rows = []
    
    for row in rows:
        req_text = row["requirement_text"]
        target_title = row["standard_title"]
        
        target_std_obj = None
        for std in standards_store.list_all():
            if std.title == target_title:
                target_std_obj = std
                break
                
        if not target_std_obj:
            print(f"Warning: Standard with title '{target_title}' not found in catalog!")
            continue
            
        target_id = target_std_obj.id
        
        rq = RetrievalQuery(query_text=req_text, top_k=2000)
        lex_result = lexical_service.search_standards(rq)
        lex_candidates_by_id = {c.standard.id: c for c in lex_result.candidates}
        max_lex_score = max((c.score for c in lex_result.candidates), default=1.0)
        if max_lex_score <= 0:
            max_lex_score = 1.0
            
        query_vec = embedding_service.encode_text(req_text)
        vector_hits = vector_store.search_standards(query_vector=query_vec, top_k=2000)
        vector_scores_by_id = {}
        for hit in vector_hits:
            std_id = hit.get("id")
            score = hit.get("score", 0.0)
            if std_id:
                vector_scores_by_id[std_id] = float(score)
                
        lex_candidate = lex_candidates_by_id.get(target_id)
        raw_lex = lex_candidate.score if lex_candidate else 0.0
        norm_lex = min(1.0, max(0.0, raw_lex / max_lex_score))
        
        raw_vec = vector_scores_by_id.get(target_id, 0.0)
        norm_vec = min(1.0, max(0.0, (raw_vec + 1.0) / 2.0)) if raw_vec != 0.0 else 0.0
        
        final_score = (hybrid_service.lexical_weight * norm_lex) + (hybrid_service.vector_weight * norm_vec)
        
        candidate_dict = target_std_obj.model_dump()
        candidate_dict["semantic_score"] = norm_vec
        candidate_dict["rrf_score"] = final_score
        candidate_dict["bm25_score"] = float('nan')
        candidate_dict["standardCode"] = getattr(target_std_obj, "is_number", "")
        
        try:
            hybrid_result = hybrid_service.search_standards(rq)
            all_candidate_dicts = []
            for c in hybrid_result.candidates:
                c_dict = c.standard.model_dump()
                c_dict["semantic_score"] = getattr(c.standard, "semantic_score", float('nan'))
                c_dict["rrf_score"] = getattr(c.standard, "relevance_score", float('nan'))
                c_dict["bm25_score"] = float('nan')
                all_candidate_dicts.append(c_dict)
                
            if not any(c["id"] == candidate_dict["id"] for c in all_candidate_dicts):
                all_candidate_dicts.append(candidate_dict)
                
            features_df = build_applicability_features(req_text, candidate_dict, all_candidate_dicts)
            features = features_df.iloc[0].to_dict()
            
            for k, v in features.items():
                if pd.isna(v):
                    features[k] = None
                    
            featured_row = {**row, **features}
            featured_rows.append(featured_row)
        except Exception as e:
            print(f"Error computing features for {target_title}: {e}")
            raise e
            
    with open(output_file, 'w') as f:
        for fr in featured_rows:
            f.write(json.dumps(fr) + "\n")
            
    print("Done! First 5 rows:")
    for fr in featured_rows[:5]:
        print(json.dumps(fr, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
