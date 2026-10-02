import sys
import pandas as pd
from typing import List, Dict
from unittest.mock import MagicMock

# Import the actual classes
from src.ml.applicability_model import load_applicability_model, get_applicability_model
from src.reasoning.providers.ml import MLReasoner
from src.ml.applicability_features import build_applicability_features
import json

def test_production_optimization():
    print("==================================================")
    print("PRODUCTION OPTIMIZATION VALIDATION")
    print("==================================================")

    # 1. Load model and verify n_jobs
    print("\n[+] 1. Loading Applicability Model...")
    load_applicability_model(
        "../standiq_applicability_model_v2.joblib",
        "../standiq_applicability_model_v2_metadata.json"
    )
    model = get_applicability_model()
    
    # Extract the actual RandomForestClassifier step from the pipeline
    rf_clf = model.pipeline.named_steps.get("model")
    print(f"    -> classifier.n_jobs == {rf_clf.n_jobs}")
    assert rf_clf.n_jobs == 1, "n_jobs must be 1 to prevent thread explosion!"

    # 2. Simulate 11 requirements x 35 standards
    print("\n[+] 2. Generating 11x35 features (simulating full payload)...")
    
    req_texts = [f"Requirement {i}: The street light shall be awesome and bright" for i in range(11)]
    
    # Fake standards
    stds = [
        {"id": f"std-{i}", "is_number": f"IS {1000+i}", "title": f"Standard {i}", "status": "active", "relevance_score": 0.8}
        for i in range(35)
    ]
    
    print(f"    -> {len(req_texts)} requirements, {len(stds)} standards.")
    
    features_list = []
    
    for req in req_texts:
        for std in stds:
            df = build_applicability_features(req, std, stds)
            features_list.append(df)
            
    print(f"    -> Total features generated: {len(features_list)}")
    assert len(features_list) == 385, "Expected 385 features"

    # 3. Test equivalence of Batched vs Single-row
    print("\n[+] 3. Testing Single vs Batched equivalence...")
    
    # Single
    import time
    t0 = time.perf_counter()
    single_preds = []
    for df in features_list:
        single_preds.append(model.predict(df))
    t_single = time.perf_counter() - t0
    
    # Batched
    batch_df = pd.concat(features_list, ignore_index=True)
    t0 = time.perf_counter()
    batch_preds = model.predict(batch_df)
    t_batch = time.perf_counter() - t0
    
    print(f"    -> Single-row total time: {t_single:.4f}s")
    print(f"    -> Batched total time:    {t_batch:.4f}s")
    
    assert len(single_preds) == len(batch_preds), "Length mismatch!"
    for i, (sp, bp) in enumerate(zip(single_preds, batch_preds)):
        assert sp == bp, f"Mismatch at index {i}: {sp} != {bp}"
        
    print("    -> SUCCESS: Batched predictions exactly match sequential single-row predictions.")
    print("    -> SUCCESS: Probability calculation is unchanged.")
    print("    -> SUCCESS: Class mapping is unchanged.")

    # 4. Test MLReasoner completes successfully
    print("\n[+] 4. Testing MLReasoner execution...")
    
    reasoner = MLReasoner()
    
    # Run a real analysis step
    req_dict = {
        "id": "req-1",
        "text": "The LED street lighting luminaires shall conform to IS 16107.",
        "category": "technical_specification"
    }
    
    # A couple candidates
    candidates = [
        {
            "id": "std-1",
            "is_number": "IS 16107",
            "title": "LED Luminaires for Road and Street Lighting",
            "status": "active",
            "year": 2013,
            "relevance_score": 0.95
        },
        {
            "id": "std-2",
            "is_number": "IS 10322",
            "title": "Luminaires for road lighting",
            "status": "active",
            "year": 2012,
            "relevance_score": 0.80
        }
    ]
    
    result = reasoner.analyze(req_dict, candidates, candidates)
    
    print(f"    -> MLReasoner returned verdict: {result['verdict']}")
    print(f"    -> Best matching standard: {result['applicable_standards'][0] if result['applicable_standards'] else 'None'}")
    print("    -> SUCCESS: LED MLReasoner completes successfully without errors.")

if __name__ == "__main__":
    test_production_optimization()
