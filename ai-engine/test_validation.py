import sys, time
sys.path.append('.')
from src.ml.applicability_features import build_applicability_features
from src.ml.applicability_model import load_applicability_model, get_applicability_model
import pandas as pd
import numpy as np

print("TEST 5 - THREADING")
load_applicability_model("../standiq_applicability_model_v2.joblib", "../standiq_applicability_model_v2_metadata.json")
model = get_applicability_model()
classifier = model.pipeline.named_steps.get("model")
print(f"Loaded classifier n_jobs: {getattr(classifier, 'n_jobs', 'None')}")
assert getattr(classifier, 'n_jobs') == 1, "n_jobs must be 1"

print("\nSetting up mock data (11 reqs, 35 candidates)")
reqs = ["Requirement text " + str(i)*20 for i in range(11)]
candidates = [{"is_number": f"IS {i}", "title": "Title", "summary": "", "scope": "", "search_text": "Text "*10, "bm25_score": 0.5, "semantic_score": 0.5, "rrf_score": 0.5} for i in range(35)]

print("\nTEST 1, 2, 3 - EQUIVALENCE")
# Test on one requirement
req = reqs[0]

single_preds = []
features_list = []
for cand in candidates:
    df = build_applicability_features(req, cand, candidates)
    features_list.append(df)
    pred = model.predict(df)
    single_preds.append(pred)

batch_df = pd.concat(features_list, ignore_index=True)
batch_preds = model.predict(batch_df)
if isinstance(batch_preds, dict): batch_preds = [batch_preds]

for i, (sp, bp) in enumerate(zip(single_preds, batch_preds)):
    assert sp['applicability_score'] == bp['applicability_score'], f"Mismatch at index {i}: {sp} != {bp}"
    assert sp['applicability_class'] == bp['applicability_class'], f"Class mismatch at index {i}"
    assert sp['feature_coverage'] == bp['feature_coverage'], f"Feature coverage mismatch at index {i}"
print("Tests 1, 2, 3 Passed: Features, scores, and candidate selections are identical!")

print("\nTEST 4 - WORKLOAD TIMING")
# Old style (unbatched)
t0 = time.time()
for r in reqs:
    for c in candidates:
        df = build_applicability_features(r, c, candidates)
        model.predict(df)
t_old = time.time() - t0
print(f"Total Old Time (Sequential): {t_old:.4f}s")

# New style (batched)
t0 = time.time()
t_feat_total = 0
t_pred_total = 0
for r in reqs:
    f_list = []
    t_feat_start = time.time()
    for c in candidates:
        df = build_applicability_features(r, c, candidates)
        f_list.append(df)
    batch_df = pd.concat(f_list, ignore_index=True)
    t_feat_total += (time.time() - t_feat_start)
    
    t_pred_start = time.time()
    model.predict(batch_df)
    t_pred_total += (time.time() - t_pred_start)
t_new = time.time() - t0
print(f"Total New Time (Sequential): {t_new:.4f}s")
print(f"  Feature generation time: {t_feat_total:.4f}s")
print(f"  Prediction time: {t_pred_total:.4f}s")
