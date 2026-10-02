"""
Phase 0 Retrieval Audit — StandIQ
===================================
Runs each retrieval stage independently and reports per-query results so we can
identify EXACTLY where recall drops to zero (or near-zero).

Outputs:
  - Per-query: what BM25, FAISS, Hybrid, and Final-ranked each returned
  - Summary metrics: Recall@1/3/5 and MRR for each stage
  - Diagnosis: is the issue in the retrieval engine or the evaluation fixture?
"""

import json
import logging
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Suppress noisy logs
logging.basicConfig(level=logging.WARNING)

from src.embedding import generate_embeddings
from src.search import HybridRetriever, BM25Retriever, VectorStore
from src.ranking import rank_results

# ──────────────────────────────────────────────────────────────
# Load evaluation queries
# ──────────────────────────────────────────────────────────────
EVAL_PATH = os.path.join(os.path.dirname(__file__), "data", "evaluation_queries.json")
KB_PATH    = os.path.join(os.path.dirname(__file__), "data", "bis_full_knowledge_base.json")

with open(EVAL_PATH, "r") as f:
    eval_queries = json.load(f)

with open(KB_PATH, "r") as f:
    standards = json.load(f)

print(f"Loaded {len(eval_queries)} queries, {len(standards)} standards\n")

# ──────────────────────────────────────────────────────────────
# Build indexes (reuse embedding cache if present)
# ──────────────────────────────────────────────────────────────
import numpy as np

cache_path = KB_PATH.replace(".json", "_embeddings.npy")
kb_mtime   = os.path.getmtime(KB_PATH)

if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= kb_mtime:
    print("Loading embeddings from cache…")
    embeddings = np.load(cache_path)
    if embeddings.shape[0] != len(standards):
        print("Cache size mismatch, regenerating…")
        embeddings = None
else:
    embeddings = None

if embeddings is None:
    print("Generating embeddings (this takes ~30s)…")
    search_texts = [std.get("search_text", "") for std in standards]
    embeddings   = generate_embeddings(search_texts)
    np.save(cache_path, embeddings)
    print("Saved embedding cache.")

print("Building BM25 + FAISS indexes…")
bm25_retriever = BM25Retriever().fit([std.get("search_text", "") for std in standards])
vector_store   = VectorStore(dimension=embeddings.shape[1])
vector_store.add_embeddings(embeddings)
hybrid         = HybridRetriever().fit(standards, embeddings)
print("Indexes ready.\n")

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def normalize_is(s: str) -> str:
    """Normalise IS references for comparison: lowercase, collapse spaces/colons."""
    return re.sub(r"[\s:]+", " ", s.lower()).strip()

def hit_at_k(returned: list[str], expected: str, k: int) -> bool:
    norm_exp = normalize_is(expected)
    for r in returned[:k]:
        if normalize_is(r) == norm_exp:
            return True
        # partial match: expected 'IS 10322 Part 5 Sec 3' matches 'IS 10322 : Part 5 : Sec 3'
        if norm_exp in normalize_is(r) or normalize_is(r) in norm_exp:
            return True
    return False

def reciprocal_rank(returned: list[str], expected: str) -> float:
    norm_exp = normalize_is(expected)
    for i, r in enumerate(returned, 1):
        if normalize_is(r) == norm_exp or norm_exp in normalize_is(r) or normalize_is(r) in norm_exp:
            return 1.0 / i
    return 0.0

TOP_K = 20

# ──────────────────────────────────────────────────────────────
# Per-query evaluation
# ──────────────────────────────────────────────────────────────

stages = {
    "bm25":   {"hits1": 0, "hits3": 0, "hits5": 0, "rr": []},
    "faiss":  {"hits1": 0, "hits3": 0, "hits5": 0, "rr": []},
    "hybrid": {"hits1": 0, "hits3": 0, "hits5": 0, "rr": []},
    "ranked": {"hits1": 0, "hits3": 0, "hits5": 0, "rr": []},
}

DIVIDER = "─" * 80

for item in eval_queries:
    query    = item["query"]
    expected = item["expected_is_number"]
    category = item.get("category", "")

    # 1. Embeddings
    q_emb = generate_embeddings(query)

    # 2. BM25-only (top 20)
    bm25_scores, bm25_idxs = bm25_retriever.search(query, top_k=TOP_K)
    bm25_numbers = [standards[int(i)]["is_number"] for i in bm25_idxs]

    # 3. FAISS-only (top 20)
    faiss_scores, faiss_idxs = vector_store.search(q_emb, top_k=TOP_K)
    faiss_numbers = [standards[int(i)]["is_number"] for i in faiss_idxs if int(i) != -1]

    # 4. Hybrid RRF (top 20)
    hybrid_results = hybrid.search(query, q_emb, top_k=TOP_K)
    hybrid_numbers = [r["is_number"] for r in hybrid_results]

    # 5. After deterministic rank_results
    # Needs a minimal query_understanding dict
    from src.query_understanding import _regex_fallback
    q_understanding = _regex_fallback(query)
    ranked_results  = rank_results([r.copy() for r in hybrid_results], q_understanding)
    ranked_numbers  = [r["is_number"] for r in ranked_results]

    # Record hits
    for stage_name, nums in [("bm25", bm25_numbers), ("faiss", faiss_numbers),
                              ("hybrid", hybrid_numbers), ("ranked", ranked_numbers)]:
        s = stages[stage_name]
        s["hits1"] += int(hit_at_k(nums, expected, 1))
        s["hits3"] += int(hit_at_k(nums, expected, 3))
        s["hits5"] += int(hit_at_k(nums, expected, 5))
        s["rr"].append(reciprocal_rank(nums, expected))

    # Print per-query breakdown
    bm25_hit   = "✅" if hit_at_k(bm25_numbers,   expected, 5) else "❌"
    faiss_hit  = "✅" if hit_at_k(faiss_numbers,  expected, 5) else "❌"
    hybrid_hit = "✅" if hit_at_k(hybrid_numbers, expected, 5) else "❌"
    ranked_hit = "✅" if hit_at_k(ranked_numbers, expected, 5) else "❌"

    print(DIVIDER)
    print(f"QUERY    : {query}")
    print(f"EXPECTED : {expected}  [{category}]")
    print(f"BM25  {bm25_hit}: {bm25_numbers[:5]}")
    print(f"FAISS {faiss_hit}: {faiss_numbers[:5]}")
    print(f"HYB   {hybrid_hit}: {hybrid_numbers[:5]}")
    print(f"RNKD  {ranked_hit}: {ranked_numbers[:5]}")

# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────
n = len(eval_queries)
print(f"\n{'═'*80}")
print(f"PHASE 0 RETRIEVAL AUDIT — SUMMARY ({n} queries)")
print(f"{'═'*80}")
print(f"{'Stage':<10} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>8}")
print(f"{'─'*40}")
for stage_name, s in stages.items():
    r1  = 100 * s["hits1"] / n
    r3  = 100 * s["hits3"] / n
    r5  = 100 * s["hits5"] / n
    mrr = sum(s["rr"]) / n
    print(f"{stage_name:<10} {r1:>5.1f}% {r3:>5.1f}% {r5:>5.1f}% {mrr:>8.3f}")

print(f"\n{'─'*80}")
print("DIAGNOSIS:")
bm25_r5  = 100 * stages['bm25']['hits5'] / n
faiss_r5 = 100 * stages['faiss']['hits5'] / n
hyb_r5   = 100 * stages['hybrid']['hits5'] / n
rnk_r5   = 100 * stages['ranked']['hits5'] / n

if hyb_r5 > 0 and rnk_r5 == 0:
    print("  ⚠️  Hybrid retrieves correctly but deterministic ranker pushes hits below rank-5!")
elif faiss_r5 == 0 and bm25_r5 > 30:
    print("  ⚠️  FAISS semantic recall is the weak link. BM25 is carrying the retrieval.")
elif bm25_r5 == 0 and faiss_r5 > 30:
    print("  ⚠️  BM25 lexical recall is the weak link. FAISS is carrying the retrieval.")
elif hyb_r5 == 0 and (bm25_r5 > 0 or faiss_r5 > 0):
    print("  ⚠️  Both BM25 and FAISS individually find hits but RRF is losing them (fusion bug).")
elif hyb_r5 == 0 and bm25_r5 == 0 and faiss_r5 == 0:
    print("  🔴 ALL stages return 0% — this is likely an evaluation STRING-MATCHING bug!")
    print("  👉 Check: does expected 'IS 10322 : Part 5 : Sec 3' match returned format?")
else:
    print(f"  ℹ️  BM25:{bm25_r5:.0f}% FAISS:{faiss_r5:.0f}% Hybrid:{hyb_r5:.0f}% Ranked:{rnk_r5:.0f}%")
print(f"{'═'*80}\n")
