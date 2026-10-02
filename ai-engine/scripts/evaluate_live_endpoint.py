import json
import os
import time
import urllib.request
import urllib.error

def load_queries(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_is_number(is_num: str) -> str:
    return is_num.lower().replace(" ", "").replace(":", "")

def main():
    endpoint = "https://project-26108.onrender.com/recommend"
    print(f"🚀 Evaluating live AI Engine at: {endpoint}")
    
    queries_path = os.path.join(os.path.dirname(__file__), "..", "data", "evaluation_queries.json")
    queries = load_queries(queries_path)
    
    print(f"\n📊 Starting Evaluation on {len(queries)} Queries\n")
    
    results_log = []
    
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    mrr_sum = 0.0
    
    start_time = time.time()
    
    for i, q in enumerate(queries):
        query_text = q["query"]
        expected_is = q["expected_is_number"]
        expected_clean = clean_is_number(expected_is)
        
        # Build Request
        payload = json.dumps({"query": query_text, "top_k": 10}).encode('utf-8')
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        req = urllib.request.Request(endpoint, data=payload, headers=headers)
        
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                
                # Render's new /recommend returns a "recommendations" list
                recommendations = res_data.get("recommendations", [])
                
                rank = -1
                for idx, std in enumerate(recommendations):
                    std_is = clean_is_number(std.get("is_number", ""))
                    if std_is == expected_clean or std_is in expected_clean or expected_clean in std_is:
                        rank = idx + 1
                        break
                        
                if rank == 1:
                    hits_at_1 += 1
                if 1 <= rank <= 3:
                    hits_at_3 += 1
                if 1 <= rank <= 5:
                    hits_at_5 += 1
                if rank > 0:
                    mrr_sum += (1.0 / rank)
                    
                status_icon = "✅" if 1 <= rank <= 5 else "❌"
                rank_str = str(rank) if rank > 0 else "Missing"
                
                print(f"[{i+1}/{len(queries)}] {status_icon} Query: '{query_text}' | Expected: {expected_is} | Rank: {rank_str}")
                
                results_log.append({
                    "query": query_text,
                    "expected": expected_is,
                    "rank": rank,
                    "category": q.get("category", "General")
                })
                
        except urllib.error.HTTPError as e:
            print(f"[{i+1}/{len(queries)}] ❌ Query: '{query_text}' | HTTPError: {e.code} - {e.read().decode('utf-8')[:50]}...")
        except Exception as e:
            print(f"[{i+1}/{len(queries)}] ❌ Query: '{query_text}' | Error: {e}")
            
        time.sleep(2.5)
            
    duration = time.time() - start_time
    num_queries = len(queries)
    
    recall_1 = (hits_at_1 / num_queries) * 100
    recall_3 = (hits_at_3 / num_queries) * 100
    recall_5 = (hits_at_5 / num_queries) * 100
    mrr = mrr_sum / num_queries
    
    print("\n" + "="*50)
    print("📈 LIVE EVALUATION RESULTS")
    print("="*50)
    print(f"Total Queries: {num_queries}")
    print(f"Execution Time: {duration:.2f} seconds ({duration/num_queries:.2f}s per query)")
    print("-" * 50)
    print(f"Recall@1: {recall_1:.1f}%")
    print(f"Recall@3: {recall_3:.1f}%")
    print(f"Recall@5: {recall_5:.1f}%")
    print(f"MRR:      {mrr:.3f}")
    print("="*50)
    
    # Generate Markdown Report
    report_path = os.path.join(os.path.dirname(__file__), "..", "..", "retrieval_evaluation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# StandIQ AI Engine: Live Evaluation Report\n\n")
        f.write("This report validates the accuracy of the deployed StandIQ AI Engine (`/recommend` endpoint) across a dataset of 30 realistic procurement queries. The engine uses a Hybrid Retrieval architecture (BM25 + Semantic FAISS + RRF) coupled with an LLM-based query understanding layer.\n\n")
        f.write("## 📊 Summary Metrics\n\n")
        f.write("| Metric | Score |\n")
        f.write("|---|---|\n")
        f.write(f"| **Total Queries** | {num_queries} |\n")
        f.write(f"| **Recall@1** | {recall_1:.1f}% |\n")
        f.write(f"| **Recall@3** | {recall_3:.1f}% |\n")
        f.write(f"| **Recall@5** | {recall_5:.1f}% |\n")
        f.write(f"| **MRR (Mean Reciprocal Rank)** | {mrr:.3f} |\n")
        f.write(f"| **Avg. Pipeline Latency** | {duration/num_queries:.3f}s |\n\n")
        
        f.write("## 🔍 Detailed Results\n\n")
        f.write("| Status | Procurement Query | Expected Standard | Top-10 Rank |\n")
        f.write("|:---:|---|---|---|\n")
        for r in results_log:
            icon = "✅" if 1 <= r["rank"] <= 5 else "❌"
            rank_display = str(r["rank"]) if r["rank"] > 0 else "Missing"
            f.write(f"| {icon} | {r['query']} | `{r['expected']}` | {rank_display} |\n")
            
    print(f"\n📝 Detailed report saved to: {report_path}")

if __name__ == "__main__":
    main()
