import os
import json
import hashlib
import traceback
import sys

def main():
    print("--- DIAGNOSTIC START ---")
    data_path = "data/bis_full_knowledge_base.json"
    
    # 4. Does bis_full_knowledge_base.json exist?
    print(f"KB FILE PRESENT: {'PASS' if os.path.exists(data_path) else 'FAIL'} ({data_path})")
    
    base = os.path.splitext(data_path)[0]
    bm25_path = f"{base}_bm25.pkl"
    idx_path  = f"{base}_faiss.index"
    meta_path = f"{base}_index_meta.json"
    
    # 6. Artifact paths correct?
    paths_exist = os.path.exists(bm25_path) and os.path.exists(idx_path) and os.path.exists(meta_path)
    print(f"ARTIFACT PATHS: {'PASS' if paths_exist else 'FAIL'}")
    print(f"  - bm25: {bm25_path} -> {os.path.exists(bm25_path)}")
    print(f"  - faiss: {idx_path} -> {os.path.exists(idx_path)}")
    print(f"  - meta: {meta_path} -> {os.path.exists(meta_path)}")

    # 5. SHA256 Check
    expected_sha = None
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            expected_sha = meta.get("kb_sha256")
            
    if expected_sha and os.path.exists(data_path):
        h = hashlib.sha256()
        with open(data_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        actual_sha = h.hexdigest()
        sha_match = actual_sha == expected_sha
        print(f"SHA256 CHECK: {'PASS' if sha_match else 'FAIL'}")
        print(f"  - Expected: {expected_sha}")
        print(f"  - Actual:   {actual_sha}")
    else:
        print("SHA256 CHECK: SKIP (missing meta or kb)")

    # 1. FAISS IMPORT
    try:
        import faiss
        print("FAISS IMPORT: PASS")
    except Exception as e:
        print(f"FAISS IMPORT: FAIL ({type(e).__name__}: {e})")

    # 3. BM25 LOAD
    try:
        import joblib
        bm25 = joblib.load(bm25_path)
        print("BM25 LOAD: PASS")
    except Exception as e:
        print(f"BM25 LOAD: FAIL ({type(e).__name__}: {e})")

    # 2. FAISS INDEX LOAD
    try:
        import faiss
        index = faiss.read_index(idx_path)
        print("FAISS INDEX LOAD: PASS")
    except Exception as e:
        print(f"FAISS INDEX LOAD: FAIL ({type(e).__name__}: {e})")

    print("--- DIAGNOSTIC END ---")

if __name__ == "__main__":
    main()
