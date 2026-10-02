"""
ai-engine/build_index.py
=========================
One-time offline script to pre-compute and serialize all retrieval indexes.

Run this LOCALLY (not on Render) whenever the knowledge base changes:

    python3.11 build_index.py [--kb data/bis_full_knowledge_base.json] [--out data/]

Outputs written to the --out directory:
    bis_full_knowledge_base_embeddings.npy  — L2-normalised embeddings (already may exist)
    bis_full_knowledge_base_bm25.pkl        — serialized BM25Retriever
    bis_full_knowledge_base_faiss.index     — serialized FAISS IndexFlatIP
    bis_full_knowledge_base_index_meta.json — version/checksum metadata

On Render startup, Recommender will load these artifacts instead of rebuilding.
Memory cost:
  - FAISS IndexFlatIP (1015 × 384 float32) ≈ 1.5 MB on disk, ~1.6 MB in RAM
  - BM25 pickle ≈ 0.5-2 MB
  - Embeddings .npy ≈ 1.5 MB (already cached, shared)
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time

import faiss
import joblib
import numpy as np

# ── path setup so we can import src modules ──────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))

from src.embedding import generate_embeddings
from src.search import BM25Retriever, VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_index")


# ─────────────────────────────────────────────────────────────────────────────

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build(kb_path: str, out_dir: str, force: bool = False) -> None:
    os.makedirs(out_dir, exist_ok=True)

    base     = os.path.splitext(os.path.basename(kb_path))[0]
    emb_path  = os.path.join(out_dir, f"{base}_embeddings.npy")
    bm25_path = os.path.join(out_dir, f"{base}_bm25.pkl")
    idx_path  = os.path.join(out_dir, f"{base}_faiss.index")
    meta_path = os.path.join(out_dir, f"{base}_index_meta.json")

    # ── Load knowledge base ──────────────────────────────────────────────────
    logger.info("Loading knowledge base: %s", kb_path)
    with open(kb_path, "r", encoding="utf-8") as f:
        standards = json.load(f)
    logger.info("  %d standards loaded.", len(standards))

    kb_mtime  = os.path.getmtime(kb_path)
    kb_sha256 = sha256_file(kb_path)

    # ── Check if artifacts are already up-to-date ────────────────────────────
    if not force and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if (
            meta.get("kb_sha256") == kb_sha256
            and meta.get("record_count") == len(standards)
            and os.path.exists(emb_path)
            and os.path.exists(bm25_path)
            and os.path.exists(idx_path)
        ):
            logger.info("All artifacts are up-to-date. Nothing to rebuild. (Use --force to override.)")
            return

    # ── Embeddings ───────────────────────────────────────────────────────────
    if (
        not force
        and os.path.exists(emb_path)
        and os.path.getmtime(emb_path) >= kb_mtime
    ):
        logger.info("Loading existing embedding cache: %s", emb_path)
        embeddings = np.load(emb_path)
        if embeddings.shape[0] != len(standards):
            logger.warning(
                "Cache size mismatch (%d vs %d) — regenerating embeddings.",
                embeddings.shape[0], len(standards),
            )
            embeddings = None
    else:
        embeddings = None

    if embeddings is None:
        logger.info("Generating embeddings for %d standards (this takes ~30s)…", len(standards))
        t0 = time.perf_counter()
        search_texts = [std.get("search_text", "") for std in standards]
        embeddings = generate_embeddings(search_texts)
        logger.info("  Embeddings generated in %.1fs — shape %s", time.perf_counter() - t0, embeddings.shape)
        np.save(emb_path, embeddings)
        logger.info("  Saved to %s", emb_path)

    dim = embeddings.shape[1]

    # ── BM25 ─────────────────────────────────────────────────────────────────
    logger.info("Building BM25 index over %d documents…", len(standards))
    t0 = time.perf_counter()
    search_texts = [std.get("search_text", "") for std in standards]
    bm25 = BM25Retriever().fit(search_texts)
    logger.info("  BM25 built in %.2fs", time.perf_counter() - t0)

    joblib.dump(bm25, bm25_path, compress=3)
    bm25_size_kb = os.path.getsize(bm25_path) / 1024
    logger.info("  BM25 saved to %s (%.1f KB)", bm25_path, bm25_size_kb)

    # ── FAISS ────────────────────────────────────────────────────────────────
    logger.info("Building FAISS IndexFlatIP (dim=%d)…", dim)
    t0 = time.perf_counter()
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))
    logger.info("  FAISS index built in %.2fs — %d vectors", time.perf_counter() - t0, index.ntotal)

    faiss.write_index(index, idx_path)
    idx_size_kb = os.path.getsize(idx_path) / 1024
    logger.info("  FAISS index saved to %s (%.1f KB)", idx_path, idx_size_kb)

    # ── Metadata ─────────────────────────────────────────────────────────────
    meta = {
        "kb_path":      kb_path,
        "kb_sha256":    kb_sha256,
        "record_count": len(standards),
        "embedding_dim": dim,
        "faiss_ntotal": index.ntotal,
        "built_at":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifacts": {
            "embeddings": emb_path,
            "bm25":       bm25_path,
            "faiss":      idx_path,
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("  Metadata saved to %s", meta_path)

    logger.info("✅ All artifacts ready. Render startup will load these instead of rebuilding.")
    logger.info("   FAISS: %.1f KB | BM25: %.1f KB | Embeddings: %.1f KB",
                idx_size_kb, bm25_size_kb, os.path.getsize(emb_path) / 1024)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-build StandIQ retrieval indexes.")
    parser.add_argument(
        "--kb",
        default="data/bis_full_knowledge_base.json",
        help="Path to the knowledge base JSON (default: data/bis_full_knowledge_base.json)",
    )
    parser.add_argument(
        "--out",
        default="data/",
        help="Output directory for serialized indexes (default: data/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if artifacts are up-to-date.",
    )
    args = parser.parse_args()

    # Resolve paths relative to this script's location
    kb_path = args.kb if os.path.isabs(args.kb) else os.path.join(_HERE, args.kb)
    out_dir = args.out if os.path.isabs(args.out) else os.path.join(_HERE, args.out)

    if not os.path.exists(kb_path):
        logger.error("Knowledge base not found: %s", kb_path)
        sys.exit(1)

    build(kb_path, out_dir, force=args.force)
