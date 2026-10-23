"""
Builds a session-scoped FAISS + BM25 index from uploaded-receipt chunks —
following scripts/embed_index.py's exact pattern (same model, same
IndexFlatIP-on-normalized-vectors setup, same metadata.jsonl/index_config.json
shape) so the result is loadable by the UNMODIFIED HybridRetriever
(scripts/fusion.py) with zero changes to that class.

This is new code, not a copy of embed_index.py, for one concrete reason:
embed_index.py is a CLI script with no reusable function, and shelling out
to it per upload would force a ~13s cold model-load on every single file a
user adds. This module keeps one embedding model warm for the life of the
server process instead.

Runs the embedding model on CPU by default (UPLOAD_EMBEDDING_DEVICE),
deliberately separate from the demo/research pipeline's GPU-loaded copy —
loading a second ~1.3GB GPU copy alongside the reranker (~2.2GB) and
Qwen3 8B via Ollama (~5.2GB) risks exceeding this machine's 8GB VRAM
(measured in PROGRESS.md). CPU embedding was measured at ~468ms/query —
fine for a one-time per-upload indexing step, not a per-query cost.
"""

import json
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from bm25_retrieve import BM25Retriever  # noqa: E402 — forces a fresh BM25 cache build below

_model = None


def _get_model(model_name: str, device: str):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(model_name, device=device)
    return _model


def append_chunks(chunks_path: Path, new_chunks: Sequence[dict]) -> None:
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_path, "a", encoding="utf-8") as f:
        for c in new_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def rebuild_index(chunks_path: Path, index_dir: Path, model_name: str, device: str) -> int:
    """Rebuilds the FAISS + BM25 index from ALL chunks currently in
    chunks_path (cheap at session scale — a handful of receipts, not
    thousands). Returns the number of chunks indexed."""
    import faiss

    chunks = []
    if chunks_path.exists():
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))

    index_dir.mkdir(parents=True, exist_ok=True)

    if not chunks:
        # Still write an empty-but-valid index so readiness checks / a
        # just-created session don't 404 on missing files.
        faiss.write_index(faiss.IndexFlatIP(1024), str(index_dir / "faiss.index"))
        (index_dir / "metadata.jsonl").write_text("")
        return 0

    model = _get_model(model_name, device)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, str(index_dir / "faiss.index"))

    metadata = [
        {"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "source": c["source"],
         "text": c["text"], "ground_truth": c["ground_truth"]}
        for c in chunks
    ]
    with open(index_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for m in metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    with open(index_dir / "index_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name, "dim": dim, "num_vectors": index.ntotal,
            "normalize_embeddings": True,
            "similarity": "cosine (via inner product on normalized vectors)",
            "query_prefix": "Represent this sentence for searching relevant passages: ",
        }, f, indent=2)

    # Force BM25Retriever to rebuild its cache for this session rather than
    # reuse a stale pickle from a prior file in the same session — cheap at
    # this scale, and correctness matters more than the save here.
    bm25_cache = index_dir / "bm25_index.pkl"
    if bm25_cache.exists():
        os.remove(bm25_cache)
    BM25Retriever(chunks_path=str(chunks_path), cache_dir=str(index_dir))

    return len(chunks)
