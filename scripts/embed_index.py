"""
Step 1b: Embedding + FAISS index.

Embeds all chunks from data/chunks/chunks.jsonl using bge-large-en-v1.5 on
GPU, builds a FAISS index, and saves everything needed to run dense
retrieval later: the index itself, plus a parallel metadata list so a
FAISS result position can be mapped back to its chunk_id/doc_id/text.

Usage:
    python scripts/embed_index.py \
        --chunks data/chunks/chunks.jsonl \
        --out-dir data/index \
        --model BAAI/bge-large-en-v1.5 \
        --batch-size 32
"""

import argparse
import json
import os
import time

import numpy as np


def load_chunks(path):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/chunks/chunks.jsonl")
    ap.add_argument("--out-dir", default="data/index")
    ap.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    print(f"Loading chunks from {args.chunks}...")
    chunks = load_chunks(args.chunks)
    print(f"  {len(chunks)} chunks")
    if not chunks:
        print("ERROR: no chunks loaded, nothing to embed.")
        return

    print(f"\nLoading model {args.model}...")
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")
    model = SentenceTransformer(args.model, device=device)

    # bge models recommend a query instruction prefix for asymmetric
    # search (query vs. passage). We're embedding passages (chunks) here,
    # so no prefix needed for the corpus side — queries will need
    # "Represent this sentence for searching relevant passages: " prefixed
    # at retrieval time. Documented so the retrieval script gets this right.
    texts = [c["text"] for c in chunks]

    print(f"\nEmbedding {len(texts)} chunks (batch size {args.batch_size})...")
    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # so we can use inner-product = cosine sim in FAISS
        convert_to_numpy=True,
    )
    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s ({len(texts)/elapsed:.1f} chunks/sec)")
    print(f"  Embedding shape: {embeddings.shape}")

    print("\nBuilding FAISS index...")
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs = cosine similarity
    index.add(embeddings.astype(np.float32))
    print(f"  Index size: {index.ntotal} vectors, dim {dim}")

    os.makedirs(args.out_dir, exist_ok=True)

    index_path = os.path.join(args.out_dir, "faiss.index")
    faiss.write_index(index, index_path)
    print(f"\nWrote FAISS index -> {index_path}")

    # Parallel metadata: position i in this list corresponds to vector i
    # in the FAISS index. This is how a search result (a row index) maps
    # back to an actual chunk_id/doc_id/text/ground_truth.
    metadata = [
        {
            "chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "source": c["source"],
            "text": c["text"],
            "ground_truth": c["ground_truth"],
        }
        for c in chunks
    ]
    metadata_path = os.path.join(args.out_dir, "metadata.jsonl")
    with open(metadata_path, "w", encoding="utf-8") as f:
        for m in metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"Wrote metadata -> {metadata_path}")

    config_path = os.path.join(args.out_dir, "index_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "dim": dim,
            "num_vectors": index.ntotal,
            "normalize_embeddings": True,
            "similarity": "cosine (via inner product on normalized vectors)",
            "query_prefix": "Represent this sentence for searching relevant passages: ",
        }, f, indent=2)
    print(f"Wrote index config -> {config_path}")

    # Sanity check: embed one query and confirm the index actually returns
    # something sensible before we trust it for real retrieval later.
    print("\n--- Sanity check ---")
    sample_query = "Represent this sentence for searching relevant passages: " + \
        "What is the total amount?"
    q_emb = model.encode([sample_query], normalize_embeddings=True, convert_to_numpy=True)
    scores, indices = index.search(q_emb.astype(np.float32), k=3)
    print(f"Query: {sample_query!r}")
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        m = metadata[idx]
        print(f"  #{rank+1} score={score:.3f} chunk_id={m['chunk_id']} "
              f"source={m['source']} text_preview={m['text'][:80]!r}")


if __name__ == "__main__":
    main()
