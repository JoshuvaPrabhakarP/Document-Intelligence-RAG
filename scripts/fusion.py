"""
Step 2b: Hybrid retrieval via Reciprocal Rank Fusion (RRF).

Combines dense (FAISS) and BM25 rankings using RRF — score is based on
RANK POSITION in each list, not raw score values, which sidesteps the
score-normalization problem entirely (cosine similarity and BM25 scores
live on completely different scales and aren't meaningfully comparable
as raw numbers).

RRF formula, standard form:
    rrf_score(chunk) = sum over each ranked list containing the chunk of:
        1 / (k + rank)
    where rank is 1-indexed position in that list, and k is a constant
    (60 is the standard default from the original RRF paper — it dampens
    the influence of very top-ranked results so the fusion isn't dominated
    by whichever single list happens to rank one chunk #1).

Usage as a library:
    from fusion import HybridRetriever
    retriever = HybridRetriever()
    results = retriever.search("What is the total amount?", k=10)

Usage as a CLI (side-by-side comparison of all three modes):
    python scripts/fusion.py --query "What is the total amount?"
"""

import argparse

from bm25_retrieve import BM25Retriever
from dense_retrieve import DenseRetriever

RRF_K = 60


def reciprocal_rank_fusion(ranked_lists, k=RRF_K):
    """
    ranked_lists: list of lists, each inner list is a rank-ordered list of
    result dicts (each must have 'chunk_id'). Order within each inner list
    IS the rank — position 0 is rank 1, etc.

    Returns: list of chunk_ids sorted by fused RRF score (descending),
    plus a lookup of the fused scores themselves.
    """
    rrf_scores = {}
    chunk_lookup = {}  # chunk_id -> full result dict, from whichever list first saw it

    for ranked_list in ranked_lists:
        for rank, result in enumerate(ranked_list, start=1):
            chunk_id = result["chunk_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            if chunk_id not in chunk_lookup:
                chunk_lookup[chunk_id] = result

    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
    return sorted_ids, rrf_scores, chunk_lookup


class HybridRetriever:
    def __init__(self, index_dir="data/index", chunks_path="data/chunks/chunks.jsonl"):
        self.dense = DenseRetriever(index_dir=index_dir)
        self.bm25 = BM25Retriever(chunks_path=chunks_path, cache_dir=index_dir)

    def search(self, query, k=10, pool_size=50):
        """
        pool_size: how many candidates to pull from EACH retriever before
        fusing — needs to be >= k, and larger than k gives RRF more to
        work with (a chunk ranked #30 by dense but #2 by BM25 can still
        surface near the top after fusion, but only if the dense pool was
        large enough to include it at all).
        """
        dense_results = self.dense.search(query, k=pool_size)
        bm25_results = self.bm25.search(query, k=pool_size)

        sorted_ids, rrf_scores, chunk_lookup = reciprocal_rank_fusion(
            [dense_results, bm25_results]
        )

        results = []
        for chunk_id in sorted_ids[:k]:
            base = chunk_lookup[chunk_id]
            results.append({
                "chunk_id": chunk_id,
                "score": rrf_scores[chunk_id],  # RRF score, NOT comparable to raw dense/bm25 scores
                "text": base["text"],
                "source": base["source"],
                "ground_truth": base["ground_truth"],
                "in_dense_results": any(r["chunk_id"] == chunk_id for r in dense_results),
                "in_bm25_results": any(r["chunk_id"] == chunk_id for r in bm25_results),
            })
        return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default="data/index")
    ap.add_argument("--chunks", default="data/chunks/chunks.jsonl")
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    print("Loading retrievers (dense model + BM25 index)...")
    hybrid = HybridRetriever(index_dir=args.index_dir, chunks_path=args.chunks)

    print(f"\n{'='*70}\nQuery: {args.query!r}\n{'='*70}")

    print("\n--- Dense only ---")
    for rank, r in enumerate(hybrid.dense.search(args.query, k=args.k)):
        print(f"#{rank+1} score={r['score']:.3f} {r['chunk_id']} ({r['source']}): "
              f"{r['text'][:70]!r}")

    print("\n--- BM25 only ---")
    bm25_res = hybrid.bm25.search(args.query, k=args.k)
    if not bm25_res:
        print("  (no term overlap)")
    for rank, r in enumerate(bm25_res):
        print(f"#{rank+1} score={r['score']:.3f} {r['chunk_id']} ({r['source']}): "
              f"{r['text'][:70]!r}")

    print("\n--- Hybrid (RRF fusion) ---")
    for rank, r in enumerate(hybrid.search(args.query, k=args.k)):
        sources = []
        if r["in_dense_results"]:
            sources.append("dense")
        if r["in_bm25_results"]:
            sources.append("bm25")
        print(f"#{rank+1} rrf_score={r['score']:.4f} found_by={'+'.join(sources)} "
              f"{r['chunk_id']} ({r['source']}): {r['text'][:70]!r}")


if __name__ == "__main__":
    main()
