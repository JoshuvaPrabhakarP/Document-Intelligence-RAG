"""
Step 5: Reranker ablation harness.

Drives the 3 NEW system-runs (variant 1, "Hybrid baseline", already exists
from Step 4 and is not rerun here):
    hybrid_reranked          / none
    metadata_hybrid_reranked / ground_truth
    metadata_hybrid_reranked / transformer

For each: pull a top-50 candidate pool from the existing, UNMODIFIED
retriever classes (HybridRetriever / MetadataRetriever), rerank with
scripts/rerank.py's Reranker (BAAI/bge-reranker-large), and log the full
reranked top-50 via the existing QueryLogger — truncation to the final
top-10 for scoring is a decision made later, in score_reranker_ablation.py,
not baked in here (so rank-movement analysis has the full pool to work
with).

No LLM generation, no Ragas — this ablation is retrieval-side only (see
plan doc). Safe by construction: writes only to NEW log filenames
(hybrid_reranked__*, metadata_hybrid_reranked__*), never touches any of
Step 4's 7 existing log files.

Usage:
    python scripts/run_reranker_ablation.py                # full run
    python scripts/run_reranker_ablation.py --limit 5       # pilot first
"""

import argparse
import json
import os
import time

from fusion import HybridRetriever
from logging_schema import QueryLogger
from metadata_filter import MetadataRetriever
from rerank import Reranker

POOL_SIZE = 50
STEP4_LOG_FILES = [
    "vector_only__none.jsonl",
    "bm25_only__none.jsonl",
    "hybrid__none.jsonl",
    "metadata_vector__ground_truth.jsonl",
    "metadata_vector__transformer.jsonl",
    "metadata_hybrid__ground_truth.jsonl",
    "metadata_hybrid__transformer.jsonl",
]

SYSTEM_RUNS = [
    ("hybrid_reranked", "none"),
    ("metadata_hybrid_reranked", "ground_truth"),
    ("metadata_hybrid_reranked", "transformer"),
]


def load_queries(path, limit=None):
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    if limit:
        queries = queries[:limit]
    return queries


def snapshot_step4_logs(log_dir):
    """Line counts for Step 4's 7 log files, taken BEFORE this script does
    anything — safeguard so a post-run diff can prove nothing was touched."""
    snapshot = {}
    for fname in STEP4_LOG_FILES:
        path = os.path.join(log_dir, fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                snapshot[fname] = sum(1 for line in f if line.strip())
    return snapshot


def verify_step4_logs_untouched(log_dir, before_snapshot):
    after_snapshot = snapshot_step4_logs(log_dir)
    if before_snapshot != after_snapshot:
        raise RuntimeError(
            f"Step 4 log files changed during this run! Before: {before_snapshot}, "
            f"After: {after_snapshot}. This should never happen — this script "
            f"only writes to hybrid_reranked__*/metadata_hybrid_reranked__* files."
        )
    print(f"Verified: all {len(before_snapshot)} Step 4 log files unchanged "
          f"({before_snapshot}).")


def build_retrievers(needed_meta_sources, index_dir, chunks_path):
    retrievers = {"hybrid": HybridRetriever(index_dir=index_dir, chunks_path=chunks_path)}
    for source in needed_meta_sources:
        retrievers[f"meta_{source}"] = MetadataRetriever(
            metadata_source=source, index_dir=index_dir, chunks_path=chunks_path
        )
    return retrievers


def get_pool(system, metadata_source, retrievers, query_text, filters):
    t0 = time.monotonic()
    if system == "hybrid_reranked":
        # pool_size=100 (the underlying dense/BM25 pool feeding RRF fusion)
        # must match Step 4's own fixed hybrid__none baseline, or this
        # reintroduces the exact pool_size confound Step 4 already found and
        # fixed — k=POOL_SIZE is a separate, intentional knob: how many of
        # the FUSED results get exposed to the reranker.
        pool = retrievers["hybrid"].search(query_text, k=POOL_SIZE, pool_size=100)
    elif system == "metadata_hybrid_reranked":
        pool = retrievers[f"meta_{metadata_source}"].search(
            query_text, filters, base="hybrid", k=POOL_SIZE
        )
    else:
        raise ValueError(f"Unknown system {system!r}")
    retrieval_latency_ms = (time.monotonic() - t0) * 1000
    return pool, retrieval_latency_ms


def run_system(system, metadata_source, retrievers, reranker, queries, log_dir):
    logger = QueryLogger(system=system, metadata_source=metadata_source, log_dir=log_dir)
    logger.reset()

    for q in queries:
        filters = q.get("metadata_filter") or {}

        pool, retrieval_latency_ms = get_pool(
            system, metadata_source, retrievers, q["query_text"], filters
        )
        reranked, rerank_latency_ms = reranker.rerank(q["query_text"], pool)

        retrieved_chunks = [
            {
                "chunk_id": c["chunk_id"],
                "score": c["rerank_score"],
                "text": c["text"],
                "pre_rerank_rank": c["pre_rerank_rank"],
            }
            for c in reranked
        ]
        reranker_scores = [c["rerank_score"] for c in reranked]

        filters_applied = filters if (system == "metadata_hybrid_reranked" and filters) else None

        logger.log_query(
            query_id=q["query_id"],
            query_text=q["query_text"],
            question_type=q["question_type"],
            retrieved_chunks=retrieved_chunks,
            metadata_filters_applied=filters_applied,
            reranker_scores=reranker_scores,
            ground_truth_answer=q.get("ground_truth_answer"),
            retrieval_latency_ms=retrieval_latency_ms,
            extra={
                "target_chunk_id": q.get("target_chunk_id"),
                "target_chunk_ids": q.get("target_chunk_ids"),
                "rerank_latency_ms": rerank_latency_ms,
                "pool_size": POOL_SIZE,
            },
        )

    print(f"  {system}/{metadata_source}: logged {len(queries)} queries -> {logger.path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="data/eval/queries.jsonl")
    ap.add_argument("--index-dir", default="data/index")
    ap.add_argument("--chunks", default="data/chunks/chunks.jsonl")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--limit", type=int, default=None,
                     help="Cap the number of queries — use for a pilot before the full run.")
    args = ap.parse_args()

    before_snapshot = snapshot_step4_logs(args.log_dir)
    print(f"Step 4 log snapshot (before): {before_snapshot}\n")

    queries = load_queries(args.queries, limit=args.limit)
    if not queries:
        print(f"ERROR: no queries loaded from {args.queries}")
        return

    print("Loading retrievers (hybrid + metadata ground_truth/transformer)...")
    retrievers = build_retrievers({"ground_truth", "transformer"}, args.index_dir, args.chunks)

    print("Loading reranker (BAAI/bge-reranker-large)...")
    reranker = Reranker()

    print(f"\nRunning {len(SYSTEM_RUNS)} system-runs over {len(queries)} queries each...\n")
    for system, metadata_source in SYSTEM_RUNS:
        print(f"=== {system} / {metadata_source} ===")
        run_system(system, metadata_source, retrievers, reranker, queries, args.log_dir)

    print()
    verify_step4_logs_untouched(args.log_dir, before_snapshot)
    print("\nDone.")


if __name__ == "__main__":
    main()
