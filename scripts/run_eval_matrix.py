"""
Step 4: The core 5x2 evaluation matrix harness.

Drives every existing retriever (scripts/dense_retrieve.py, bm25_retrieve.py,
fusion.py, metadata_filter.py) across the query set built by
build_eval_queries.py, logs every query via the existing QueryLogger
(scripts/logging_schema.py), and generates an answer with the local LLM
(scripts/llm_client.py) ONLY for queries flagged `ragas_subset: true` in the
query set — objective Recall@10/MRR never need an LLM at all, and
generation is the slow part, so this is a deliberate cost control (agreed
before running the full matrix).

Seven system-runs, matching Section 8 of PROJECT_GRAPH_REPORT.md's 5x2
design table (non-metadata rows run once; metadata rows run twice, once
per metadata source):
    vector_only    / none
    bm25_only      / none
    hybrid         / none
    metadata_vector   / ground_truth
    metadata_vector   / transformer
    metadata_hybrid   / ground_truth
    metadata_hybrid   / transformer

A single MetadataRetriever instance per metadata source serves BOTH its
metadata_vector (base="vector") and metadata_hybrid (base="hybrid") rows —
no need for four separate metadata-retriever objects.

Usage:
    python scripts/run_eval_matrix.py                  # full run
    python scripts/run_eval_matrix.py --limit 5         # pilot run first
    python scripts/run_eval_matrix.py --systems vector_only,hybrid
"""

import argparse
import json
import os
import time

from bm25_retrieve import BM25Retriever
from dense_retrieve import DenseRetriever
from fusion import HybridRetriever, RRF_K
from logging_schema import QueryLogger
from metadata_filter import MetadataRetriever

ALL_SYSTEM_RUNS = [
    ("vector_only", "none"),
    ("bm25_only", "none"),
    ("hybrid", "none"),
    ("metadata_vector", "ground_truth"),
    ("metadata_vector", "transformer"),
    ("metadata_hybrid", "ground_truth"),
    ("metadata_hybrid", "transformer"),
]

TOP_K = 10
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
META_HYBRID_POOL_SIZE = 100  # must match MetadataRetriever's default (metadata_filter.py)


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


def write_run_config(system_runs, out_path):
    config = {
        "systems": [f"{s}__{m}" for s, m in system_runs],
        "top_k": TOP_K,
        "ragas_subset_only_for_generation": True,
        "llm": "qwen3:8b",
        "ollama_base_url": "http://localhost:11434/v1",
        "llm_thinking": False,
        "fusion": "RRF",
        "rrf_k": RRF_K,
        "hybrid_pool_size": META_HYBRID_POOL_SIZE,
        "embedding_model": EMBEDDING_MODEL,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return config


def build_retrievers(system_runs, index_dir, chunks_path):
    """Only builds the retriever objects actually needed for the requested
    system_runs, and reuses one MetadataRetriever per metadata source across
    both its metadata_vector and metadata_hybrid rows."""
    needed_systems = {s for s, _ in system_runs}
    needed_meta_sources = {m for s, m in system_runs if s in ("metadata_vector", "metadata_hybrid")}

    retrievers = {}
    if "vector_only" in needed_systems:
        print("Loading dense retriever...")
        retrievers["vector_only"] = DenseRetriever(index_dir=index_dir)
    if "bm25_only" in needed_systems:
        print("Loading BM25 retriever...")
        retrievers["bm25_only"] = BM25Retriever(chunks_path=chunks_path, cache_dir=index_dir)
    if "hybrid" in needed_systems:
        print("Loading hybrid retriever...")
        retrievers["hybrid"] = HybridRetriever(index_dir=index_dir, chunks_path=chunks_path)
    for source in needed_meta_sources:
        print(f"Loading metadata retriever ({source})...")
        retrievers[f"meta_{source}"] = MetadataRetriever(metadata_source=source,
                                                           index_dir=index_dir,
                                                           chunks_path=chunks_path)
    return retrievers


def search(system, metadata_source, retrievers, query_text, filters, k=TOP_K):
    if system == "vector_only":
        return retrievers["vector_only"].search(query_text, k=k)
    if system == "bm25_only":
        return retrievers["bm25_only"].search(query_text, k=k)
    if system == "hybrid":
        # pool_size=100 to match MetadataRetriever's default hybrid pool depth
        # (metadata_filter.py) — RRF's score depends on which candidates enter
        # the fusion sum at all, so a shallower pool here would make plain
        # hybrid and metadata_hybrid differ from pool depth alone, contaminating
        # any comparison of the metadata filter's actual effect (confirmed via
        # a pilot mismatch: metadata_hybrid scored differently from hybrid even
        # on queries with an empty filter, where the two should be identical).
        return retrievers["hybrid"].search(query_text, k=k, pool_size=META_HYBRID_POOL_SIZE)
    if system == "metadata_vector":
        return retrievers[f"meta_{metadata_source}"].search(query_text, filters, base="vector", k=k)
    if system == "metadata_hybrid":
        return retrievers[f"meta_{metadata_source}"].search(query_text, filters, base="hybrid", k=k)
    raise ValueError(f"Unknown system {system!r}")


def run_system(system, metadata_source, retrievers, queries, log_dir):
    logger = QueryLogger(system=system, metadata_source=metadata_source, log_dir=log_dir)
    logger.reset()

    n_generated = 0
    for q in queries:
        filters = q.get("metadata_filter") or {}

        t0 = time.monotonic()
        results = search(system, metadata_source, retrievers, q["query_text"], filters)
        retrieval_latency_ms = (time.monotonic() - t0) * 1000

        retrieved_chunks = [
            {"chunk_id": r["chunk_id"], "score": r["score"], "text": r["text"]}
            for r in results
        ]

        generated_answer = None
        generation_latency_ms = None
        if q.get("ragas_subset") and results:
            from llm_client import generate_answer
            generated_answer, generation_latency_ms = generate_answer(q["query_text"], results)
            n_generated += 1

        # only the two metadata-aware systems actually apply `filters` in
        # search() above — logging it for vector_only/bm25_only/hybrid would
        # mislabel those runs as filtered when the filter was never used.
        filters_applied = filters if (system in ("metadata_vector", "metadata_hybrid") and filters) else None

        logger.log_query(
            query_id=q["query_id"],
            query_text=q["query_text"],
            question_type=q["question_type"],
            retrieved_chunks=retrieved_chunks,
            metadata_filters_applied=filters_applied,
            generated_answer=generated_answer,
            ground_truth_answer=q.get("ground_truth_answer"),
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            extra={
                "target_chunk_id": q.get("target_chunk_id"),
                "target_chunk_ids": q.get("target_chunk_ids"),
            },
        )

    print(f"  {system}/{metadata_source}: logged {len(queries)} queries "
          f"({n_generated} with generation) -> {logger.path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="data/eval/queries.jsonl")
    ap.add_argument("--index-dir", default="data/index")
    ap.add_argument("--chunks", default="data/chunks/chunks.jsonl")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--run-config-out", default="data/eval/run_config.json")
    ap.add_argument("--limit", type=int, default=None,
                     help="Cap the number of queries per system-run — use for a pilot before the full run.")
    ap.add_argument("--systems", default=None,
                     help="Comma-separated subset of system__metadata_source pairs, "
                          "e.g. vector_only__none,hybrid__none. Default: all 7.")
    args = ap.parse_args()

    if args.systems:
        wanted = set(args.systems.split(","))
        system_runs = [(s, m) for s, m in ALL_SYSTEM_RUNS if f"{s}__{m}" in wanted]
    else:
        system_runs = ALL_SYSTEM_RUNS

    queries = load_queries(args.queries, limit=args.limit)
    if not queries:
        print(f"ERROR: no queries loaded from {args.queries}")
        return

    config = write_run_config(system_runs, args.run_config_out)
    print(f"Run config written -> {args.run_config_out}")
    print(json.dumps(config, indent=2))
    print(f"\nRunning {len(system_runs)} system-runs over {len(queries)} queries each...\n")

    retrievers = build_retrievers(system_runs, args.index_dir, args.chunks)

    for system, metadata_source in system_runs:
        print(f"=== {system} / {metadata_source} ===")
        run_system(system, metadata_source, retrievers, queries, args.log_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
