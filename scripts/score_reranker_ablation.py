"""
Step 5: Score the reranker ablation.

Reuses (imports, does not duplicate) score_system_run / score_filtered_subset
from compute_objective_metrics.py — applied to a TOP-10 TRUNCATED copy of
each new log's records, since the logs themselves keep the full reranked
top-50 (truncation to the final top-10 is a scoring-time decision, per the
plan, so rank movement can still be computed for chunks outside the top-10).

Also computes RANK MOVEMENT (pre-rerank rank minus post-rerank rank, for
each single-target query's correct chunk) directly from the full,
untruncated log — this needs the full top-50, not the top-10 slice.

Loads Step 4's existing data/eval/objective_metrics_report.json read-only
for the "Hybrid baseline" comparison row — never re-derives or overwrites
Step 4's own numbers.

Usage:
    python scripts/score_reranker_ablation.py
"""

import argparse
import json
import os

from compute_objective_metrics import score_system_run, score_filtered_subset

NEW_SYSTEM_RUNS = [
    ("hybrid_reranked", "none"),
    ("metadata_hybrid_reranked", "ground_truth"),
    ("metadata_hybrid_reranked", "transformer"),
]

FINAL_TOP_K = 10


def read_log(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def truncate_records(records, k=FINAL_TOP_K):
    """Returns a copy of records with retrieved_chunks (and reranker_scores)
    sliced to the first k entries — they're already sorted by rerank score
    descending, so first k = final top-k."""
    truncated = []
    for r in records:
        r2 = dict(r)
        r2["retrieved_chunks"] = r["retrieved_chunks"][:k]
        truncated.append(r2)
    return truncated


def compute_rank_movement(records):
    """Uses the FULL (untruncated) retrieved_chunks list — a target chunk
    might be outside the final top-10 both before and after reranking, and
    that's still a real, reportable "not recovered" case, not something to
    silently drop."""
    movements = []
    not_recovered = 0
    n_single_target = 0

    for r in records:
        target_id = r["extra"].get("target_chunk_id")
        if not target_id:
            continue
        n_single_target += 1

        post_rank, pre_rank = None, None
        for i, c in enumerate(r["retrieved_chunks"], start=1):
            if c["chunk_id"] == target_id:
                post_rank = i
                pre_rank = c["pre_rerank_rank"]
                break

        if post_rank is None:
            not_recovered += 1
            continue
        # positive = reranker moved the correct chunk to a BETTER (lower) rank
        movements.append(pre_rank - post_rank)

    return {
        "n_single_target": n_single_target,
        "n_rank_computed": len(movements),
        "n_not_recovered_in_pool": not_recovered,
        "mean_rank_movement": sum(movements) / len(movements) if movements else None,
    }


def compute_latency(records):
    retrieval = [r["retrieval_latency_ms"] for r in records if r.get("retrieval_latency_ms") is not None]
    rerank = [r["extra"]["rerank_latency_ms"] for r in records if r.get("extra", {}).get("rerank_latency_ms") is not None]
    mean_retrieval = sum(retrieval) / len(retrieval) if retrieval else None
    mean_rerank = sum(rerank) / len(rerank) if rerank else None
    mean_total = (mean_retrieval or 0) + (mean_rerank or 0) if (retrieval and rerank) else None
    return {
        "mean_retrieval_latency_ms": mean_retrieval,
        "mean_rerank_latency_ms": mean_rerank,
        "mean_total_latency_ms": mean_total,
    }


def score_new_system(log_dir, system, metadata_source):
    path = os.path.join(log_dir, f"{system}__{metadata_source}.jsonl")
    records = read_log(path)

    truncated = truncate_records(records, k=FINAL_TOP_K)
    objective = score_system_run(truncated)
    filtered = score_filtered_subset(truncated)
    rank_movement = compute_rank_movement(records)
    latency = compute_latency(records)

    result = {
        "hit_at_10": objective["hit_at_10"],
        "mrr": objective["mrr"],
        "n_single_target_queries": objective["n_single_target_queries"],
        "recall_at_10_small_sets": objective["recall_at_10_small_sets"],
        "precision_at_10_small_sets": objective["precision_at_10_small_sets"],
        "filtered_queries_only": filtered,
        "rank_movement": rank_movement,
        "latency": latency,
    }
    return result


def load_baseline(objective_metrics_path):
    with open(objective_metrics_path, "r", encoding="utf-8") as f:
        step4_report = json.load(f)
    baseline = step4_report["per_system"]["hybrid__none"]
    baseline_filtered = step4_report.get("per_system_filtered_queries_only", {}).get("metadata_hybrid__ground_truth")
    return {
        "hit_at_10": baseline["hit_at_10"],
        "mrr": baseline["mrr"],
        "n_single_target_queries": baseline["n_single_target_queries"],
        "recall_at_10_small_sets": baseline["recall_at_10_small_sets"],
        "precision_at_10_small_sets": baseline["precision_at_10_small_sets"],
        "filtered_queries_only": baseline_filtered,
        "rank_movement": None,  # Step 4 didn't track this — not a reranked system
        "latency": None,        # Step 4 logged retrieval latency only, no rerank stage
        "note": "Read-only reference from Step 4's objective_metrics_report.json — not rerun.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--baseline-report", default="data/eval/objective_metrics_report.json")
    ap.add_argument("--out", default="data/eval/reranker_ablation_report.json")
    args = ap.parse_args()

    report = {"hybrid_baseline": load_baseline(args.baseline_report)}

    for system, metadata_source in NEW_SYSTEM_RUNS:
        key = f"{system}__{metadata_source}"
        path = os.path.join(args.log_dir, f"{key}.jsonl")
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping")
            continue
        report[key] = score_new_system(args.log_dir, system, metadata_source)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote {args.out}\n")
    print(f"{'Variant':40s} {'Hit@10':>8s} {'MRR':>8s} {'RankMove':>10s} {'RerankMs':>10s}")
    for key, v in report.items():
        rm = v["rank_movement"]["mean_rank_movement"] if v.get("rank_movement") and v["rank_movement"]["mean_rank_movement"] is not None else float("nan")
        lat = v["latency"]["mean_rerank_latency_ms"] if v.get("latency") and v["latency"]["mean_rerank_latency_ms"] is not None else float("nan")
        print(f"{key:40s} {v['hit_at_10']:>8.3f} {v['mrr']:>8.3f} {rm:>10.2f} {lat:>10.1f}")


if __name__ == "__main__":
    main()
