"""
Step 4: Objective retrieval metrics (Recall@10, MRR) — no LLM needed, so
this runs over every query logged by run_eval_matrix.py for all 7
system-runs, not just the Ragas subset.

Two genuinely different scoring modes, because the query set has two
genuinely different kinds of ground truth (see build_eval_queries.py):
  - Single-target queries (type A, most of the set): does the ONE correct
    chunk appear in the top 10, and at what rank -> hit@10, MRR.
  - Set-target queries (type B and the negation/absence type-E queries):
    the correct answer is a whole set of chunks (e.g. every Walmart
    receipt, or every CORD chunk for "no vendor listed") -> recall@10
    (fraction of the target set recovered) and precision@10 (fraction of
    the top 10 that are genuinely in the target set).

    Recall@10 against a 1000-chunk target set (the negation/absence "all of
    CORD" queries) is STRUCTURALLY capped at 10/1000 = 1% for even a
    perfect retriever — that's a property of the metric at this target-set
    size, not a bug, so type-E is reported separately rather than averaged
    in with type-B's much smaller (5-49 chunk) target sets.

Type-D (semantic paraphrase) queries have no verifiable target at all by
design — they're listed in a separate qualitative section (top retrieved
chunks per system) for manual review, matching how the Step 2 "receipt for
refueling a car" finding was actually used: read the real results, don't
force a number onto a case that doesn't have one.

The metadata-quality linkage table (Section 8 of PROJECT_GRAPH_REPORT.md)
is computed ONLY from queries that actually carry a metadata_filter —
ground_truth and transformer metadata sources are indistinguishable on
every other query (MetadataStore.matches() with an empty filter dict always
returns True), so averaging over the full query set would dilute the exact
signal RQ2 is asking about.

Usage:
    python scripts/compute_objective_metrics.py
"""

import argparse
import json
import os

SYSTEM_RUNS = [
    ("vector_only", "none"),
    ("bm25_only", "none"),
    ("hybrid", "none"),
    ("metadata_vector", "ground_truth"),
    ("metadata_vector", "transformer"),
    ("metadata_hybrid", "ground_truth"),
    ("metadata_hybrid", "transformer"),
]

LARGE_SET_THRESHOLD = 100  # target sets bigger than this get reported separately (see module docstring)


def read_log(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def score_single_target(retrieved_ids, target_id):
    if target_id in retrieved_ids:
        rank = retrieved_ids.index(target_id) + 1
        return {"hit": 1, "rank": rank, "reciprocal_rank": 1.0 / rank}
    return {"hit": 0, "rank": None, "reciprocal_rank": 0.0}


def score_set_target(retrieved_ids, target_ids):
    target_set = set(target_ids)
    if not target_set:
        return None
    recovered = target_set & set(retrieved_ids)
    return {
        "recall_at_10": len(recovered) / len(target_set),
        "precision_at_10": len(recovered) / len(retrieved_ids) if retrieved_ids else 0.0,
        "target_set_size": len(target_set),
    }


def score_system_run(records):
    single_scores = []
    set_scores_small = []
    set_scores_large = []
    qualitative = []
    by_question_type = {}
    by_source = {}

    for r in records:
        retrieved_ids = [c["chunk_id"] for c in r["retrieved_chunks"]]
        target_id = r["extra"].get("target_chunk_id")
        target_ids = r["extra"].get("target_chunk_ids")
        qtype = r["question_type"]

        result = None
        if target_id:
            result = score_single_target(retrieved_ids, target_id)
            single_scores.append(result)
        elif target_ids:
            result = score_set_target(retrieved_ids, target_ids)
            if result is None:
                continue
            if result["target_set_size"] > LARGE_SET_THRESHOLD:
                set_scores_large.append(result)
            else:
                set_scores_small.append(result)
        else:
            qualitative.append({
                "query_id": r["query_id"],
                "query_text": r["query_text"],
                "top_3": retrieved_ids[:3],
            })
            continue

        by_question_type.setdefault(qtype, []).append(result)

    def avg(key, items):
        vals = [x[key] for x in items if x.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "n_single_target_queries": len(single_scores),
        "hit_at_10": avg("hit", single_scores),
        "mrr": avg("reciprocal_rank", single_scores),
        "n_set_target_small": len(set_scores_small),
        "recall_at_10_small_sets": avg("recall_at_10", set_scores_small),
        "precision_at_10_small_sets": avg("precision_at_10", set_scores_small),
        "n_set_target_large": len(set_scores_large),
        "recall_at_10_large_sets": avg("recall_at_10", set_scores_large),
        "precision_at_10_large_sets": avg("precision_at_10", set_scores_large),
        "n_qualitative_only": len(qualitative),
        "qualitative_samples": qualitative,
        "by_question_type": {
            qt: {"n": len(items), "hit_at_10": avg("hit", items), "mrr": avg("reciprocal_rank", items),
                 "recall_at_10": avg("recall_at_10", items), "precision_at_10": avg("precision_at_10", items)}
            for qt, items in by_question_type.items()
        },
    }
    return summary


def score_filtered_subset(records):
    """Recall@10/precision@10 restricted to queries that actually carry a
    metadata_filter — the only queries where ground-truth vs. Transformer
    metadata sources can possibly diverge (see module docstring)."""
    filtered_records = [r for r in records if r.get("metadata_filters_applied")]
    scores = []
    for r in filtered_records:
        retrieved_ids = [c["chunk_id"] for c in r["retrieved_chunks"]]
        target_ids = r["extra"].get("target_chunk_ids") or (
            [r["extra"]["target_chunk_id"]] if r["extra"].get("target_chunk_id") else None
        )
        if not target_ids:
            continue
        result = score_set_target(retrieved_ids, target_ids)
        if result:
            scores.append(result)
    if not scores:
        return None
    return {
        "n_filtered_queries": len(scores),
        "recall_at_10": sum(s["recall_at_10"] for s in scores) / len(scores),
        "precision_at_10": sum(s["precision_at_10"] for s in scores) / len(scores),
    }


def build_linkage_table(per_system_filtered, extraction_report_path):
    """Section 8's metadata-quality linkage table: per-field extraction F1
    (from Step 3c) against downstream Recall@10 (from this step), for each
    metadata source, filtered-queries-only."""
    with open(extraction_report_path, "r", encoding="utf-8") as f:
        extraction = json.load(f)

    fuzzy = extraction["fuzzy"]
    table = {
        "ground_truth": {
            "vendor_f1": 1.0, "date_f1": 1.0, "amount_f1": 1.0,
            "retrieval_recall_at_10": per_system_filtered.get("metadata_hybrid__ground_truth", {}).get("recall_at_10"),
        },
        "transformer": {
            "vendor_f1": fuzzy["vendor"]["f1"], "date_f1": fuzzy["date"]["f1"], "amount_f1": fuzzy["amount"]["f1"],
            "retrieval_recall_at_10": per_system_filtered.get("metadata_hybrid__transformer", {}).get("recall_at_10"),
        },
    }
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--extraction-report", default="data/metadata/extraction_accuracy_report.json")
    ap.add_argument("--out", default="data/eval/objective_metrics_report.json")
    args = ap.parse_args()

    report = {"per_system": {}, "per_system_filtered_queries_only": {}}

    for system, metadata_source in SYSTEM_RUNS:
        key = f"{system}__{metadata_source}"
        path = os.path.join(args.log_dir, f"{key}.jsonl")
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping")
            continue
        records = read_log(path)
        report["per_system"][key] = score_system_run(records)

        filtered = score_filtered_subset(records)
        if filtered:
            report["per_system_filtered_queries_only"][key] = filtered

    if os.path.exists(args.extraction_report):
        report["metadata_quality_linkage_table"] = build_linkage_table(
            report["per_system_filtered_queries_only"], args.extraction_report
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote {args.out}\n")
    print("--- Summary (all queries) ---")
    for key, s in report["per_system"].items():
        print(f"{key:35s} hit@10={s['hit_at_10']:.3f} MRR={s['mrr']:.3f}  "
              f"(n={s['n_single_target_queries']}, +{s['n_set_target_small']} small-set, "
              f"+{s['n_set_target_large']} large-set, +{s['n_qualitative_only']} qualitative-only)")

    print("\n--- Filtered-queries-only (metadata signal, small target sets) ---")
    for key, s in report["per_system_filtered_queries_only"].items():
        print(f"{key:35s} recall@10={s['recall_at_10']:.3f} precision@10={s['precision_at_10']:.3f} (n={s['n_filtered_queries']})")

    if "metadata_quality_linkage_table" in report:
        print("\n--- Metadata-quality linkage table ---")
        print(json.dumps(report["metadata_quality_linkage_table"], indent=2))


if __name__ == "__main__":
    main()
