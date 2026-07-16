"""
Step 6: Error analysis — from logs, no re-running.

Purely reads what already exists: the 10 system-run logs (7 from Step 4 +
3 reranked from Step 5), the query set, the chunk corpus, and the 4 existing
summary reports (objective metrics, Ragas, reranker ablation, extraction
accuracy). Makes zero new retrieval calls, zero new LLM calls, needs no GPU.
The only things this script ever writes are data/eval/error_analysis_report.json
(new file) — it never touches logs/*.jsonl or any other existing report.

Six error categories, each grounded in a specific thread this project already
flagged as "worth a closer look in Step 6" rather than invented fresh:
  1. BM25 keyword-collision failures      (Section 12)
  2. Retrieval-depth misses                (Section 15 rank-movement)
  3. Metadata-quality-erosion masking       (Section 14 "12/14 identical")
  4. Reranker cluster-breaking              (Section 15 small-set regression)
  5. Semantic-paraphrase (type D) review    (qualitative, no target)
  6. Negation/absence (type E) structural cap (documented, not investigated)

Plus two additions:
  - Per-source (WildReceipt/SROIE/CORD) breakdown across all 10 systems.
  - Per-field (vendor/date/amount) metadata coverage — explicitly documenting
    where the current query set does NOT exercise a field at all.

Every category includes concrete query/chunk examples and the evidence used
to classify them — not just aggregate counts.

Usage:
    python scripts/error_analysis.py              # print only, for spot-check
    python scripts/error_analysis.py --write       # also write the report file
"""

import argparse
import json
import os

STEP4_SYSTEMS = [
    ("vector_only", "none"),
    ("bm25_only", "none"),
    ("hybrid", "none"),
    ("metadata_vector", "ground_truth"),
    ("metadata_vector", "transformer"),
    ("metadata_hybrid", "ground_truth"),
    ("metadata_hybrid", "transformer"),
]
STEP5_SYSTEMS = [
    ("hybrid_reranked", "none"),
    ("metadata_hybrid_reranked", "ground_truth"),
    ("metadata_hybrid_reranked", "transformer"),
]
ALL_SYSTEMS = STEP4_SYSTEMS + STEP5_SYSTEMS
SOURCES = ["wildreceipt", "sroie", "cord"]


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_all(log_dir="logs", chunks_path="data/chunks/chunks.jsonl",
             queries_path="data/eval/queries.jsonl", eval_dir="data/eval"):
    chunks = {c["chunk_id"]: c for c in read_jsonl(chunks_path)}
    queries = {q["query_id"]: q for q in read_jsonl(queries_path)}

    logs = {}
    for system, metadata_source in ALL_SYSTEMS:
        key = f"{system}__{metadata_source}"
        path = os.path.join(log_dir, f"{key}.jsonl")
        logs[key] = {r["query_id"]: r for r in read_jsonl(path)}

    def load_json(name):
        path = os.path.join(eval_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    reports = {
        "objective": load_json("objective_metrics_report.json"),
        "ragas": load_json("ragas_report.json"),
        "reranker": load_json("reranker_ablation_report.json"),
        "extraction": json.load(open("data/metadata/extraction_accuracy_report.json")),
    }

    return chunks, queries, logs, reports


def top_ids(record, k=10):
    return [c["chunk_id"] for c in record["retrieved_chunks"][:k]]


def snippet(chunks, chunk_id, n=90):
    return chunks[chunk_id]["text"][:n].replace("\n", " ")


# --- Category 1: BM25 keyword-collision failures ---

def category_bm25_collisions(chunks, queries, logs):
    examples = []
    bm25_log = logs["bm25_only__none"]
    vector_log = logs["vector_only__none"]

    for qid, q in queries.items():
        target = q.get("target_chunk_id")
        if not target:
            continue
        bm25_rec, vector_rec = bm25_log.get(qid), vector_log.get(qid)
        if not bm25_rec or not vector_rec:
            continue

        bm25_hit = target in top_ids(bm25_rec)
        vector_hit = target in top_ids(vector_rec)
        if bm25_hit or not vector_hit:
            continue  # only interested in cases BM25 uniquely misses

        bm25_top1 = bm25_rec["retrieved_chunks"][0]["chunk_id"] if bm25_rec["retrieved_chunks"] else None
        if not bm25_top1:
            continue

        # Spot-check found a real classification error here: a "wrong" top-1
        # from the SAME vendor as the target (e.g. two different AEON
        # receipts) is a same-vendor disambiguation miss, not the car-wash-
        # style topical collision this category is actually about — exclude
        # same-vendor cases so the category only reports genuine
        # different-topic false positives.
        target_vendor = chunks[target]["ground_truth"].get("vendor")
        top1_vendor = chunks[bm25_top1]["ground_truth"].get("vendor")
        if target_vendor and top1_vendor and target_vendor.strip().lower() == top1_vendor.strip().lower():
            continue

        query_words = set(w.lower().strip(".,!?#") for w in q["query_text"].split())
        top1_text = chunks[bm25_top1]["text"].lower()
        overlap = [w for w in query_words if len(w) >= 3 and w in top1_text]
        if not overlap:
            continue  # no literal-token explanation — not a collision case, just a miss

        examples.append({
            "query_id": qid,
            "query_text": q["query_text"],
            "target_chunk_id": target,
            "target_snippet": snippet(chunks, target),
            "bm25_wrong_top1_chunk_id": bm25_top1,
            "bm25_wrong_top1_snippet": snippet(chunks, bm25_top1),
            "bm25_wrong_top1_source": chunks[bm25_top1]["source"],
            "literal_token_overlap": overlap,
            "evidence": (f"BM25 top-1 shares literal token(s) {overlap} with the query but is a "
                         f"DIFFERENT vendor/topic from the target (target vendor: {target_vendor!r}, "
                         f"BM25's pick vendor: {top1_vendor!r}) — dense retrieval found the correct "
                         f"target, BM25 was pulled off by incidental lexical overlap instead."),
        })

    return {
        "description": "Queries where BM25 uniquely failed (missed the target) while dense "
                        "succeeded, on a DIFFERENT vendor/topic than the target with genuine "
                        "literal-token overlap explaining it — the 'Car#304' keyword-collision "
                        "pattern from Step 2/Section 12. Scoped to single-target factual queries only.",
        "n_found": len(examples),
        "examples": examples[:8],
        "note": (
            "0 found among this set's single-target factual queries — an honest result, not a "
            "sign the pattern doesn't exist. The original Section 12 car-wash collision was "
            "observed on a QUALITATIVE semantic-paraphrase query ('receipt for refueling a car', "
            "type D, no single target), which this category structurally can't see since it only "
            "scans single-target queries. See category 5 below, where that exact case is captured."
        ) if not examples else None,
    }


# --- Category 2: Retrieval-depth misses ---

def category_retrieval_depth_misses(chunks, queries, logs):
    """Step 5's reranked logs keep the full top-50 pool — a genuine 'not in
    pool at all' miss is only detectable there. Step 4's logs only have
    top-10, so we separately report the shallower 'not in top-10' rate for
    those, and are explicit about which is which."""
    per_system_top10_miss = {}
    for system, metadata_source in ALL_SYSTEMS:
        key = f"{system}__{metadata_source}"
        log = logs[key]
        n_single, n_miss = 0, 0
        by_source = {s: {"n": 0, "miss": 0} for s in SOURCES}
        for qid, q in queries.items():
            target = q.get("target_chunk_id")
            if not target or qid not in log:
                continue
            n_single += 1
            src = chunks[target]["source"]
            by_source[src]["n"] += 1
            if target not in top_ids(log[qid], k=10):
                n_miss += 1
                by_source[src]["miss"] += 1
        per_system_top10_miss[key] = {
            "n_single_target": n_single,
            "n_top10_miss": n_miss,
            "top10_miss_rate": n_miss / n_single if n_single else None,
            "by_source": by_source,
        }

    # Deep pool-miss (full top-50) — only meaningful for Step 5's reranked logs
    pool_miss_examples = []
    pool_miss_stats = {}
    for system, metadata_source in STEP5_SYSTEMS:
        key = f"{system}__{metadata_source}"
        log = logs[key]
        n_single, n_pool_miss = 0, 0
        for qid, q in queries.items():
            target = q.get("target_chunk_id")
            if not target or qid not in log:
                continue
            n_single += 1
            full_ids = [c["chunk_id"] for c in log[qid]["retrieved_chunks"]]  # full ~50
            if target not in full_ids:
                n_pool_miss += 1
                if key == "hybrid_reranked__none" and len(pool_miss_examples) < 5:
                    pool_miss_examples.append({
                        "query_id": qid,
                        "query_text": q["query_text"],
                        "target_chunk_id": target,
                        "target_source": chunks[target]["source"],
                        "target_snippet": snippet(chunks, target),
                        "evidence": "Target chunk not present anywhere in the top-50 candidate "
                                    "pool — a genuine retrieval-depth miss reranking cannot fix, "
                                    "since reranking only re-orders what retrieval already found.",
                    })
        pool_miss_stats[key] = {
            "n_single_target": n_single,
            "n_pool_miss_top50": n_pool_miss,
            "pool_miss_rate": n_pool_miss / n_single if n_single else None,
        }

    return {
        "description": "Two depths of 'miss': top-10 miss (available for all 10 systems, but "
                        "reranking could still recover a target ranked 11-50) vs. genuine "
                        "top-50 pool miss (only measurable for Step 5's 3 reranked systems, "
                        "which log the full 50-candidate pool) — a pool miss is a retrieval "
                        "failure no reranker can fix.",
        "top10_miss_by_system": per_system_top10_miss,
        "top50_pool_miss_by_system": pool_miss_stats,
        "top50_pool_miss_examples": pool_miss_examples,
    }


# --- Category 3: Metadata-quality-erosion masking ---

def category_metadata_masking(chunks, queries, logs, reports, transformer_path="data/metadata/transformer_extracted.jsonl"):
    gt_log = logs["metadata_hybrid__ground_truth"]
    tf_log = logs["metadata_hybrid__transformer"]

    # Direct, precise "easy vendor" test: for each filtered query's actual
    # target chunk cluster, does the Transformer's OWN predicted vendor for
    # THOSE SPECIFIC chunks match ground truth? (Not a lookup against the
    # sparse ~5-sample illustrative mismatch list in extraction_accuracy_report.json
    # — that's too thin to say anything meaningful; this checks the real chunks.)
    transformer_by_chunk = {r["chunk_id"]: r["predicted"] for r in read_jsonl(transformer_path)}

    def fuzzy_contains(a, b):
        if not a or not b:
            return False
        a, b = a.strip().lower(), b.strip().lower()
        shorter, longer = sorted([a, b], key=len)
        return len(shorter) >= 2 and shorter in longer

    identical, differing = [], []
    for qid, q in queries.items():
        if not q.get("metadata_filter"):
            continue
        gt_rec, tf_rec = gt_log.get(qid), tf_log.get(qid)
        if not gt_rec or not tf_rec:
            continue
        gt_ids = top_ids(gt_rec)
        tf_ids = top_ids(tf_rec)
        vendor = q["metadata_filter"].get("vendor")
        target_ids = q.get("target_chunk_ids") or []

        n_correct = 0
        for cid in target_ids:
            predicted_vendors = transformer_by_chunk.get(cid, {}).get("all_vendors", [])
            if any(fuzzy_contains(vendor, pv) for pv in predicted_vendors):
                n_correct += 1

        record = {
            "query_id": qid,
            "query_text": q["query_text"],
            "vendor_filter": vendor,
            "gt_top10": gt_ids,
            "transformer_top10": tf_ids,
            "vendor_cluster_size": len(target_ids),
            "transformer_correctly_extracted_vendor_for_n_of_cluster": n_correct,
            "transformer_cluster_extraction_accuracy": n_correct / len(target_ids) if target_ids else None,
        }
        (identical if gt_ids == tf_ids else differing).append(record)

    return {
        "description": "Of the 14 metadata-filtered queries, how many produced byte-identical "
                        "top-10 results between Ground Truth and Transformer metadata (Section 14's "
                        "'12/14 identical' finding) — cross-referenced against the Transformer's ACTUAL "
                        "per-chunk vendor extraction accuracy for that specific vendor's full chunk "
                        "cluster (not a sparse illustrative sample), to test the 'easy vendor' "
                        "hypothesis directly and precisely.",
        "n_identical": len(identical),
        "n_differing": len(differing),
        "identical_examples": identical,
        "differing_examples": differing,
    }


# --- Category 4: Reranker cluster-breaking ---

def _cluster_breaking_examples(chunks, queries, logs, baseline_key, reranked_key):
    baseline_log = logs[baseline_key]
    reranked_log = logs[reranked_key]

    examples = []
    for qid, q in queries.items():
        target_ids = q.get("target_chunk_ids")
        if not target_ids or not isinstance(target_ids, list) or not q.get("metadata_filter"):
            continue
        base_rec, rerank_rec = baseline_log.get(qid), reranked_log.get(qid)
        if not base_rec or not rerank_rec:
            continue

        target_set = set(target_ids)
        base_recovered = target_set & set(top_ids(base_rec))
        rerank_recovered = target_set & set(top_ids(rerank_rec))

        if len(rerank_recovered) < len(base_recovered):
            demoted = base_recovered - rerank_recovered
            # what took their place in the reranked top-10 instead?
            intruders = [cid for cid in top_ids(rerank_rec) if cid not in target_set][:3]

            # Spot-check found a real confound here: some "intruders" turned
            # out to be genuine same-vendor receipts whose ground_truth.vendor
            # string just lacks a suffix (e.g. a registration number) present
            # on other receipts from the same real business — build_eval_queries.py
            # groups vendors by EXACT string match, so these get excluded from
            # target_ids even though they're the same vendor. Flag these
            # explicitly rather than let them look like clean reranker errors.
            vendor_filter = q["metadata_filter"].get("vendor", "")
            likely_mislabeled_intruders = []
            for cid in intruders:
                intruder_vendor = chunks[cid]["ground_truth"].get("vendor") or ""
                shorter, longer = sorted([vendor_filter.strip().lower(), intruder_vendor.strip().lower()], key=len)
                if shorter and shorter in longer:
                    likely_mislabeled_intruders.append(cid)

            examples.append({
                "query_id": qid,
                "query_text": q["query_text"],
                "vendor_filter": vendor_filter,
                "target_set_size": len(target_set),
                "recovered_before_rerank": len(base_recovered),
                "recovered_after_rerank": len(rerank_recovered),
                "demoted_chunk_ids": list(demoted),
                "demoted_snippets": [snippet(chunks, c) for c in list(demoted)[:2]],
                "intruder_chunk_ids_now_in_top10": intruders,
                "intruder_snippets": [snippet(chunks, c) for c in intruders],
                "intruder_vendors": [chunks[c]["ground_truth"].get("vendor") for c in intruders],
                "likely_mislabeled_intruders": likely_mislabeled_intruders,
                "evidence": (f"{len(demoted)} genuine same-vendor receipt(s) dropped out of the "
                             f"top-10 after reranking. Of the {len(intruders)} replacements shown, "
                             f"{len(likely_mislabeled_intruders)} are themselves fuzzy-matches for the "
                             f"same vendor ('{vendor_filter}') under a slightly different exact string "
                             f"(e.g. missing a registration-number suffix) — this is a real gap in "
                             f"build_eval_queries.py's EXACT-string vendor grouping (Step 4), not "
                             f"purely a reranker error; the true regression is only the remaining "
                             f"{len(intruders) - len(likely_mislabeled_intruders)} genuinely "
                             f"different-vendor intruder(s)."),
            })

    return examples


def category_reranker_cluster_breaking(chunks, queries, logs):
    # Spot-check found the wrong pair was compared here initially: comparing
    # the METADATA-FILTERED baseline/reranked pair showed almost no
    # regression (only 1/14 queries, and even that one turned out to be
    # mostly a labeling-gap artifact, not a real reranker error) — because
    # once a metadata filter has already narrowed the pool to the correct
    # vendor, the reranker has nothing off-vendor left to promote. The
    # genuine Section 15 regression (76.0%->64.4% recall) lives in the
    # UNFILTERED pair (hybrid vs hybrid_reranked), where the reranker sees a
    # generic candidate pool with no vendor constraint at all. Comparing
    # both pairs side by side surfaces a real, valuable mechanistic finding:
    # metadata filtering appears to PROTECT reranking from cluster-breaking,
    # not just add a separate, unrelated benefit.
    unfiltered_examples = _cluster_breaking_examples(
        chunks, queries, logs, "hybrid__none", "hybrid_reranked__none"
    )
    filtered_examples = _cluster_breaking_examples(
        chunks, queries, logs, "metadata_hybrid__ground_truth", "metadata_hybrid_reranked__ground_truth"
    )

    return {
        "description": "Vendor-cluster queries where reranking recovered FEWER correct cluster "
                        "members in the top-10 than the un-reranked baseline — compared for BOTH "
                        "the unfiltered pair (hybrid vs hybrid_reranked, no metadata constraint) "
                        "and the metadata-filtered pair (metadata_hybrid vs metadata_hybrid_reranked, "
                        "candidates pre-narrowed to the correct vendor).",
        "key_finding": (
            f"{len(unfiltered_examples)} genuine regressions found WITHOUT metadata filtering "
            f"(the real Section 15 small-set drop, 76.0%->64.4% aggregate recall) vs. only "
            f"{len(filtered_examples)} WITH metadata filtering applied first (and that one is "
            f"mostly a vendor-string labeling-gap artifact, not a real reranker error — see its "
            f"'likely_mislabeled_intruders' field). This suggests metadata filtering PROTECTS "
            f"reranking from cluster-breaking by removing off-vendor candidates before the "
            f"cross-encoder ever sees them, rather than being an independent, unrelated benefit."
        ),
        "n_regressed_unfiltered": len(unfiltered_examples),
        "unfiltered_examples": unfiltered_examples,
        "n_regressed_metadata_filtered": len(filtered_examples),
        "metadata_filtered_examples": filtered_examples,
    }


# --- Category 5: Semantic-paraphrase (type D) qualitative review ---

def category_semantic_paraphrase(chunks, queries, logs):
    d_queries = {qid: q for qid, q in queries.items() if q["question_type"] == "D"}
    compare_systems = ["vector_only__none", "bm25_only__none", "hybrid__none", "hybrid_reranked__none"]

    results = []
    for qid, q in d_queries.items():
        entry = {"query_id": qid, "query_text": q["query_text"], "by_system": {}}
        for key in compare_systems:
            rec = logs[key].get(qid)
            if not rec:
                continue
            top3 = rec["retrieved_chunks"][:3]
            entry["by_system"][key] = [
                {"chunk_id": c["chunk_id"], "source": chunks[c["chunk_id"]]["source"],
                 "snippet": snippet(chunks, c["chunk_id"])}
                for c in top3
            ]
        results.append(entry)

    return {
        "description": "The 5 semantic-paraphrase (type D) queries have no single correct target "
                        "by design (see build_eval_queries.py) — qualitative top-3 comparison across "
                        "vector/BM25/hybrid/hybrid_reranked, for manual read of which systems stay "
                        "topically on-target vs. get pulled off by literal keyword overlap.",
        "queries": results,
    }


# --- Category 6: Negation/absence (type E) structural cap ---

def category_negation_absence_cap(chunks, queries, logs, reports):
    e_queries = {qid: q for qid, q in queries.items() if q["question_type"] == "E"}
    results = []
    for qid, q in e_queries.items():
        target_ids = q.get("target_chunk_ids")
        entry = {
            "query_id": qid,
            "query_text": q["query_text"],
            "target_set_size": len(target_ids) if target_ids else 0,
            "structural_cap_recall_at_10": round(10 / len(target_ids), 4) if target_ids else None,
            "by_system": {},
        }
        for system, metadata_source in ALL_SYSTEMS:
            key = f"{system}__{metadata_source}"
            rec = logs[key].get(qid)
            if not rec or not target_ids:
                continue
            recovered = set(target_ids) & set(top_ids(rec, k=10))
            entry["by_system"][key] = {
                "n_recovered": len(recovered),
                "recall_at_10": len(recovered) / len(target_ids),
            }
        results.append(entry)

    return {
        "description": "Negation/absence queries target the full ~1000-chunk CORD set (e.g. 'no "
                        "vendor listed') — Recall@10 is structurally capped at 10/1000=1% for even "
                        "a perfect retriever. Documented here as confirmed, not re-investigated "
                        "(Section 4d already established this is a metric-scale property, not a bug).",
        "queries": results,
    }


# --- Addition A: per-source breakdown ---

def source_breakdown(chunks, queries, logs):
    breakdown = {}
    for system, metadata_source in ALL_SYSTEMS:
        key = f"{system}__{metadata_source}"
        log = logs[key]
        per_source = {s: {"n": 0, "hit": 0} for s in SOURCES}
        for qid, q in queries.items():
            target = q.get("target_chunk_id")
            if not target or qid not in log:
                continue
            src = chunks[target]["source"]
            per_source[src]["n"] += 1
            if target in top_ids(log[qid], k=10):
                per_source[src]["hit"] += 1
        breakdown[key] = {
            src: {
                "n": v["n"],
                "hit_at_10": v["hit"] / v["n"] if v["n"] else None,
            } for src, v in per_source.items()
        }
    return breakdown


# --- Addition B: per-field metadata coverage ---

def field_coverage_analysis(queries, reports):
    field_usage = {"vendor": 0, "date": 0, "amount": 0}
    filter_field_usage = {"vendor": 0, "date": 0, "amount": 0}

    for q in queries.values():
        qt = q["query_text"].lower()
        if "vendor" in qt or "store or business" in qt or q["metadata_filter"].get("vendor"):
            field_usage["vendor"] += 1
        if "date" in qt:
            field_usage["date"] += 1
        if "amount" in qt or "total" in qt:
            field_usage["amount"] += 1
        for f in filter_field_usage:
            if q.get("metadata_filter", {}).get(f):
                filter_field_usage[f] += 1

    fuzzy_f1 = reports["extraction"]["fuzzy"]
    coverage = {}
    for field in ["vendor", "date", "amount"]:
        coverage[field] = {
            "extraction_fuzzy_f1": fuzzy_f1[field]["f1"],
            "n_queries_referencing_field": field_usage[field],
            "n_queries_using_as_metadata_filter": filter_field_usage[field],
            "gap_note": (
                f"NO query in the current set uses '{field}' as a metadata_filter — "
                f"the metadata-quality linkage table's Recall@10 numbers (Section 14) "
                f"only ever reflect vendor-filter quality, never date or amount, "
                f"regardless of this field's own extraction F1."
            ) if filter_field_usage[field] == 0 else None,
        }

    return {
        "description": "Per-field metadata coverage in the current eval query set, cross-referenced "
                        "against Step 3c's extraction F1 — explicitly flagging fields with zero "
                        "metadata-filter coverage, since the linkage table can only be informative "
                        "about fields the query set actually exercises as filters.",
        "fields": coverage,
        "key_finding": (
            "AMOUNT has the WORST extraction F1 (61.3% fuzzy) of the three fields, but NO query "
            "in the current set filters by amount — meaning the metadata-quality linkage table "
            "(Section 14) cannot show whether amount-extraction errors erode retrieval at all. "
            "The 'no separation between GT and Transformer' finding is specifically a vendor-filter "
            "finding, not a general one, and should not be read as 'metadata quality doesn't matter' "
            "for the worst-performing field."
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/eval/error_analysis_report.json")
    ap.add_argument("--write", action="store_true",
                     help="Write the full report to --out. Without this flag, only prints "
                          "to stdout for spot-check review.")
    args = ap.parse_args()

    chunks, queries, logs, reports = load_all()

    report = {
        "1_bm25_keyword_collisions": category_bm25_collisions(chunks, queries, logs),
        "2_retrieval_depth_misses": category_retrieval_depth_misses(chunks, queries, logs),
        "3_metadata_quality_masking": category_metadata_masking(chunks, queries, logs, reports),
        "4_reranker_cluster_breaking": category_reranker_cluster_breaking(chunks, queries, logs),
        "5_semantic_paraphrase_review": category_semantic_paraphrase(chunks, queries, logs),
        "6_negation_absence_cap": category_negation_absence_cap(chunks, queries, logs, reports),
        "source_breakdown": source_breakdown(chunks, queries, logs),
        "field_coverage": field_coverage_analysis(queries, reports),
    }

    print(json.dumps(report, indent=2)[:6000])
    print("... (truncated stdout preview — see --write output for full report)")

    if args.write:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote {args.out}")
    else:
        print("\n(Spot-check mode — pass --write to save data/eval/error_analysis_report.json)")


if __name__ == "__main__":
    main()
