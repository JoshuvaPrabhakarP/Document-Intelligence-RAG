"""
Step 3c: Per-field extraction accuracy — Transformer predictions vs. ground
truth.

This is the first half of the metadata-quality linkage table (Section 8 of
the project report): the upstream extraction accuracy that Step 4's eval
matrix will later correlate against downstream Recall@10, to answer RQ2
(how much does imperfect metadata erode the benefit of structure?).

Field-level comparison, not exact-string comparison — "WALMART" predicted
against "Walmart Inc." should count as a partial match, not a hard miss,
since downstream metadata filtering typically normalizes/fuzzy-matches
anyway. Reports both exact-match and fuzzy-match (normalized, substring-
tolerant) precision/recall so the difference between the two is visible,
not hidden behind one number.

Usage:
    python scripts/eval_metadata_extraction.py \
        --chunks data/chunks/chunks.jsonl \
        --predictions data/metadata/transformer_extracted.jsonl \
        --out data/metadata/extraction_accuracy_report.json
"""

import argparse
import json
import re


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalize(s):
    if s is None:
        return None
    return re.sub(r"\s+", " ", s.strip().lower())


def fuzzy_equal(a, b):
    """Tolerant match: exact after normalization, or one is a substring of
    the other (handles cases like predicted 'SAFEWAY' vs GT 'SAFEWAY STORE
    #123', which are the same real-world entity)."""
    a_norm, b_norm = normalize(a), normalize(b)
    if a_norm is None or b_norm is None:
        return a_norm == b_norm  # both None counts as a match (both absent)
    if a_norm == b_norm:
        return True
    shorter, longer = sorted([a_norm, b_norm], key=len)
    return len(shorter) >= 3 and shorter in longer


def score_field_set(gt_lists, pred_lists, match_fn):
    """
    Set-based scoring: for each document, compare the SET of ground-truth
    values against the SET of predicted values for one field — not a
    single joined string. A receipt commonly has multiple real AMOUNT
    spans (subtotal, tax, total); comparing "the first GT value" against
    "all predicted values joined into one string" (the original, wrong
    approach) systematically fails even when the model correctly found
    the right value among several.

    - TP: a GT value that has a matching predicted value (recall-side),
      counted separately from a predicted value that has a matching GT
      value (precision-side) — these can differ when list lengths differ.
    - FP: a predicted value with no matching GT value
    - FN: a GT value with no matching predicted value
    - TN: both lists empty
    """
    tp_recall = fp = fn = tn = 0
    tp_precision = 0

    for gt_vals, pred_vals in zip(gt_lists, pred_lists):
        gt_vals = gt_vals or []
        pred_vals = pred_vals or []

        if not gt_vals and not pred_vals:
            tn += 1
            continue

        matched_gt = set()
        matched_pred = set()
        for gi, gv in enumerate(gt_vals):
            for pi, pv in enumerate(pred_vals):
                if match_fn(gv, pv):
                    matched_gt.add(gi)
                    matched_pred.add(pi)

        tp_recall += len(matched_gt)
        fn += len(gt_vals) - len(matched_gt)
        tp_precision += len(matched_pred)
        fp += len(pred_vals) - len(matched_pred)

    precision = tp_precision / (tp_precision + fp) if (tp_precision + fp) > 0 else None
    recall = tp_recall / (tp_recall + fn) if (tp_recall + fn) > 0 else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    return {
        "tp_precision_side": tp_precision, "tp_recall_side": tp_recall,
        "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/chunks/chunks.jsonl")
    ap.add_argument("--predictions", default="data/metadata/transformer_extracted.jsonl")
    ap.add_argument("--out", default="data/metadata/extraction_accuracy_report.json")
    ap.add_argument("--mismatch-samples", type=int, default=5)
    args = ap.parse_args()

    print(f"Loading ground truth from {args.chunks}...")
    chunks = load_jsonl(args.chunks)
    gt_by_id = {c["chunk_id"]: c["ground_truth"] for c in chunks}
    print(f"  {len(chunks)} chunks")

    print(f"Loading predictions from {args.predictions}...")
    predictions = load_jsonl(args.predictions)
    pred_by_id = {p["chunk_id"]: p["predicted"] for p in predictions}
    print(f"  {len(predictions)} predictions")

    common_ids = sorted(set(gt_by_id.keys()) & set(pred_by_id.keys()))
    if len(common_ids) != len(chunks):
        missing = len(chunks) - len(common_ids)
        print(f"  WARNING: {missing} chunks have no matching prediction "
              f"(chunk_id mismatch or incomplete extraction run)")
    print(f"  {len(common_ids)} chunks with both GT and prediction available\n")

    fields = ["vendor", "date", "amount"]
    list_field_names = {"vendor": "all_vendors", "date": "all_dates", "amount": "all_amounts"}
    report = {"exact": {}, "fuzzy": {}, "mismatch_samples": {}}

    for match_name, match_fn in [("exact", lambda a, b: normalize(a) == normalize(b)),
                                   ("fuzzy", fuzzy_equal)]:
        print(f"--- {match_name.upper()} match scoring (set-based, multi-value aware) ---")
        for field in fields:
            list_key = list_field_names[field]
            gt_lists = [gt_by_id[cid].get(list_key, []) for cid in common_ids]
            pred_lists = [pred_by_id[cid].get(list_key, []) for cid in common_ids]
            result = score_field_set(gt_lists, pred_lists, match_fn)
            report[match_name][field] = result

            p = f"{result['precision']:.3f}" if result['precision'] is not None else "N/A"
            r = f"{result['recall']:.3f}" if result['recall'] is not None else "N/A"
            f1 = f"{result['f1']:.3f}" if result['f1'] is not None else "N/A"
            print(f"  {field:8s} precision={p} recall={r} f1={f1} "
                  f"(fp={result['fp']} fn={result['fn']} tn={result['tn']})")
        print()

    # Collect a few concrete mismatch examples per field (fuzzy match basis)
    # for spot-checking — same discipline as every prior verification step:
    # read actual disagreements, don't just trust the aggregate numbers.
    for field in fields:
        list_key = list_field_names[field]
        samples = []
        for cid in common_ids:
            gt_vals = gt_by_id[cid].get(list_key, []) or []
            pred_vals = pred_by_id[cid].get(list_key, []) or []

            unmatched_gt = [gv for gv in gt_vals if not any(fuzzy_equal(gv, pv) for pv in pred_vals)]
            unmatched_pred = [pv for pv in pred_vals if not any(fuzzy_equal(gv, pv) for gv in gt_vals)]

            if unmatched_gt or unmatched_pred:
                samples.append({
                    "chunk_id": cid,
                    "ground_truth": gt_vals,
                    "predicted": pred_vals,
                })
            if len(samples) >= args.mismatch_samples:
                break
        report["mismatch_samples"][field] = samples

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Wrote full report -> {args.out}")

    print("\n--- Sample mismatches (fuzzy basis, set comparison) ---")
    for field in fields:
        print(f"\n{field.upper()}:")
        for s in report["mismatch_samples"][field]:
            print(f"  {s['chunk_id']}: GT={s['ground_truth']!r} vs "
                  f"Predicted={s['predicted']!r}")


if __name__ == "__main__":
    main()
