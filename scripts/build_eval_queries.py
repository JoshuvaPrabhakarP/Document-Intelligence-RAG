"""
Step 4 (prep): Build the evaluation query set for the core 5x2 matrix.

Every query needs a KNOWN ground-truth target so Recall@10/MRR can be
scored objectively, without a human in the loop:
  - Most queries have a single `target_chunk_id` (the chunk the question
    was generated from).
  - Metadata-targeted queries (type B) and the negation/absence hand-written
    queries (type E) have a `target_chunk_ids` LIST instead — the correct
    answer is a whole set of chunks (e.g. "every Walmart receipt"), not one.

Question taxonomy (confirmed against the project's actual scope):
  A - single-document factual field lookup
  B - metadata-filtered lookup (target = a set, has a metadata_filter)
  C - multi-document aggregation/comparison
  D - semantic/conceptual paraphrase (no lexical overlap with the target)
  E - negation/absence

CORD chunks never have vendor/date (by design, see chunk_corpus.py) — so
vendor/date queries only ever sample from wildreceipt/sroie. Amount queries
sample from all three sources.

Output: data/eval/queries.jsonl, one record per line:
  {query_id, query_text, question_type, target_chunk_id OR target_chunk_ids,
   metadata_filter, ground_truth_answer, ragas_subset}

Usage:
    python scripts/build_eval_queries.py
"""

import argparse
import json
import random
import re


def normalize(s):
    return re.sub(r"\s+", " ", s.strip().lower()) if s else s


def fuzzy_contains(a, b):
    a, b = normalize(a), normalize(b)
    if not a or not b:
        return False
    shorter, longer = sorted([a, b], key=len)
    return len(shorter) >= 2 and shorter in longer


def load_chunks(path):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def build_templated_queries(chunks, rng, n_per_category=10):
    """Factual field-lookup + metadata-targeted queries, all with a real,
    verifiable ground-truth target drawn directly from the corpus."""
    queries = []
    qi = [0]

    def next_id():
        qi[0] += 1
        return f"tmpl_{qi[0]:04d}"

    by_vendor_date = [c for c in chunks if c["ground_truth"].get("vendor") and c["source"] != "cord"]
    by_amount = [c for c in chunks if c["ground_truth"].get("amount")]
    by_date = [c for c in chunks if c["ground_truth"].get("date") and c["source"] != "cord"]

    # An anchor field is only a fair test if it UNIQUELY identifies one chunk
    # — e.g. "the receipt from SANYU" is ambiguous across SANYU's 49 receipts.
    # Verified necessary by spot-checking the first draft of this script:
    # 5/12 amount-anchored queries collided on a shared amount (e.g. '2.00'
    # appeared in 9 different chunks), and vendor-only anchors are ambiguous
    # for every high-volume vendor. Build a value->chunk_ids index per field
    # (or field-combo) and only keep chunks where the anchor is unique.
    def unique_index(chunk_list, key_fn):
        idx = {}
        for c in chunk_list:
            k = key_fn(c)
            if k is not None:
                idx.setdefault(k, []).append(c["chunk_id"])
        return idx

    amount_idx = unique_index(chunks, lambda c: c["ground_truth"].get("amount"))
    vendor_amount_idx = unique_index(
        by_vendor_date,
        lambda c: (c["ground_truth"]["vendor"], c["ground_truth"].get("amount")),
    )
    vendor_date_idx = unique_index(
        by_vendor_date,
        lambda c: (c["ground_truth"]["vendor"], c["ground_truth"].get("date")),
    )

    # --- Type A: vendor lookup, anchored on a uniquely-occurring amount ---
    # (pasting the receipt's own raw text into the query would make
    # retrieval trivial by definition — an independent, non-target field is
    # a fair test; it must also be verified unique, or "the" target is a
    # coin flip among several equally-valid receipts.)
    unique_amount_chunks = [c for c in by_vendor_date
                             if c["ground_truth"].get("amount")
                             and len(amount_idx.get(c["ground_truth"]["amount"], [])) == 1]
    for c in rng.sample(unique_amount_chunks, min(n_per_category, len(unique_amount_chunks))):
        vendor = c["ground_truth"]["vendor"]
        amount = c["ground_truth"]["amount"]
        queries.append({
            "query_id": next_id(),
            "query_text": f"Which store or business issued a receipt for a total of {amount}?",
            "question_type": "A",
            "target_chunk_id": c["chunk_id"],
            "metadata_filter": {},
            "ground_truth_answer": vendor,
        })

    # --- Type A: date lookup, anchored on a uniquely-occurring (vendor, amount) pair ---
    unique_vendor_amount_chunks = [c for c in by_vendor_date
                                    if c["ground_truth"].get("amount")
                                    and len(vendor_amount_idx.get(
                                        (c["ground_truth"]["vendor"], c["ground_truth"].get("amount")), [])) == 1]
    for c in rng.sample(unique_vendor_amount_chunks, min(n_per_category, len(unique_vendor_amount_chunks))):
        date = c["ground_truth"]["date"]
        vendor = c["ground_truth"]["vendor"]
        amount = c["ground_truth"]["amount"]
        queries.append({
            "query_id": next_id(),
            "query_text": f"What date is on the receipt from {vendor} for a total of {amount}?",
            "question_type": "A",
            "target_chunk_id": c["chunk_id"],
            "metadata_filter": {},
            "ground_truth_answer": date,
        })

    # --- Type A: amount lookup, anchored on a uniquely-occurring (vendor, date)
    # pair for WildReceipt/SROIE, or doc_id (inherently unique) for CORD ---
    unique_vendor_date_chunks = [c for c in by_vendor_date
                                  if c["ground_truth"].get("date")
                                  and len(vendor_date_idx.get(
                                      (c["ground_truth"]["vendor"], c["ground_truth"].get("date")), [])) == 1]
    cord_amount_chunks = [c for c in chunks if c["source"] == "cord" and c["ground_truth"].get("amount")]
    amount_lookup_pool = unique_vendor_date_chunks + cord_amount_chunks
    for c in rng.sample(amount_lookup_pool, min(n_per_category, len(amount_lookup_pool))):
        amount = c["ground_truth"]["amount"]
        vendor = c["ground_truth"].get("vendor")
        date = c["ground_truth"].get("date")
        phrase = f"from {vendor} dated {date}" if vendor else f"(doc_id {c['doc_id']})"
        queries.append({
            "query_id": next_id(),
            "query_text": f"What is the total amount on the receipt {phrase}?",
            "question_type": "A",
            "target_chunk_id": c["chunk_id"],
            "metadata_filter": {},
            "ground_truth_answer": amount,
        })

    # --- Type B: metadata-targeted lookup (target = a SET of chunks) ---
    vendor_groups = {}
    for c in by_vendor_date:
        v = c["ground_truth"]["vendor"].strip()
        vendor_groups.setdefault(v, []).append(c["chunk_id"])
    # only vendors with a meaningful number of receipts make a sensible "show me all X" query
    common_vendors = [v for v, ids in vendor_groups.items() if len(ids) >= 5]
    for vendor in rng.sample(common_vendors, min(n_per_category, len(common_vendors))):
        ids = vendor_groups[vendor]
        queries.append({
            "query_id": next_id(),
            "query_text": f"Show me receipts from {vendor}.",
            "question_type": "B",
            "target_chunk_ids": ids,
            "metadata_filter": {"vendor": vendor},
            "ground_truth_answer": vendor,
        })

    return queries


def build_handwritten_queries():
    """D (semantic paraphrase), E (negation/absence), C (comparison) —
    the categories a template can't produce naturally. Targets are still
    concrete and verifiable, just picked by hand against real corpus facts
    established during Step 2/3 (e.g. the "refueling a car" case already
    surfaced a genuine RQ3 finding)."""
    return [
        # --- D: semantic / conceptual paraphrase, no lexical overlap ---
        {"query_id": "hw_d001", "query_text": "receipt for refueling a car",
         "question_type": "D", "target_chunk_ids": None,  # scored qualitatively — see harness note
         "metadata_filter": {}, "ground_truth_answer": None},
        {"query_id": "hw_d002", "query_text": "a purchase of bread or bakery items",
         "question_type": "D", "target_chunk_ids": None,
         "metadata_filter": {}, "ground_truth_answer": None},
        {"query_id": "hw_d003", "query_text": "money spent on office supplies",
         "question_type": "D", "target_chunk_ids": None,
         "metadata_filter": {}, "ground_truth_answer": None},
        {"query_id": "hw_d004", "query_text": "a meal purchased at a restaurant",
         "question_type": "D", "target_chunk_ids": None,
         "metadata_filter": {}, "ground_truth_answer": None},
        {"query_id": "hw_d005", "query_text": "hardware or tools bought from a shop",
         "question_type": "D", "target_chunk_ids": None,
         "metadata_filter": {}, "ground_truth_answer": None},

        # --- E: negation / absence (genuinely hard — no retriever can search
        # for a MISSING field natively; CORD chunks are the real ground truth
        # for "no vendor listed" since that field is absent there by design) ---
        {"query_id": "hw_e001", "query_text": "receipts that have no vendor or business name listed",
         "question_type": "E", "target_chunk_ids": "__CORD_ALL__",
         "metadata_filter": {}, "ground_truth_answer": None},
        {"query_id": "hw_e002", "query_text": "receipts with no date printed on them",
         "question_type": "E", "target_chunk_ids": "__CORD_ALL__",
         "metadata_filter": {}, "ground_truth_answer": None},

        # --- C: multi-document aggregation / comparison ---
        {"query_id": "hw_c001", "query_text": "the Walmart receipt with the highest total amount",
         "question_type": "C", "target_chunk_ids": "__VENDOR_MAX_AMOUNT__:WALMART",
         "metadata_filter": {"vendor": "WALMART"}, "ground_truth_answer": None},
        {"query_id": "hw_c002", "query_text": "the cheapest receipt from Target",
         "question_type": "C", "target_chunk_ids": "__VENDOR_MIN_AMOUNT__:TARGET",
         "metadata_filter": {"vendor": "TARGET"}, "ground_truth_answer": None},
    ]


def resolve_special_targets(queries, chunks):
    """Resolve the __CORD_ALL__ / __VENDOR_MAX_AMOUNT__ placeholders from
    build_handwritten_queries() against the real corpus, so every query
    ships with a concrete, checkable target — not a placeholder string."""
    cord_ids = [c["chunk_id"] for c in chunks if c["source"] == "cord"]

    def parse_amount(a):
        try:
            return float(re.sub(r"[^0-9.]", "", a))
        except (TypeError, ValueError):
            return None

    for q in queries:
        t = q.get("target_chunk_ids")
        if t == "__CORD_ALL__":
            q["target_chunk_ids"] = cord_ids
        elif isinstance(t, str) and t.startswith("__VENDOR_MAX_AMOUNT__:"):
            vendor = t.split(":", 1)[1]
            candidates = [c for c in chunks if fuzzy_contains(vendor, c["ground_truth"].get("vendor"))
                          and parse_amount(c["ground_truth"].get("amount")) is not None]
            if candidates:
                best = max(candidates, key=lambda c: parse_amount(c["ground_truth"]["amount"]))
                q["target_chunk_ids"] = [best["chunk_id"]]
                q["ground_truth_answer"] = best["ground_truth"]["amount"]
            else:
                q["target_chunk_ids"] = []
        elif isinstance(t, str) and t.startswith("__VENDOR_MIN_AMOUNT__:"):
            vendor = t.split(":", 1)[1]
            # exclude zero/negative amounts — spot-checking the first draft
            # surfaced a "$0.00 cheapest Target receipt" as the min, which is
            # a void/comp-line extraction artifact, not a genuine purchase.
            candidates = [c for c in chunks if fuzzy_contains(vendor, c["ground_truth"].get("vendor"))
                          and (parse_amount(c["ground_truth"].get("amount")) or 0) > 0]
            if candidates:
                best = min(candidates, key=lambda c: parse_amount(c["ground_truth"]["amount"]))
                q["target_chunk_ids"] = [best["chunk_id"]]
                q["ground_truth_answer"] = best["ground_truth"]["amount"]
            else:
                q["target_chunk_ids"] = []
    return queries


def assign_ragas_subset(queries, rng, target_size=18):
    """Pick a stratified subset (by question_type) that every system-run
    will use for the LLM-judged Ragas metrics — the SAME subset across all
    7 runs, so scores stay comparable."""
    for q in queries:
        q["ragas_subset"] = False

    by_type = {}
    for q in queries:
        by_type.setdefault(q["question_type"], []).append(q)

    n_types = len(by_type)
    per_type = max(1, target_size // n_types)
    for t, qs in by_type.items():
        for q in rng.sample(qs, min(per_type, len(qs))):
            q["ragas_subset"] = True

    return queries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/chunks/chunks.jsonl")
    ap.add_argument("--out", default="data/eval/queries.jsonl")
    ap.add_argument("--n-per-category", type=int, default=12)
    ap.add_argument("--ragas-subset-size", type=int, default=18)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    chunks = load_chunks(args.chunks)
    if not chunks:
        print(f"ERROR: no chunks loaded from {args.chunks}")
        return

    queries = build_templated_queries(chunks, rng, n_per_category=args.n_per_category)
    queries += build_handwritten_queries()
    queries = resolve_special_targets(queries, chunks)
    queries = assign_ragas_subset(queries, rng, target_size=args.ragas_subset_size)

    with open(args.out, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    by_type = {}
    for q in queries:
        by_type[q["question_type"]] = by_type.get(q["question_type"], 0) + 1

    print(f"Wrote {len(queries)} queries -> {args.out}")
    print(f"By question_type: {by_type}")
    print(f"In Ragas subset: {sum(1 for q in queries if q['ragas_subset'])}")


if __name__ == "__main__":
    main()
