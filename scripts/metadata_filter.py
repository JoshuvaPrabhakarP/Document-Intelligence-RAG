"""
Step 3d: Metadata-aware retrieval.

Wraps the existing dense/hybrid retrievers with a metadata filter, drawing
from either the ground-truth source (chunks.jsonl) or the Transformer-
extracted source (transformer_extracted.jsonl) — this is what makes the
5x2 experimental matrix's "Metadata + Vector" and "Metadata + Hybrid" rows
real, runnable systems, in both metadata-source variants.

Approach: retrieve-then-filter. The FAISS index is a flat (non-filterable)
index, so rather than rebuilding it as a filterable structure at this
scale (3,712 docs), pull a wider candidate pool from the base retriever,
then filter that pool down to chunks matching the metadata criteria. This
is a standard, well-understood approach at this corpus size — pre-filtering
the vector index itself only pays off at much larger scale.

Usage as a library:
    from metadata_filter import MetadataRetriever
    retriever = MetadataRetriever(metadata_source="ground_truth")
    results = retriever.search(
        query="total amount",
        filters={"vendor": "Walmart"},
        base="hybrid",  # or "vector"
        k=10,
    )

Usage as a CLI (side-by-side comparison across both metadata sources):
    python scripts/metadata_filter.py \
        --query "total amount" --vendor "Walmart" --base hybrid
"""

import argparse
import json
import re


def normalize(s):
    if s is None:
        return None
    return re.sub(r"\s+", " ", s.strip().lower())


def fuzzy_contains(filter_value, candidate_value):
    """Case-insensitive substring match, either direction — same tolerant
    matching convention as eval_metadata_extraction.py's fuzzy_equal, so
    filtering behavior is consistent with how extraction accuracy itself
    was measured."""
    f_norm, c_norm = normalize(filter_value), normalize(candidate_value)
    if f_norm is None or c_norm is None:
        return False
    shorter, longer = sorted([f_norm, c_norm], key=len)
    return len(shorter) >= 2 and shorter in longer


class MetadataStore:
    """
    Loads one metadata source (ground_truth or transformer) and exposes
    per-chunk field values for filtering.
    """

    def __init__(self, metadata_source, chunks_path="data/chunks/chunks.jsonl",
                 predictions_path="data/metadata/transformer_extracted.jsonl"):
        if metadata_source not in ("ground_truth", "transformer"):
            raise ValueError(f"metadata_source must be 'ground_truth' or 'transformer', "
                              f"got {metadata_source!r}")
        self.metadata_source = metadata_source
        self.by_chunk_id = {}

        if metadata_source == "ground_truth":
            with open(chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    c = json.loads(line)
                    self.by_chunk_id[c["chunk_id"]] = c["ground_truth"]
        else:
            with open(predictions_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    p = json.loads(line)
                    self.by_chunk_id[p["chunk_id"]] = p["predicted"]

    def matches(self, chunk_id, filters):
        """
        filters: dict like {"vendor": "Walmart", "date": "12/22/17"} —
        ALL provided filters must match (AND semantics) against the
        chunk's all_<field> list for that metadata source. A chunk with
        no value for a filtered field (e.g. every CORD chunk has no
        vendor/date) correctly never matches a filter on that field.
        """
        meta = self.by_chunk_id.get(chunk_id)
        if meta is None:
            return False

        field_to_list_key = {"vendor": "all_vendors", "date": "all_dates", "amount": "all_amounts"}
        for field, filter_value in filters.items():
            list_key = field_to_list_key.get(field)
            if list_key is None:
                raise ValueError(f"Unknown filter field {field!r}, must be one of "
                                  f"{list(field_to_list_key)}")
            candidates = meta.get(list_key, []) or []
            if not any(fuzzy_contains(filter_value, c) for c in candidates):
                return False
        return True


class MetadataRetriever:
    def __init__(self, metadata_source, index_dir="data/index",
                 chunks_path="data/chunks/chunks.jsonl",
                 predictions_path="data/metadata/transformer_extracted.jsonl"):
        from dense_retrieve import DenseRetriever
        from fusion import HybridRetriever

        self.dense = DenseRetriever(index_dir=index_dir)
        self.hybrid = HybridRetriever(index_dir=index_dir, chunks_path=chunks_path)
        self.store = MetadataStore(metadata_source, chunks_path=chunks_path,
                                    predictions_path=predictions_path)

    def search(self, query, filters, base="hybrid", k=10, pool_size=100):
        """
        base: "vector" or "hybrid" — which underlying retriever supplies
        the candidate pool before filtering.
        pool_size: how many candidates to pull BEFORE filtering — needs to
        be generous, since a tight metadata filter can eliminate most of
        the pool. If fewer than k results survive filtering, that's a
        real, informative outcome (not a bug) — it means the filter is
        more restrictive than the pool was deep enough to satisfy, which
        is itself useful signal for Step 4's evaluation.
        """
        if base == "vector":
            candidates = self.dense.search(query, k=pool_size)
        elif base == "hybrid":
            candidates = self.hybrid.search(query, k=pool_size, pool_size=pool_size)
        else:
            raise ValueError(f"base must be 'vector' or 'hybrid', got {base!r}")

        filtered = [c for c in candidates if self.store.matches(c["chunk_id"], filters)]
        return filtered[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--vendor", default=None)
    ap.add_argument("--date", default=None)
    ap.add_argument("--amount", default=None)
    ap.add_argument("--base", default="hybrid", choices=["vector", "hybrid"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--pool-size", type=int, default=100)
    args = ap.parse_args()

    filters = {}
    if args.vendor:
        filters["vendor"] = args.vendor
    if args.date:
        filters["date"] = args.date
    if args.amount:
        filters["amount"] = args.amount

    if not filters:
        print("WARNING: no filters given — this just runs unfiltered "
              "retrieval, add --vendor/--date/--amount to actually test filtering.\n")

    print(f"Query: {args.query!r}  |  Filters: {filters}  |  Base: {args.base}")
    print("Loading retrievers for both metadata sources...\n")

    for source in ["ground_truth", "transformer"]:
        print(f"{'='*70}\nMetadata source: {source}\n{'='*70}")
        retriever = MetadataRetriever(metadata_source=source)
        results = retriever.search(args.query, filters, base=args.base,
                                    k=args.k, pool_size=args.pool_size)
        if not results:
            print("  (no results survived filtering)")
        for rank, r in enumerate(results):
            gt = r.get("ground_truth", {})
            print(f"  #{rank+1} score={r['score']:.3f} {r['chunk_id']} ({r['source']})")
            print(f"      text: {r['text'][:80]!r}")


if __name__ == "__main__":
    main()
