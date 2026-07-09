"""
Step 2a: BM25 (lexical) retrieval.

Builds a BM25 index over the same 3,712 chunks used for dense retrieval,
and provides a matching search() interface so fusion.py can call both
retrievers identically.

Usage as a library:
    from bm25_retrieve import BM25Retriever
    retriever = BM25Retriever(chunks_path="data/chunks/chunks.jsonl")
    results = retriever.search("What is the total amount?", k=10)

Usage as a CLI (builds+caches the index, then runs a sanity-check query):
    python scripts/bm25_retrieve.py --query "What is the total amount?"
"""

import argparse
import json
import os
import pickle
import re


# Standard English stopwords. BM25 (unlike dense embeddings) treats every
# query token as an equally real term with no notion of semantic importance
# — without removing these, a document containing "is"/"the"/"on" as
# incidental natural-language filler can out-rank a document that actually
# matches the query's real content words, purely by sharing function words.
# Confirmed via a real failure case: a boilerplate disclaimer sentence
# ("This is not a receipt...") out-scored an actual Walmart receipt on the
# query "What is the total amount on the Walmart receipt?" because the
# disclaimer's natural-language phrasing shared more stopwords with the
# query than the real receipt (which has no reason to contain "is"/"on").
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was",
    "were", "will", "with", "what", "when", "where", "who", "which", "why",
    "how", "this", "these", "those", "i", "you", "your", "do", "does",
    "did", "have", "had", "can", "could", "would", "should", "not", "no",
}


def tokenize(text):
    """Lowercase word tokenizer with stopword removal. BM25 needs this
    (dense embeddings don't) — see STOPWORDS comment above for why."""
    words = re.findall(r"\w+", text.lower())
    return [w for w in words if w not in STOPWORDS]


class BM25Retriever:
    def __init__(self, chunks_path="data/chunks/chunks.jsonl", cache_dir="data/index"):
        self.chunks_path = chunks_path
        self.cache_path = os.path.join(cache_dir, "bm25_index.pkl")
        self.chunks = self._load_chunks()
        self.bm25 = self._load_or_build_index()

    def _load_chunks(self):
        chunks = []
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        return chunks

    def _load_or_build_index(self):
        from rank_bm25 import BM25Okapi

        # cache invalidation: rebuild if the cache is older than chunks.jsonl,
        # not just "does a cache file exist" — a stale index that silently
        # doesn't match the current corpus would be a hard bug to trace.
        if os.path.exists(self.cache_path) and (
            os.path.getmtime(self.cache_path) > os.path.getmtime(self.chunks_path)
        ):
            with open(self.cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached["num_chunks"] == len(self.chunks):
                return cached["bm25"]
            # count mismatch despite a "fresh" timestamp — don't trust it
            print("BM25 cache chunk count mismatch, rebuilding...")

        print("Building BM25 index (not cached, or cache stale)...")
        tokenized = [tokenize(c["text"]) for c in self.chunks]
        bm25 = BM25Okapi(tokenized)

        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump({"bm25": bm25, "num_chunks": len(self.chunks)}, f)

        return bm25

    def search(self, query, k=10):
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = scores.argsort()[::-1][:k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue  # BM25 score of 0 means no term overlap at all — not a real match
            c = self.chunks[idx]
            results.append({
                "chunk_id": c["chunk_id"],
                "score": float(scores[idx]),
                "text": c["text"],
                "source": c["source"],
                "ground_truth": c["ground_truth"],
            })
        return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/chunks/chunks.jsonl")
    ap.add_argument("--cache-dir", default="data/index")
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    retriever = BM25Retriever(chunks_path=args.chunks, cache_dir=args.cache_dir)
    results = retriever.search(args.query, k=args.k)

    print(f"\nQuery: {args.query!r}\n")
    if not results:
        print("  (no results — query terms don't overlap with any chunk)")
    for rank, r in enumerate(results):
        print(f"#{rank+1} score={r['score']:.3f} chunk_id={r['chunk_id']} "
              f"source={r['source']}")
        print(f"    {r['text'][:100]!r}")


if __name__ == "__main__":
    main()