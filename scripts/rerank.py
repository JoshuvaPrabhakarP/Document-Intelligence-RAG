"""
Step 5: Reranker ablation — the reranker itself.

Wraps BAAI/bge-reranker-large (a cross-encoder — scores query and passage
jointly, unlike the bi-encoder used for retrieval). This is a pure
post-retrieval re-scoring stage: it takes candidates an existing,
unmodified retriever already found and re-orders them, it does not retrieve
anything itself.

Already decided in Step 0 (PROJECT_GRAPH_REPORT.md Section 6: "Runs
bge-reranker-large — Step 5 only") and already installed — no new
dependency.

Loads the model directly via `transformers` (AutoTokenizer +
AutoModelForSequenceClassification) rather than FlagEmbedding's own
FlagReranker wrapper — found during this step's own pilot verification
that FlagEmbedding 1.4.0 (the latest release; no newer version exists)
calls `tokenizer.prepare_for_model()`, a method removed entirely in the
installed transformers 5.13.0 (`AttributeError: XLMRobertaTokenizer has no
attribute prepare_for_model`). Downgrading transformers to chase
FlagEmbedding's older expected API risked destabilizing the whole working
stack (sentence-transformers/embeddings all currently depend on 5.13.0) —
loading the same model directly via `transformers`'s standard, still-fully-
supported sequence-classification interface is the documented, official
alternative usage pattern for BGE reranker models and avoids the broken
code path entirely.

Usage:
    from rerank import Reranker
    reranker = Reranker()
    reranked, latency_ms = reranker.rerank(query, candidates)  # candidates: list of dicts with "text"
"""

import time


class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-large"):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

    def rerank(self, query, candidates, batch_size=32, max_length=512):
        """
        candidates: list of dicts (as returned by any retriever's search()),
        each with at least a "text" key. Order on input is the PRE-rerank
        order (i.e. whatever the base retriever produced).

        Returns a NEW list, re-sorted by rerank score descending. Each dict
        is annotated with:
          - rerank_score: the cross-encoder's relevance score (raw logit —
            not a probability, but monotonic and sufficient for ranking)
          - pre_rerank_rank: 1-indexed position in the INPUT list (so rank
            movement can be computed later without needing the original
            list kept around separately)

        Also returns the total rerank latency in ms (the cross-encoder
        forward pass only — retrieval time is measured separately by the
        caller).
        """
        import torch

        if not candidates:
            return [], 0.0

        pairs = [(query, c["text"]) for c in candidates]

        t0 = time.monotonic()
        scores = []
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i + batch_size]
                inputs = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**inputs).logits.view(-1).float()
                scores.extend(logits.cpu().tolist())
        latency_ms = (time.monotonic() - t0) * 1000

        annotated = []
        for pre_rank, (c, score) in enumerate(zip(candidates, scores), start=1):
            c = dict(c)
            c["rerank_score"] = float(score)
            c["pre_rerank_rank"] = pre_rank
            annotated.append(c)

        annotated.sort(key=lambda c: c["rerank_score"], reverse=True)
        return annotated, latency_ms


if __name__ == "__main__":
    # Self-test: rerank the known car-wash-vs-fuel-station case from Step 2
    # (Section 12 of PROJECT_GRAPH_REPORT.md) — a real, inspectable example
    # where BM25 got fooled by a literal "Car#304" transaction-ID token
    # matching the query's word "car", not just "did it run without error."
    import json

    chunks = {}
    with open("data/chunks/chunks.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunks[c["chunk_id"]] = c

    import sys
    sys.path.insert(0, "scripts")
    from fusion import HybridRetriever

    query = "receipt for refueling a car"
    print(f"Query: {query!r}\n")

    print("Loading hybrid retriever...")
    hybrid = HybridRetriever()
    pre_rerank = hybrid.search(query, k=10, pool_size=50)

    print("--- Pre-rerank (hybrid RRF order) ---")
    for i, c in enumerate(pre_rerank, start=1):
        print(f"  #{i} {c['chunk_id']} ({c['source']}): {c['text'][:70]!r}")

    print("\nLoading reranker (first run downloads bge-reranker-large)...")
    reranker = Reranker()
    reranked, latency_ms = reranker.rerank(query, pre_rerank)

    print(f"\n--- Post-rerank (cross-encoder order, {latency_ms:.1f}ms) ---")
    for i, c in enumerate(reranked, start=1):
        moved = c["pre_rerank_rank"] - i
        print(f"  #{i} (was #{c['pre_rerank_rank']}, moved {moved:+d}) "
              f"score={c['rerank_score']:.3f} {c['chunk_id']} ({c['source']}): "
              f"{c['text'][:70]!r}")

    order_changed = [c["chunk_id"] for c in pre_rerank] != [c["chunk_id"] for c in reranked]
    print(f"\nOrder changed: {order_changed}")
    assert order_changed or len(pre_rerank) <= 1, (
        "Reranker produced the exact same order as the base retriever — "
        "either it isn't doing anything, or this happens to be a case "
        "where the cross-encoder agrees with RRF. Inspect manually."
    )
    print("Self-test passed: reranker loaded, scored, and reordered candidates.")
