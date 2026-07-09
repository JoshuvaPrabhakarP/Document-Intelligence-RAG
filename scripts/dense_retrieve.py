"""
Dense (vector) retrieval — query function against the FAISS index built by
embed_index.py. Separated out as its own module (rather than left as an
inline sanity check) so fusion.py and the eval harness can both import and
call `search()` directly.

Usage as a library:
    from dense_retrieve import DenseRetriever
    retriever = DenseRetriever(index_dir="data/index")
    results = retriever.search("What is the total amount?", k=10)

Usage as a CLI (sanity check):
    python scripts/dense_retrieve.py --query "What is the total amount?"
"""

import argparse
import json
import os

import numpy as np


class DenseRetriever:
    def __init__(self, index_dir="data/index"):
        import faiss

        config_path = os.path.join(index_dir, "index_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        index_path = os.path.join(index_dir, "faiss.index")
        self.index = faiss.read_index(index_path)

        metadata_path = os.path.join(index_dir, "metadata.jsonl")
        self.metadata = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.metadata.append(json.loads(line))

        if len(self.metadata) != self.index.ntotal:
            raise ValueError(
                f"metadata.jsonl has {len(self.metadata)} entries but the "
                f"FAISS index has {self.index.ntotal} vectors — they must "
                f"stay in sync (rebuild both together, never one alone)."
            )

        self._model = None  # lazy-loaded, avoids loading the model if unused

    def _get_model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = SentenceTransformer(self.config["model"], device=device)
        return self._model

    def search(self, query, k=10):
        model = self._get_model()
        prefixed_query = self.config["query_prefix"] + query
        q_emb = model.encode(
            [prefixed_query], normalize_embeddings=True, convert_to_numpy=True
        )
        scores, indices = self.index.search(q_emb.astype(np.float32), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS pads with -1 if k > index size
                continue
            m = self.metadata[idx]
            results.append({
                "chunk_id": m["chunk_id"],
                "score": float(score),
                "text": m["text"],
                "source": m["source"],
                "ground_truth": m["ground_truth"],
            })
        return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default="data/index")
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    retriever = DenseRetriever(index_dir=args.index_dir)
    results = retriever.search(args.query, k=args.k)

    print(f"Query: {args.query!r}\n")
    for rank, r in enumerate(results):
        print(f"#{rank+1} score={r['score']:.3f} chunk_id={r['chunk_id']} "
              f"source={r['source']}")
        print(f"    {r['text'][:100]!r}")


if __name__ == "__main__":
    main()
