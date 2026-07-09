"""
Step 1c: Structured logging schema.

Defines the log record every query run writes, regardless of which system
produced it (vector-only, BM25, hybrid, metadata+vector, metadata+hybrid,
with or without reranking). This exists BEFORE any retrieval code, per the
build order — logging wired in from day 1 is what makes Step 6's error
analysis possible without re-running the whole 5x2 matrix later.

One JSONL file per system-run (e.g. logs/vector_only__none.jsonl,
logs/metadata_hybrid__transformer.jsonl) — naming convention encodes the
system and metadata source so files are self-describing without opening
them.

Usage:
    from logging_schema import QueryLogger

    logger = QueryLogger(system="vector_only", metadata_source="none")
    logger.log_query(
        query_id="q001",
        query_text="What is the total on the Walmart receipt?",
        question_type="A",  # single-document
        retrieved_chunks=[
            {"chunk_id": "chunk_000123", "score": 0.81, "text": "..."},
            ...
        ],
        metadata_filters_applied=None,
        reranker_scores=None,
        generated_answer="The total is $89.09.",
        ground_truth_answer="$89.09",
        retrieval_latency_ms=42.3,
        generation_latency_ms=850.1,
    )
"""

import json
import os
import time
from datetime import datetime, timezone

SCHEMA_VERSION = "1.0"

# Valid values — kept as constants so callers don't invent new ones by typo
VALID_SYSTEMS = {
    "vector_only",
    "bm25_only",
    "hybrid",
    "metadata_vector",
    "metadata_hybrid",
}
VALID_METADATA_SOURCES = {"none", "ground_truth", "transformer"}
VALID_QUESTION_TYPES = {"A", "B", "C", "D", "E"}  # per scope doc taxonomy


def _validate(system, metadata_source, question_type):
    if system not in VALID_SYSTEMS:
        raise ValueError(f"Unknown system {system!r}. Must be one of {VALID_SYSTEMS}")
    if metadata_source not in VALID_METADATA_SOURCES:
        raise ValueError(
            f"Unknown metadata_source {metadata_source!r}. "
            f"Must be one of {VALID_METADATA_SOURCES}"
        )
    if question_type is not None and question_type not in VALID_QUESTION_TYPES:
        raise ValueError(
            f"Unknown question_type {question_type!r}. "
            f"Must be one of {VALID_QUESTION_TYPES} or None"
        )


class QueryLogger:
    """
    One instance per system-run. Writes one JSONL line per query to
    logs/{system}__{metadata_source}.jsonl, appending — safe to reuse
    across multiple sessions running the same config without overwriting
    prior queries (call reset() explicitly if you want a clean slate).
    """

    def __init__(self, system, metadata_source, log_dir="logs"):
        _validate(system, metadata_source, None)
        self.system = system
        self.metadata_source = metadata_source
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, f"{system}__{metadata_source}.jsonl")

    def reset(self):
        """Explicitly clear this run's log file. Not called automatically —
        accidental data loss from re-running a script shouldn't be silent."""
        if os.path.exists(self.path):
            os.remove(self.path)

    def log_query(
        self,
        query_id,
        query_text,
        retrieved_chunks,
        question_type=None,
        metadata_filters_applied=None,
        reranker_scores=None,
        generated_answer=None,
        ground_truth_answer=None,
        retrieval_latency_ms=None,
        generation_latency_ms=None,
        extra=None,
    ):
        """
        retrieved_chunks: list of {"chunk_id": str, "score": float, "text": str}
            — score meaning depends on `system` (cosine sim for vector_only,
            BM25 score for bm25_only, fused score for hybrid, etc.) — the
            log doesn't normalize across systems, comparisons that need
            normalized scores should handle that at analysis time, not
            silently rescale here and hide what actually happened.
        metadata_filters_applied: dict or None, e.g. {"vendor": "Walmart"}
        reranker_scores: list of float aligned with retrieved_chunks, or
            None if this run didn't use a reranker
        extra: dict for anything system-specific that doesn't fit the
            standard fields (e.g. fusion weights for hybrid) — avoids
            forcing every system into an identical schema when they
            genuinely differ, while keeping the core fields consistent
        """
        _validate(self.system, self.metadata_source, question_type)

        if reranker_scores is not None and len(reranker_scores) != len(retrieved_chunks):
            raise ValueError(
                f"reranker_scores length ({len(reranker_scores)}) must match "
                f"retrieved_chunks length ({len(retrieved_chunks)})"
            )

        record = {
            "schema_version": SCHEMA_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system": self.system,
            "metadata_source": self.metadata_source,
            "query_id": query_id,
            "query_text": query_text,
            "question_type": question_type,
            "retrieved_chunks": retrieved_chunks,
            "metadata_filters_applied": metadata_filters_applied,
            "reranker_scores": reranker_scores,
            "generated_answer": generated_answer,
            "ground_truth_answer": ground_truth_answer,
            "retrieval_latency_ms": retrieval_latency_ms,
            "generation_latency_ms": generation_latency_ms,
            "extra": extra or {},
        }

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return record


def read_log(path):
    """Read a log file back into a list of dicts — used by the eval
    harness (Step 4) and error analysis (Step 6), never by retrieval
    code itself."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


if __name__ == "__main__":
    # Self-test: write a record, read it back, verify round-trip. This is
    # foundational infrastructure every later step depends on — worth
    # verifying immediately rather than discovering a bug three steps in.
    import tempfile

    test_dir = tempfile.mkdtemp()
    logger = QueryLogger(system="vector_only", metadata_source="none", log_dir=test_dir)
    logger.reset()

    written = logger.log_query(
        query_id="test_q001",
        query_text="What is the total amount?",
        question_type="A",
        retrieved_chunks=[
            {"chunk_id": "chunk_000668", "score": 0.675, "text": "sample text"},
        ],
        generated_answer="$89.09",
        ground_truth_answer="$89.09",
        retrieval_latency_ms=12.4,
    )

    read_back = read_log(logger.path)
    assert len(read_back) == 1, f"Expected 1 record, got {len(read_back)}"
    assert read_back[0]["query_id"] == "test_q001"
    assert read_back[0]["retrieved_chunks"][0]["chunk_id"] == "chunk_000668"
    assert read_back[0]["schema_version"] == SCHEMA_VERSION

    print("Self-test passed.")
    print(f"Sample record:\n{json.dumps(written, indent=2)}")
