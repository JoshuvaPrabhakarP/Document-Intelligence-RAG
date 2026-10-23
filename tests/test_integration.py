"""
Real end-to-end smoke test — actually loads the embedding model, builds the
FAISS/BM25 indexes, reranks, and calls Ollama for real. No mocks. This is
slow (pays the cold-start cost measured in PROGRESS.md) and requires a GPU
or patience, plus Ollama running locally with qwen3:8b pulled.

Skipped automatically (not failed) if Ollama isn't reachable, so CI without
a local Ollama instance doesn't break on this file — the fast, mocked tests
in test_api.py are what CI actually gates on.
"""

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app

settings = get_settings()


def _ollama_available() -> bool:
    try:
        resp = httpx.get(settings.ollama_base_url.replace("/v1", "") + "/api/tags", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ollama_available(), reason="Ollama not reachable at the configured base URL — skipping real-model smoke test."
)


def test_real_end_to_end_query_against_demo_corpus():
    client = TestClient(app)
    resp = client.post(
        "/query",
        json={"question": "What is the total amount on the Example Grocers Inc. receipt?", "top_k": 3},
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["error"] is None
    assert body["answer"] is not None
    assert "42.17" in body["answer"]
    assert body["has_sufficient_evidence"] is True
    assert len(body["sources"]) > 0
    assert body["sources"][0]["chunk_id"] == "demo_000"
    assert body["retrieval_latency_ms"] > 0
    assert body["generation_latency_ms"] > 0


def test_real_insufficient_evidence_on_off_topic_question():
    client = TestClient(app)
    resp = client.post("/query", json={"question": "What is the capital of France?", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["has_sufficient_evidence"] is False
