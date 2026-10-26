"""
Fast, deterministic API tests — the pipeline's `.answer()` is mocked so
these never need a GPU, a loaded model, or Ollama running. Safe for CI.

A separate, explicitly-marked real-model smoke test lives in
tests/test_integration.py (skipped automatically if Ollama isn't reachable).
"""

from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready_reports_structure():
    resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert "ready" in body
    assert "checks" in body
    assert "index_files_present" in body["checks"]
    assert "generation_provider_reachable" in body["checks"]


def test_ready_checks_generation_provider_via_generic_openai_endpoint_not_ollama_specific_one():
    """Milestone 5 deployment-prep finding: /ready used to call Ollama's
    own proprietary /api/tags, which only Ollama implements -- against a
    hosted OpenAI-compatible provider (e.g. Together.ai) this always
    reported generation_provider_reachable=False even while generation
    itself worked, which would have kept a Cloud Run readiness probe from
    ever passing. Must hit {base_url}/models -- part of the OpenAI-
    compatible surface itself, not a provider-specific path -- so this
    works unchanged for ANY configured provider, not hardcoded to one."""
    captured = {}

    def fake_get(url, headers=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        return httpx.Response(status_code=200, request=httpx.Request("GET", url))

    with patch("app.main.httpx.get", side_effect=fake_get):
        resp = client.get("/ready")

    assert resp.status_code == 200
    assert resp.json()["checks"]["generation_provider_reachable"] is True
    assert captured["url"].endswith("/models")
    assert "/api/tags" not in captured["url"]
    assert "Authorization" in captured["headers"]


def test_index_page_loads():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_sample_questions_endpoint():
    resp = client.get("/sample-questions")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["questions"], list)
    assert len(body["questions"]) > 0


@pytest.mark.parametrize("payload", [
    {"question": ""},                       # too short
    {"question": "a" * 501},                # too long
    {"top_k": 3},                            # missing required field
    {"question": "valid?", "top_k": 0},      # top_k below minimum
    {"question": "valid?", "top_k": 11},     # top_k above maximum
])
def test_query_validation_rejects_bad_input(payload):
    resp = client.post("/query", json=payload)
    assert resp.status_code == 422


def test_query_returns_grounded_answer_with_sources():
    fake_result = {
        "answer": "The total is $42.17.",
        "sources": [
            {"chunk_id": "demo_000", "source": "synthetic_demo", "score": 1.2, "excerpt": "EXAMPLE GROCERS..."},
        ],
        "has_sufficient_evidence": True,
        "retrieval_latency_ms": 12.3,
        "rerank_latency_ms": 45.6,
        "generation_latency_ms": 789.0,
    }
    with patch("app.main.pipeline.answer", return_value=fake_result):
        resp = client.post("/query", json={"question": "What is the total?", "top_k": 3, "workflow": "standard"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "The total is $42.17."
    assert body["has_sufficient_evidence"] is True
    assert len(body["sources"]) == 1
    assert body["sources"][0]["chunk_id"] == "demo_000"
    assert body["config"] == "demo"


def test_query_insufficient_evidence_is_flagged():
    fake_result = {
        "answer": "I don't know.",
        "sources": [],
        "has_sufficient_evidence": False,
        "retrieval_latency_ms": 10.0,
        "rerank_latency_ms": None,
        "generation_latency_ms": 500.0,
    }
    with patch("app.main.pipeline.answer", return_value=fake_result):
        resp = client.post("/query", json={"question": "What is the capital of France?", "workflow": "standard"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["has_sufficient_evidence"] is False
    assert body["sources"] == []


def test_query_no_matching_documents_returns_empty_sources():
    fake_result = {
        "answer": "I don't have any relevant documents to answer that question.",
        "sources": [],
        "has_sufficient_evidence": False,
        "retrieval_latency_ms": 5.0,
        "rerank_latency_ms": None,
        "generation_latency_ms": None,
    }
    with patch("app.main.pipeline.answer", return_value=fake_result):
        resp = client.post("/query", json={"question": "anything", "workflow": "standard"})

    assert resp.status_code == 200
    assert resp.json()["generation_latency_ms"] is None


def test_generation_provider_failure_returns_clean_error_not_stack_trace():
    fake_result = {
        "answer": None,
        "sources": [{"chunk_id": "demo_000", "source": "synthetic_demo", "score": 1.0, "excerpt": "..."}],
        "has_sufficient_evidence": None,
        "retrieval_latency_ms": 10.0,
        "rerank_latency_ms": None,
        "generation_latency_ms": None,
        "error": "Ollama generation failed: connection refused",
    }
    with patch("app.main.pipeline.answer", return_value=fake_result):
        resp = client.post("/query", json={"question": "anything", "workflow": "standard"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["error"] == "Ollama generation failed: connection refused"
    assert body["answer"] is None


def test_unhandled_pipeline_exception_returns_503_not_stack_trace():
    with patch("app.main.pipeline.answer", side_effect=RuntimeError("disk full")):
        resp = client.post("/query", json={"question": "anything", "workflow": "standard"})

    assert resp.status_code == 503
    body = resp.json()
    assert "disk full" not in body["detail"]  # internal detail never leaks to the client
    assert "temporarily unavailable" in body["detail"]
