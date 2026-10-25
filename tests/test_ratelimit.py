"""
Deterministic tests for the practical deployment limits added in
Milestone 5 (app/ratelimit.py + their wiring in app/main.py). All fast,
all mocked at the pipeline level -- no real model calls.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app, query_rate_limiter, upload_rate_limiter
from app.ratelimit import ConcurrencyLimiter, RateLimiter

FAKE_RESULT = {
    "answer": "x", "sources": [], "has_sufficient_evidence": True,
    "retrieval_latency_ms": 1.0, "rerank_latency_ms": None, "generation_latency_ms": 1.0,
}


def test_rate_limiter_allows_up_to_the_configured_count_then_blocks():
    limiter = RateLimiter(max_per_minute=3)
    assert limiter.allow("1.2.3.4") is True
    assert limiter.allow("1.2.3.4") is True
    assert limiter.allow("1.2.3.4") is True
    assert limiter.allow("1.2.3.4") is False


def test_rate_limiter_tracks_each_key_independently():
    limiter = RateLimiter(max_per_minute=1)
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is False
    assert limiter.allow("client-b") is True  # a different client is unaffected


def test_concurrency_limiter_blocks_beyond_the_configured_count():
    limiter = ConcurrencyLimiter(max_concurrent=2)
    assert limiter.try_acquire() is True
    assert limiter.try_acquire() is True
    assert limiter.try_acquire() is False
    limiter.release()
    assert limiter.try_acquire() is True  # a released slot becomes available again


def test_query_endpoint_returns_429_after_the_per_ip_limit_is_exceeded():
    client = TestClient(app)
    with patch("app.main.pipeline.answer", return_value=FAKE_RESULT):
        statuses = [
            client.post("/query", json={"question": "x?", "mode": "demo", "workflow": "standard"}).status_code
            for _ in range(query_rate_limiter.max_per_minute + 1)
        ]
    assert statuses[:-1] == [200] * query_rate_limiter.max_per_minute
    assert statuses[-1] == 429


def test_upload_endpoint_returns_429_after_the_per_ip_limit_is_exceeded():
    client = TestClient(app)
    # Deliberately sends an unsupported file type -- the rate limiter check
    # runs BEFORE file-type validation, so this still exercises the limit
    # without needing a real PDF/image per call, and a 400 (bad type) vs. a
    # 429 (rate limited) are easy to tell apart.
    statuses = [
        client.post("/upload", files={"file": ("x.txt", b"hi", "text/plain")}).status_code
        for _ in range(upload_rate_limiter.max_per_minute + 1)
    ]
    assert statuses[:-1] == [400] * upload_rate_limiter.max_per_minute
    assert statuses[-1] == 429


def test_upload_session_file_count_limit_is_enforced():
    from app.config import get_settings
    from app.upload.session import UploadedFile, sessions

    client = TestClient(app)
    resp = client.get("/session")  # establishes a session cookie
    session_id = client.cookies.get("session_id")
    limit = get_settings().upload_max_files_per_session

    for i in range(limit):
        sessions.add_file(session_id, UploadedFile(
            file_id=f"f{i}", original_filename=f"f{i}.pdf", saved_path="/tmp/x",
            content_type="application/pdf", page_count=1, extracted_text="x", chunk_ids=[f"f{i}_p1"],
        ))

    resp = client.post("/upload", files={"file": ("one_too_many.pdf", b"%PDF-1.4 fake", "application/pdf")})
    client.delete("/session")

    assert resp.status_code == 429
    assert "limit" in resp.json()["detail"].lower()
