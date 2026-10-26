"""
Tests for the upload feature. Extraction/indexing run for REAL (they're
local and fast — no network, no GPU required for text-layer PDFs); the
generation step is mocked (same boundary as test_api.py) so these don't
need Ollama running and stay fast/deterministic for CI.

A PDF fixture is generated on the fly with PyMuPDF (the same library the
app uses to read PDFs) rather than committing a binary test fixture file.
"""

import io

import pymupdf
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, sessions

client = TestClient(app)


def make_test_pdf(text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    return doc.tobytes()


def make_test_image() -> bytes:
    img = Image.new("RGB", (100, 100), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _clean_sessions():
    """Every test starts and ends with no leftover sessions on disk —
    directly exercises the same cleanup path 'Clear uploaded receipts' uses."""
    yield
    for sid in list(sessions._sessions.keys()):
        sessions.clear_session(sid)


def test_upload_pdf_extracts_embedded_text():
    pdf_bytes = make_test_pdf("UNIT TEST CAFE\nDate: 01/01/2026\nTOTAL 7.50")
    resp = client.post("/upload", files={"file": ("receipt.pdf", pdf_bytes, "application/pdf")})

    assert resp.status_code == 200
    body = resp.json()
    assert body["page_count"] == 1
    assert body["chunks_indexed"] == 1
    assert "UNIT TEST CAFE" in body["pages"][0]["text"]
    assert body["pages"][0]["used_ocr"] is False


def test_upload_rejects_unsupported_file_type():
    resp = client.post("/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["detail"]


def test_upload_rejects_oversized_file():
    big_content = b"\x00" * (11 * 1024 * 1024)  # 11MB > 10MB default limit
    resp = client.post("/upload", files={"file": ("big.png", big_content, "image/png")})
    assert resp.status_code == 413


def test_upload_rejects_empty_file():
    resp = client.post("/upload", files={"file": ("empty.png", b"", "image/png")})
    assert resp.status_code == 400


def test_upload_rejects_corrupt_pdf():
    resp = client.post("/upload", files={"file": ("fake.pdf", b"not a real pdf", "application/pdf")})
    assert resp.status_code == 400
    assert "Could not open this file as a PDF" in resp.json()["detail"]


def test_query_against_uploaded_receipt_returns_cited_answer():
    pdf_bytes = make_test_pdf("UNIQUE VENDOR XYZ\nDate: 03/03/2026\nTOTAL 99.99")
    session = requests_session_with_cookies()

    upload_resp = session.post("/upload", files={"file": ("vendor_xyz.pdf", pdf_bytes, "application/pdf")})
    assert upload_resp.status_code == 200

    fake_answer = {
        "answer": "The total is $99.99.",
        "sources": [{"chunk_id": "x", "source": "vendor_xyz.pdf", "score": 1.0, "excerpt": "..."}],
        "has_sufficient_evidence": True,
        "retrieval_latency_ms": 10.0,
        "rerank_latency_ms": 20.0,
        "generation_latency_ms": 500.0,
    }
    from unittest.mock import patch
    with patch("app.main.pipeline.answer_from_retriever", return_value=fake_answer):
        query_resp = session.post("/query", json={"question": "What is the total?", "mode": "upload", "workflow": "standard"})

    assert query_resp.status_code == 200
    body = query_resp.json()
    assert body["answer"] == "The total is $99.99."
    assert body["sources"][0]["source"] == "vendor_xyz.pdf"
    assert body["config"] == "upload"


def test_upload_mode_query_without_any_upload_returns_clear_error():
    resp = client.post("/query", json={"question": "anything", "mode": "upload"})
    assert resp.status_code == 400
    assert "No receipts uploaded" in resp.json()["detail"]


def test_session_isolation_different_sessions_cannot_see_each_others_uploads():
    pdf_bytes = make_test_pdf("SESSION A ONLY VENDOR\nTOTAL 11.11")

    session_a = requests_session_with_cookies()
    upload_resp = session_a.post("/upload", files={"file": ("a.pdf", pdf_bytes, "application/pdf")})
    assert upload_resp.status_code == 200

    # Session B (fresh client, no shared cookies) queries upload mode —
    # must NOT see session A's file at all.
    session_b = requests_session_with_cookies()
    resp_b = session_b.post("/query", json={"question": "What vendor is this?", "mode": "upload"})
    assert resp_b.status_code == 400  # session B has uploaded nothing


def test_clear_session_removes_files_and_resets_listing():
    pdf_bytes = make_test_pdf("TO BE CLEARED\nTOTAL 1.00")
    session = requests_session_with_cookies()

    session.post("/upload", files={"file": ("clear_me.pdf", pdf_bytes, "application/pdf")})
    before = session.get("/session").json()
    assert len(before["files"]) == 1

    delete_resp = session.delete("/session")
    assert delete_resp.json()["cleared"] is True

    after = session.get("/session").json()
    assert len(after["files"]) == 0
    assert after["session_id"] != before["session_id"]  # a fresh session, not the cleared one


def requests_session_with_cookies():
    """TestClient instances share no state by default; each call to this
    helper gets its own cookie jar, simulating a genuinely separate browser
    session — the actual mechanism the isolation guarantee depends on."""
    return TestClient(app)
