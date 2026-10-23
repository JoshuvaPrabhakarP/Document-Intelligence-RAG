"""
FastAPI application — the thin HTTP layer around the existing, unmodified
research pipeline (app/rag_pipeline.py) plus the upload feature (app/upload/).
No retrieval/rerank/generation logic lives in this file; it only handles
HTTP concerns: validation, status codes, readiness checks, session cookies,
and serving the static web interface.
"""

import logging
import os
import time
import uuid

import httpx
from fastapi import FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.rag_pipeline import RAGPipeline
from app.schemas import (
    HealthResponse,
    PageExtraction,
    QueryRequest,
    QueryResponse,
    ReadinessResponse,
    SessionFileSummary,
    SessionResponse,
    UploadResponse,
)
from app.upload import retriever_cache
from app.upload.extraction import ExtractionError, OCRUnavailable, ProcessingTimeout, extract_pages
from app.upload.indexing import append_chunks, rebuild_index
from app.upload.session import SessionManager, UploadedFile, new_session_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

app = FastAPI(title="Document Intelligence RAG — Demo API")

settings = get_settings()
pipeline = RAGPipeline(settings)
sessions = SessionManager(base_dir=settings.upload_dir, ttl_minutes=settings.upload_session_ttl_minutes)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

DEMO_SAMPLE_QUESTIONS = [
    "What is the total amount on the Example Grocers Inc. receipt?",
    "Which vendor issued a receipt for $156.42?",
    "What date is on the Sample Coffee Co. receipt?",
    "What is the total at Sample Gas Station?",
]

ALLOWED_CONTENT_TYPES = {"application/pdf", "image/jpeg", "image/png"}
SESSION_COOKIE = "session_id"


def _get_session_id(request: Request, response: Response) -> str:
    """Server-generated opaque ID only — the client never supplies or
    influences this value. Every upload/session/query-in-upload-mode path
    goes through this, which is what makes cross-session isolation
    actually hold (there is no way for a request to name another
    session's ID and succeed, since IDs are UUIDs and sessions are looked
    up by exact match against server-side state)."""
    session_id = request.cookies.get(SESSION_COOKIE)
    if not session_id or not sessions.is_known(session_id):
        session_id = new_session_id()
        sessions.touch_or_create(session_id)
        response.set_cookie(SESSION_COOKIE, session_id, httponly=True, samesite="lax",
                             max_age=settings.upload_session_ttl_minutes * 60)
    return session_id


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadinessResponse)
def ready():
    checks = {}
    checks["index_files_present"] = os.path.exists(
        os.path.join(settings.index_dir, "faiss.index")
    ) and os.path.exists(settings.chunks_path)
    try:
        resp = httpx.get(settings.ollama_base_url.replace("/v1", "") + "/api/tags", timeout=3.0)
        checks["generation_provider_reachable"] = resp.status_code == 200
    except Exception:
        checks["generation_provider_reachable"] = False

    all_ready = all(checks.values())
    detail = None if all_ready else "See `checks` for which dependency is unavailable."
    return ReadinessResponse(ready=all_ready, config=settings.app_config, checks=checks, detail=detail)


@app.get("/sample-questions")
def sample_questions():
    return {"config": settings.app_config, "questions": DEMO_SAMPLE_QUESTIONS}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, http_request: Request, response: Response):
    start = time.monotonic()
    try:
        if request.mode == "upload":
            session_id = _get_session_id(http_request, response)
            files = sessions.list_files(session_id)
            if not files:
                raise HTTPException(
                    status_code=400,
                    detail="No receipts uploaded in this session yet. Upload a file first, "
                           "or switch to demo mode.",
                )
            retriever = retriever_cache.get_retriever(
                session_id, str(sessions.index_dir(session_id)), str(sessions.chunks_path(session_id))
            )
            result = pipeline.answer_from_retriever(
                request.question, retriever, top_k=request.top_k, use_reranker=request.use_reranker
            )
        elif request.mode == "demo":
            result = pipeline.answer(
                request.question, top_k=request.top_k, use_reranker=request.use_reranker
            )
        else:
            raise HTTPException(status_code=422, detail=f"Unknown mode {request.mode!r}. Use 'demo' or 'upload'.")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error answering query (mode=%s)", request.mode)
        raise HTTPException(
            status_code=503,
            detail="The retrieval/answer pipeline is temporarily unavailable. Please try again.",
        ) from exc

    elapsed_ms = (time.monotonic() - start) * 1000
    # Deliberately logs only metadata (mode, latency, evidence flag) — never
    # the question text or any retrieved/extracted content, per the
    # "keep uploads and extracted content out of ... application logs" requirement.
    logger.info("query answered in %.0fms (mode=%s, evidence=%s)",
                elapsed_ms, request.mode, result.get("has_sufficient_evidence"))

    return QueryResponse(config=request.mode, **result)


@app.post("/upload", response_model=UploadResponse)
def upload(file: UploadFile, http_request: Request, response: Response):
    session_id = _get_session_id(http_request, response)

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {file.content_type!r}. Allowed: PDF, JPG, PNG.",
        )

    content = file.file.read()
    max_bytes = settings.upload_max_file_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File is {len(content) / 1e6:.1f}MB, which exceeds the "
                   f"{settings.upload_max_file_mb}MB limit.",
        )
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        pages = extract_pages(
            content, file.content_type,
            max_pages=settings.upload_max_pdf_pages,
            timeout_s=settings.upload_processing_timeout_s,
        )
    except ProcessingTimeout as exc:
        raise HTTPException(status_code=408, detail=str(exc)) from exc
    except OCRUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ExtractionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Safe, server-generated filename — the original name is kept ONLY as
    # display/citation metadata, never used to construct a disk path, so
    # there's no path-traversal or overwrite risk from a crafted filename.
    file_id = str(uuid.uuid4())
    ext = {"application/pdf": ".pdf", "image/jpeg": ".jpg", "image/png": ".png"}[file.content_type]
    saved_path = sessions.files_dir(session_id) / f"{file_id}{ext}"
    saved_path.parent.mkdir(parents=True, exist_ok=True)
    saved_path.write_bytes(content)

    # One chunk per page — gives page-level citation for multi-page PDFs.
    # "source" carries the ORIGINAL filename (reusing the existing schema
    # field the UI already displays) and "ground_truth" is left null/empty:
    # there is no extraction model run on uploads, only retrieval+generation
    # over the raw extracted text, per the stated scope.
    new_chunks = []
    overall_warning = None
    for p in pages:
        chunk_id = f"{file_id}_p{p['page_number']}"
        new_chunks.append({
            "chunk_id": chunk_id,
            "doc_id": f"{file.filename}::page{p['page_number']}",
            "source": file.filename,
            "text": p["text"] or "(no extractable text on this page)",
            "tokens": (p["text"] or "").split(),
            "char_len": len(p["text"] or ""),
            "word_len": len((p["text"] or "").split()),
            "ground_truth": {"vendor": None, "date": None, "amount": None,
                              "all_vendors": [], "all_dates": [], "all_amounts": []},
        })
        if p.get("warning") and not overall_warning:
            overall_warning = p["warning"]

    append_chunks(sessions.chunks_path(session_id), new_chunks)
    n_indexed = rebuild_index(
        sessions.chunks_path(session_id), sessions.index_dir(session_id),
        settings.embedding_model, settings.upload_embedding_device,
    )
    retriever_cache.invalidate(session_id)  # next query in this session picks up the new index

    uploaded = UploadedFile(
        file_id=file_id, original_filename=file.filename, saved_path=str(saved_path),
        content_type=file.content_type, page_count=len(pages),
        extracted_text="\n\n".join(p["text"] for p in pages),
        chunk_ids=[c["chunk_id"] for c in new_chunks], warning=overall_warning,
    )
    sessions.add_file(session_id, uploaded)

    # Logs metadata only (filename, page/chunk counts) — never the
    # extracted text itself.
    logger.info("upload processed: file_id=%s pages=%d chunks=%d warning=%s",
                file_id, len(pages), len(new_chunks), bool(overall_warning))

    return UploadResponse(
        file_id=file_id, original_filename=file.filename, content_type=file.content_type,
        page_count=len(pages),
        pages=[PageExtraction(**p) for p in pages],
        chunks_indexed=n_indexed, warning=overall_warning,
    )


@app.get("/session", response_model=SessionResponse)
def get_session(http_request: Request, response: Response):
    session_id = _get_session_id(http_request, response)
    files = sessions.list_files(session_id)
    return SessionResponse(
        session_id=session_id,
        files=[
            SessionFileSummary(
                file_id=f.file_id, original_filename=f.original_filename,
                page_count=f.page_count, extracted_char_count=len(f.extracted_text),
                warning=f.warning,
            ) for f in files
        ],
        total_chunks=sum(len(f.chunk_ids) for f in files),
    )


@app.delete("/session")
def clear_session(http_request: Request, response: Response):
    """'Clear uploaded receipts' — deletes this session's files, index, and
    cached retriever. Does not affect any other session or the demo/research
    corpus."""
    session_id = http_request.cookies.get(SESSION_COOKIE)
    if not session_id:
        return {"cleared": False, "detail": "No active session."}
    retriever_cache.invalidate(session_id)
    existed = sessions.clear_session(session_id)
    response.delete_cookie(SESSION_COOKIE)
    return {"cleared": existed}
