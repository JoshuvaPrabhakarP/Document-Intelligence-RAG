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

from app.agent.graph import MAX_CLARIFICATION_ROUNDS, MAX_CUMULATIVE_TOOL_CALLS, get_graph
from app.agent.state import new_state, resume_state
from app.config import get_settings
from app.excerpt import select_excerpt
from app.ratelimit import ConcurrencyLimiter, RateLimiter
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
from app.upload.session import PendingClarification, UploadedFile, new_session_id, sessions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

app = FastAPI(title="Document Intelligence RAG — Demo API")

settings = get_settings()
pipeline = RAGPipeline(settings)
# `sessions` is the shared singleton defined in app/upload/session.py (not
# constructed here) specifically so app/agent/tools.py can import the SAME
# instance without a circular import — see that module's comment.

# Practical deployment limits (Milestone 5) — see app/ratelimit.py's
# module docstring for why these are plain in-memory counters, not a
# general-purpose rate-limiting library.
query_rate_limiter = RateLimiter(settings.rate_limit_query_per_minute)
upload_rate_limiter = RateLimiter(settings.rate_limit_upload_per_minute)
heavy_request_limiter = ConcurrencyLimiter(settings.max_concurrent_heavy_requests)


def _client_key(request: Request) -> str:
    """Best-effort client identity for rate limiting — the direct peer
    address. Deliberately does NOT trust X-Forwarded-For (a client could
    set that header to anything); a real deployment behind a proxy that
    sets it from a trusted source could extend this, but trusting it
    unconditionally would make the per-IP limit trivially bypassable."""
    return request.client.host if request.client else "unknown"

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


def _run_agent_query(question: str, mode: str, session_id: str | None) -> dict:
    """Runs the Milestone 3 LangGraph workflow for one request and adapts
    its result into the same response shape pipeline.answer()/
    answer_from_retriever() produce (plus the agent-only route/
    clarification_question/execution_trace fields), so the frontend and
    QueryResponse schema don't need a second shape to handle.

    Clarification state is looked up / stored keyed by session_id, reusing
    the SAME per-session store uploads already use (app/upload/session.py)
    — so "clear uploaded receipts" or session expiry wipes any pending
    clarification for free, with no separate cleanup path to maintain.

    MAX_CLARIFICATION_ROUNDS / MAX_CUMULATIVE_TOOL_CALLS (Milestone 4) bound
    the WHOLE conversation thread, not just one turn — see graph.py's
    module docstring for why a per-turn-only budget isn't enough on its own.
    """
    pending = sessions.get_pending_clarification(session_id) if session_id else None
    if pending and pending.workflow != "agent":
        # Standard RAG also asks for clarification now (document-ambiguity
        # fix) and shares this one per-session slot -- a pending
        # clarification left by the OTHER workflow isn't a resume for
        # this one; treat it as a fresh question instead.
        pending = None
    if pending and (pending.rounds >= MAX_CLARIFICATION_ROUNDS
                     or pending.cumulative_tool_calls >= MAX_CUMULATIVE_TOOL_CALLS):
        sessions.clear_pending_clarification(session_id)
        return {
            "answer": "I wasn't able to resolve this after several rounds of clarification. "
                      "Please ask a single, more specific question.",
            "sources": [], "has_sufficient_evidence": False,
            "retrieval_latency_ms": 0.0, "rerank_latency_ms": None, "generation_latency_ms": 0.0,
            "error": f"Clarification limit reached ({pending.rounds} rounds, "
                     f"{pending.cumulative_tool_calls} cumulative tool calls).",
            "route": "insufficient_evidence", "clarification_question": None, "execution_trace": [],
        }

    if pending:
        original_question = pending.original_question
        state = resume_state(
            original_question=pending.original_question,
            clarification_question=pending.clarification_question,
            clarification_answer=question,
            prior_messages=pending.messages,
            prior_retrieved_passages=pending.retrieved_passages,
            session_mode=mode,
            session_id=session_id,
        )
    else:
        original_question = question
        state = new_state(question, mode, session_id)

    result = get_graph().invoke(state)

    if result["route"] == "clarify" and session_id:
        sessions.set_pending_clarification(session_id, PendingClarification(
            original_question=original_question,
            clarification_question=result["clarification_question"] or "",
            messages=result["messages"],
            retrieved_passages=result["retrieved_passages"],
            workflow="agent",
            rounds=(pending.rounds if pending else 0) + 1,
            cumulative_tool_calls=(pending.cumulative_tool_calls if pending else 0) + result["tool_call_count"],
        ))
    elif session_id:
        sessions.clear_pending_clarification(session_id)

    sources = [
        {
            "chunk_id": passage["chunk_id"],
            "source": passage["source"],
            "score": passage.get("score") or 0.0,
            "excerpt": select_excerpt(passage["text"], original_question, result["answer"]),
        }
        for pid in result["cited_passage_ids"]
        if (passage := result["retrieved_passages"].get(pid)) is not None
    ]

    # Reranking happens inside the "tools" node now too (app/agent/tools.py),
    # same model as the standard pipeline's reranker — it's just not a
    # separately timed phase here, folded into tool_ms below, so
    # rerank_latency_ms is deliberately None rather than a fabricated 0.
    tool_ms = sum(t["duration_ms"] for t in result["execution_trace"] if t["node"] == "tools")
    model_ms = sum(t["duration_ms"] for t in result["execution_trace"] if t["node"] in ("agent", "assess_evidence"))

    has_evidence = {"answer": True, "insufficient_evidence": False}.get(result["route"])

    return {
        "answer": result["answer"],
        "sources": sources,
        "has_sufficient_evidence": has_evidence,
        "retrieval_latency_ms": tool_ms,
        "rerank_latency_ms": None,
        "generation_latency_ms": model_ms,
        "error": result["error"],
        "route": result["route"],
        "clarification_question": result["clarification_question"],
        "execution_trace": result["execution_trace"],
    }


# Fixed, documented template combining the original ambiguous question
# with the user's clarification answer -- applied uniformly, never tuned
# per question, matching the same discipline eval/run_eval.py's own
# _resolve_question() used for scoring Standard RAG's equivalent-
# information case in Milestone 4.
def _resolve_question(original_question: str, clarification_answer: str) -> str:
    return f"{original_question} Specifically, I mean: {clarification_answer}."


def _run_standard_query(question: str, mode: str, session_id: str | None,
                         top_k: int | None, use_reranker: bool | None) -> dict:
    """Standard RAG's clarification path (document-ambiguity fix): a
    single extra generation call when app/disambiguation.py's metadata
    signal plus the model's own judgment decide a question could have a
    genuinely different answer depending on which of several comparably-
    relevant documents it means (see app/rag_pipeline.py). Resuming is
    much simpler than the agent's: there's no tool-calling conversation to
    carry over, just the original question + the user's answer, combined
    into one resolved question via _resolve_question() and re-run with
    disambiguation suppressed (the user already named a specific
    document) -- not a second LLM call, not a chat history.

    Shares the SAME per-session PendingClarification slot and
    MAX_CLARIFICATION_ROUNDS cap as the agent workflow (see
    PendingClarification.workflow for why a pending clarification from
    the OTHER workflow is never treated as a resume here)."""
    pending = sessions.get_pending_clarification(session_id) if session_id else None
    if pending and pending.workflow != "standard":
        pending = None
    if pending and pending.rounds >= MAX_CLARIFICATION_ROUNDS:
        sessions.clear_pending_clarification(session_id)
        return {
            "answer": "I wasn't able to resolve this after several rounds of clarification. "
                      "Please ask a single, more specific question.",
            "sources": [], "has_sufficient_evidence": False,
            "retrieval_latency_ms": 0.0, "rerank_latency_ms": None, "generation_latency_ms": 0.0,
            "error": f"Clarification limit reached ({pending.rounds} rounds).",
            "route": "insufficient_evidence", "clarification_question": None,
        }

    if pending:
        resolved_question = _resolve_question(pending.original_question, question)
        original_question = pending.original_question
        skip_disambiguation = True
        rounds_so_far = pending.rounds
    else:
        resolved_question = question
        original_question = question
        skip_disambiguation = False
        rounds_so_far = 0

    if mode == "upload":
        retriever = retriever_cache.get_retriever(
            session_id, str(sessions.index_dir(session_id)), str(sessions.chunks_path(session_id))
        )
        result = pipeline.answer_from_retriever(resolved_question, retriever, top_k=top_k,
                                                  use_reranker=use_reranker, skip_disambiguation=skip_disambiguation)
    else:
        result = pipeline.answer(resolved_question, top_k=top_k, use_reranker=use_reranker,
                                   skip_disambiguation=skip_disambiguation)

    if result.get("route") == "clarify" and session_id:
        sessions.set_pending_clarification(session_id, PendingClarification(
            original_question=original_question,
            clarification_question=result["clarification_question"] or "",
            messages=[], retrieved_passages={},
            workflow="standard",
            rounds=rounds_so_far + 1,
        ))
    elif session_id:
        sessions.clear_pending_clarification(session_id)

    return result


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, http_request: Request, response: Response):
    start = time.monotonic()
    if request.workflow not in ("standard", "agent"):
        raise HTTPException(status_code=422, detail=f"Unknown workflow {request.workflow!r}. Use 'standard' or 'agent'.")

    if not query_rate_limiter.allow(_client_key(http_request)):
        raise HTTPException(status_code=429, detail="Too many questions from this address. Please wait a minute and try again.")
    if not heavy_request_limiter.try_acquire():
        raise HTTPException(status_code=503, detail="The server is busy handling other requests right now. Please try again shortly.")
    try:
        session_id = None
        if request.mode == "upload":
            session_id = _get_session_id(http_request, response)
            files = sessions.list_files(session_id)
            if not files:
                raise HTTPException(
                    status_code=400,
                    detail="No receipts uploaded in this session yet. Upload a file first, "
                           "or switch to demo mode.",
                )
        elif request.mode == "demo":
            # Clarification state (both workflows now) needs a per-browser
            # identity even in demo mode.
            session_id = _get_session_id(http_request, response)
        else:
            raise HTTPException(status_code=422, detail=f"Unknown mode {request.mode!r}. Use 'demo' or 'upload'.")

        if request.workflow == "agent":
            result = _run_agent_query(request.question, request.mode, session_id)
        else:
            result = _run_standard_query(
                request.question, request.mode, session_id, request.top_k, request.use_reranker
            )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error answering query (mode=%s, workflow=%s)", request.mode, request.workflow)
        raise HTTPException(
            status_code=503,
            detail="The retrieval/answer pipeline is temporarily unavailable. Please try again.",
        ) from exc
    finally:
        heavy_request_limiter.release()

    elapsed_ms = (time.monotonic() - start) * 1000
    # Deliberately logs only metadata (mode, latency, evidence flag) — never
    # the question text or any retrieved/extracted content, per the
    # "keep uploads and extracted content out of ... application logs" requirement.
    logger.info("query answered in %.0fms (mode=%s, workflow=%s, route=%s, evidence=%s)",
                elapsed_ms, request.mode, request.workflow, result.get("route"), result.get("has_sufficient_evidence"))

    return QueryResponse(config=request.mode, **result)


@app.post("/upload", response_model=UploadResponse)
def upload(file: UploadFile, http_request: Request, response: Response):
    session_id = _get_session_id(http_request, response)

    if not upload_rate_limiter.allow(_client_key(http_request)):
        raise HTTPException(status_code=429, detail="Too many uploads from this address. Please wait a minute and try again.")

    existing_files = sessions.list_files(session_id)
    if len(existing_files) >= settings.upload_max_files_per_session:
        raise HTTPException(
            status_code=429,
            detail=f"This session already has {len(existing_files)} documents, the limit per session. "
                   "Clear your documents first, or ask about what's already uploaded.",
        )

    if not heavy_request_limiter.try_acquire():
        raise HTTPException(status_code=503, detail="The server is busy handling other requests right now. Please try again shortly.")
    try:
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
    finally:
        heavy_request_limiter.release()


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
