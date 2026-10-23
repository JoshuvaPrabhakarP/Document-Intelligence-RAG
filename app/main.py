"""
FastAPI application — the thin HTTP layer around the existing, unmodified
research pipeline (app/rag_pipeline.py). No retrieval/rerank/generation
logic lives in this file; it only handles HTTP concerns: validation,
status codes, readiness checks, and serving the static web interface.
"""

import logging
import os
import time

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.rag_pipeline import RAGPipeline
from app.schemas import HealthResponse, QueryRequest, QueryResponse, ReadinessResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

app = FastAPI(title="Document Intelligence RAG — Demo API")

settings = get_settings()
pipeline = RAGPipeline(settings)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Sample questions with known-correct answers against the synthetic demo
# corpus (demo_data/build_synthetic_demo_corpus.py) — used by the web UI's
# "try an example" buttons, and by the smoke test in tests/.
DEMO_SAMPLE_QUESTIONS = [
    "What is the total amount on the Example Grocers Inc. receipt?",
    "Which vendor issued a receipt for $156.42?",
    "What date is on the Sample Coffee Co. receipt?",
    "What is the total at Sample Gas Station?",
]


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness only — no dependency checks, must always be fast. A
    monitoring system restarting the process on a slow /health would be
    wrong; that's what /ready is for."""
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadinessResponse)
def ready():
    """Readiness — checks the things that would make a real query fail,
    WITHOUT forcing the heavy models to load just to answer this check
    (that would make every uptime monitor ping pay the cold-start cost)."""
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
def query(request: QueryRequest):
    start = time.monotonic()
    try:
        result = pipeline.answer(
            request.question, top_k=request.top_k, use_reranker=request.use_reranker
        )
    except Exception as exc:  # noqa: BLE001
        # Any unexpected failure (model load error, disk issue, etc.) —
        # logged with full detail server-side, but the client gets a clean
        # message, never a raw stack trace.
        logger.exception("Unhandled error answering query: %s", request.question)
        raise HTTPException(
            status_code=503,
            detail="The retrieval/answer pipeline is temporarily unavailable. Please try again.",
        ) from exc

    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info(
        "query answered in %.0fms (config=%s, evidence=%s)",
        elapsed_ms, settings.app_config, result.get("has_sufficient_evidence"),
    )

    return QueryResponse(config=settings.app_config, **result)
