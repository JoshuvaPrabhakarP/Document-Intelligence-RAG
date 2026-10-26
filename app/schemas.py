from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)
    use_reranker: bool | None = Field(
        default=None, description="Override the configured default for this request."
    )
    mode: str = Field(
        default="demo",
        description="'demo' (synthetic sample receipts) or 'upload' (this session's uploaded receipts only).",
    )
    workflow: str = Field(
        default="agent",
        description=(
            "'agent' (LangGraph tool-calling workflow, Milestone 3; DEFAULT as of Milestone 5 -- has a "
            "verified evidence-support gate, app/agent/llm.py::assess_evidence(), that checks its cited "
            "quote is a real substring of the retrieved text before answering) or 'standard' (direct "
            "retrieve-then-generate; weaker grounding -- relies on the model saying \"I don't know\" with "
            "no equivalent verification)."
        ),
    )


class SourceReference(BaseModel):
    chunk_id: str
    source: str
    score: float
    excerpt: str


class ExecutionStep(BaseModel):
    """One node/tool execution in an agent run — metadata only (node name,
    tool name, duration, a short outcome description). Never raw document
    content or full tool arguments, per the 'keep sensitive tool arguments
    out of persistent logs' requirement — this is a response field shown to
    the requesting user for this one run, not written to any log."""
    node: str
    tool: str | None
    duration_ms: float
    detail: str


class QueryResponse(BaseModel):
    answer: str | None
    sources: list[SourceReference]
    has_sufficient_evidence: bool | None
    retrieval_latency_ms: float
    rerank_latency_ms: float | None
    generation_latency_ms: float | None
    config: str
    error: str | None = None
    # Agent-workflow-only fields (None/omitted for workflow="standard"):
    route: str | None = None
    clarification_question: str | None = None
    execution_trace: list[ExecutionStep] | None = None


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    ready: bool
    config: str
    checks: dict[str, bool]
    detail: str | None = None


class PageExtraction(BaseModel):
    page_number: int
    text: str
    used_ocr: bool
    warning: str | None = None


class UploadResponse(BaseModel):
    file_id: str
    original_filename: str
    content_type: str
    page_count: int
    pages: list[PageExtraction]
    chunks_indexed: int
    warning: str | None = None


class SessionFileSummary(BaseModel):
    file_id: str
    original_filename: str
    page_count: int
    extracted_char_count: int
    warning: str | None = None


class SessionResponse(BaseModel):
    session_id: str
    files: list[SessionFileSummary]
    total_chunks: int
