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


class SourceReference(BaseModel):
    chunk_id: str
    source: str
    score: float
    excerpt: str


class QueryResponse(BaseModel):
    answer: str | None
    sources: list[SourceReference]
    has_sufficient_evidence: bool | None
    retrieval_latency_ms: float
    rerank_latency_ms: float | None
    generation_latency_ms: float | None
    config: str
    error: str | None = None


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
