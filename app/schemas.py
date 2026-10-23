from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)
    use_reranker: bool | None = Field(
        default=None, description="Override the configured default for this request."
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
