"""
App configuration — every setting is read from environment variables (via
pydantic-settings), never hardcoded. See .env.example for the full list.

Two named configurations, selected by APP_CONFIG:
  - "research": the real, evaluated 3,712-document corpus + bge-large +
    bge-reranker-large — exactly Step 4/5's system. NEVER served publicly
    (see PROGRESS.md for why: SROIE/WildReceipt licensing is unclear).
  - "demo": the synthetic sample-receipt corpus (demo_data/), safe for
    public deployment. Uses the SAME models by default (they're not the
    resource bottleneck — see PROGRESS.md's RAM/VRAM measurements), only
    the corpus/index paths differ.

Loading research metrics (Hit@10, MRR, etc.) is only ever valid for
APP_CONFIG=research; the demo config has not been evaluated and the API
response says so explicitly when APP_CONFIG=demo.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_config: str = "demo"  # "research" or "demo" — demo is the safe default

    # Corpus / index paths — demo values by default; research deployments
    # override these explicitly via .env (never committed with real paths
    # pointing at a public deployment).
    chunks_path: str = "demo_data/demo_chunks.jsonl"
    index_dir: str = "demo_data/index"

    embedding_model: str = "BAAI/bge-large-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-large"
    use_reranker: bool = True
    device: str = "cuda"  # "cuda" or "cpu" — falls back to cpu automatically if cuda unavailable

    retrieval_pool_size: int = 50
    top_k: int = 5

    generation_provider: str = "ollama"  # only "ollama" implemented now; see app/generation/
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen3:8b"
    generation_timeout_s: int = 60

    max_question_length: int = 500

    # --- Upload mode (user-supplied receipts, session-isolated) ---
    upload_dir: str = "/tmp/doc-rag-uploads"  # OUTSIDE the repo — never gitignored-by-convention, just not in it
    upload_max_file_mb: int = 10
    upload_max_pdf_pages: int = 20
    upload_processing_timeout_s: int = 45
    upload_session_ttl_minutes: int = 30
    upload_embedding_device: str = "cpu"  # see app/upload/indexing.py docstring for why not "cuda"
    upload_max_files_per_session: int = 10  # caps disk/index growth from one session looping uploads

    # --- Practical request limits (Milestone 5: public deployment) ---
    # Deliberately simple, in-memory, single-process limits (see
    # app/ratelimit.py) -- this app runs as one uvicorn worker (see
    # Dockerfile's comment on why), so there's no shared-state problem a
    # heavier solution (Redis-backed, etc.) would be solving here.
    rate_limit_query_per_minute: int = 20       # per client IP
    rate_limit_upload_per_minute: int = 5       # per client IP (heavier than a query)
    max_concurrent_heavy_requests: int = 3      # /query + /upload combined, across ALL clients


@lru_cache
def get_settings() -> Settings:
    return Settings()
