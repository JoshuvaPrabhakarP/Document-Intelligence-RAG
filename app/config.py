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


@lru_cache
def get_settings() -> Settings:
    return Settings()
