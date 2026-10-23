"""
Ollama-backed GenerationProvider — the only provider wired in for this
milestone, per the stated scope ("keep Ollama for local development, do
not introduce a paid provider yet").

Reuses the ALREADY-TESTED prompt-building and <think>-stripping logic from
scripts/llm_client.py (Step 4's research code — found and fixed the ~13x
"thinking mode" latency trap there) rather than reimplementing it. This
file adds what the research script didn't need: configurable base_url/
model/timeout (via app/config.py, not hardcoded globals) and proper error
handling that raises GenerationError instead of letting a raw httpx/openai
exception reach the API layer.
"""

import sys
import time
from pathlib import Path

# scripts/ is research code, not a package — same sys.path convention used
# elsewhere in this project (e.g. rerank.py's own self-test) to reuse it
# without copying logic.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

from llm_client import build_prompt, strip_thinking  # noqa: E402

from .base import GenerationError, GenerationProvider


class OllamaProvider(GenerationProvider):
    def __init__(self, base_url: str, model: str, timeout_s: int = 60):
        from openai import OpenAI

        self.model = model
        self.timeout_s = timeout_s
        self.client = OpenAI(base_url=base_url, api_key="ollama", timeout=timeout_s)

    def generate(self, query: str, context_chunks):
        prompt = build_prompt(query, context_chunks)

        t0 = time.monotonic()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"think": False},  # ~13x latency fix, see scripts/llm_client.py
            )
        except Exception as exc:  # noqa: BLE001 — deliberately broad: any provider
            # failure (connection refused, timeout, malformed response) should
            # surface as a clean GenerationError, not leak a raw client exception.
            raise GenerationError(f"Ollama generation failed: {exc}") from exc
        latency_ms = (time.monotonic() - t0) * 1000

        raw = response.choices[0].message.content
        return strip_thinking(raw), latency_ms
