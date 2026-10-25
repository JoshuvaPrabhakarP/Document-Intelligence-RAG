"""
OpenAI-compatible GenerationProvider -- named "Ollama" for history (it was
the only provider for Milestones 1-4), but works against ANY OpenAI-
compatible chat-completions endpoint purely through configuration
(base_url/model/api_key, all from app/config.py) -- no code change needed
to point this at a hosted provider instead of local Ollama (Milestone 5).

Reuses the ALREADY-TESTED prompt-building and <think>-stripping logic from
scripts/llm_client.py (Step 4's research code — found and fixed the ~13x
"thinking mode" latency trap there) rather than reimplementing it. This
file adds what the research script didn't need: configurable base_url/
model/api_key/timeout (via app/config.py, not hardcoded globals) and
proper error handling that raises GenerationError instead of letting a
raw httpx/openai exception reach the API layer.
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
    def __init__(self, base_url: str, model: str, timeout_s: int = 60, api_key: str = "ollama"):
        from openai import OpenAI

        self.model = model
        self.timeout_s = timeout_s
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)

    def generate(self, query: str, context_chunks, candidates=None):
        prompt = build_prompt(query, context_chunks, candidates=candidates)

        t0 = time.monotonic()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                # Disables thinking mode for both Ollama/Qwen3 ("think") and
                # Together.ai's Qwen3.5-9B ("reasoning") -- see
                # scripts/llm_client.py's generate_answer() for the full
                # explanation of this cross-provider compatibility fix.
                extra_body={"think": False, "reasoning": {"enabled": False}},
            )
        except Exception as exc:  # noqa: BLE001 — deliberately broad: any provider
            # failure (connection refused, timeout, malformed response) should
            # surface as a clean GenerationError, not leak a raw client exception.
            raise GenerationError(f"Ollama generation failed: {exc}") from exc
        latency_ms = (time.monotonic() - t0) * 1000

        raw = response.choices[0].message.content
        return strip_thinking(raw), latency_ms
