"""
Step 4: Thin wrapper around the local LLM (Qwen3 8B via Ollama), used for
answer generation in the eval harness.

Talks to Ollama's OpenAI-compatible endpoint using the `openai` package
that's already installed as a transitive dependency (no new package needed)
— this also lets Ragas's LangchainLLMWrapper judge role point at the exact
same endpoint via `ChatOpenAI`, so there's only one LLM integration path
in the whole project, not two.

Qwen3 is a "thinking" model — Ollama returns a visible <think>...</think>
reasoning block ahead of the real answer (confirmed during this project's
own Ollama setup test). Left un-managed this is a serious problem two ways
over: (1) it must be stripped before the text is treated as "the answer",
or it corrupts the logged `generated_answer` field and every Ragas metric
that scores it; (2) it is drastically slower — measured at ~55s for a
single trivial factual answer with thinking on, vs. ~1-3s with it off, a
~20-60x difference that would turn the full eval matrix into a multi-hour-
to-overnight run. Ollama honors a `"think": false` field on both its native
API and its OpenAI-compatible endpoint (confirmed directly against this
project's running Ollama instance) — passed here via `extra_body`.

Usage:
    from llm_client import generate_answer
    answer, latency_ms = generate_answer(query, context_chunks)
"""

import re
import time

OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL = "qwen3:8b"

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    return _client


def strip_thinking(text):
    """Remove a <think>...</think> block, if present. Falls back to the
    original text untouched if no such block is found (some responses may
    not include one, e.g. very short answers)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def build_prompt(query, context_chunks):
    context_block = "\n\n".join(
        f"[Context {i+1}]\n{c['text']}" for i, c in enumerate(context_chunks)
    )
    return (
        "Answer the question using ONLY the context provided below. "
        "Be concise — a short phrase or sentence, not an essay. "
        "If the answer is not present in the context, say \"I don't know.\"\n\n"
        f"{context_block}\n\n"
        f"Question: {query}\nAnswer:"
    )


def generate_answer(query, context_chunks, model=MODEL):
    """context_chunks: list of dicts with a 'text' key (as returned by any
    retriever's search()). Returns (answer_text, latency_ms)."""
    client = _get_client()
    prompt = build_prompt(query, context_chunks)

    start = time.monotonic()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"think": False},
    )
    latency_ms = (time.monotonic() - start) * 1000

    raw = response.choices[0].message.content
    answer = strip_thinking(raw)
    return answer, latency_ms


if __name__ == "__main__":
    # Self-test: confirm the Ollama endpoint is reachable and thinking-block
    # stripping actually works on a REAL response, not just the regex in
    # isolation — same "verify on real output before trusting it" discipline
    # used throughout this project.
    sample_context = [{"text": "WALMART\nTotal: $89.09\nDate: 12/22/17"}]
    answer, latency_ms = generate_answer(
        "What is the total amount on this receipt?", sample_context
    )
    print(f"Answer: {answer!r}")
    print(f"Latency: {latency_ms:.1f}ms")
    assert "<think>" not in answer.lower(), "thinking block leaked into the answer"
    print("Self-test passed: no <think> block leaked into the final answer.")
