"""
ONE-OFF hosted-model compatibility + cost test for Together.ai's
Qwen/Qwen3.5-9B -- run manually, once, against the real hosted API, using
ONLY the synthetic demo corpus. This is NOT part of the regular pytest
suite and NOT eval/run_eval.py's full evaluation harness -- a small,
budget-capped, manually-invoked check of exactly what Milestone 5's
hosting decision needs verified before any deployment: does the real
hosted model actually do structured tool calling and schema-constrained
output the way this app's code expects, and what does that cost.

SAFETY / SCOPE, enforced in code below, not just documented here:
  - Refuses to run unless OLLAMA_BASE_URL is explicitly pointed away from
    localhost (so this can never silently run against local Ollama
    instead of the intended hosted provider, and never runs by accident).
  - Hard $0.25 total budget cap, tracked from REAL per-call token usage
    against Together's published Qwen3.5-9B rates ($0.17/1M input,
    $0.25/1M output -- re-verify these are still current before relying
    on this script's estimate for a real spending decision), checked
    BEFORE every call; stops and reports partial results rather than
    risk exceeding it.
  - max_tokens=400 per call, bounding worst-case single-call cost.
  - Exactly 10 question-flows, covering both workflows and every
    required scenario (see QUESTIONS below) -- not a full evaluation,
    not a re-run of eval/run_eval.py's held-out set.
  - Demo corpus only (mode="demo") -- never touches upload/session code
    or any real document content.
  - Never logs the API key. Never logs hidden reasoning/thinking text
    (if present) -- only final answers, not the model's internal
    deliberation. Never logs private content (there is none here --
    synthetic demo corpus only).

Run: venv/bin/python3 eval/together_compat_test.py
Output: prints a report to stdout; saves eval/results/together_compat_test.json
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import graph as agent_graph  # noqa: E402
from app.agent.state import new_state, resume_state  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.rag_pipeline import RAGPipeline  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent

# Together.ai published rates for Qwen/Qwen3.5-9B at the time this was
# written (confirmed from together.ai/pricing, not an aggregator) --
# re-check before relying on this for a real budget decision later.
INPUT_RATE_PER_TOKEN = 0.17 / 1_000_000
OUTPUT_RATE_PER_TOKEN = 0.25 / 1_000_000
BUDGET_USD = 0.25
MAX_TOKENS_PER_CALL = 400
# Conservative worst-case input size assumption for the pre-call check
# (this app's prompts: system instructions + a few retrieved passages +
# question -- generously bounded, not a real limit on the provider).
WORST_CASE_INPUT_TOKENS_PER_CALL = 4000

call_log: list[dict] = []
budget_state = {"spent_usd": 0.0}


class BudgetExceeded(RuntimeError):
    pass


def _worst_case_next_call_cost() -> float:
    return (WORST_CASE_INPUT_TOKENS_PER_CALL * INPUT_RATE_PER_TOKEN
            + MAX_TOKENS_PER_CALL * OUTPUT_RATE_PER_TOKEN)


def install_usage_tracker():
    """Wraps the OpenAI SDK's Completions.create globally for this
    process so every real call made by app/agent/llm.py AND
    app/generation/ollama_provider.py (both construct their own OpenAI
    client instances, but both route through this same SDK method) is
    logged and budget-checked, without modifying either production file."""
    import openai.resources.chat.completions as completions_module

    original_create = completions_module.Completions.create

    def tracked_create(self, *args, **kwargs):
        if budget_state["spent_usd"] + _worst_case_next_call_cost() > BUDGET_USD:
            raise BudgetExceeded(
                f"Refusing another call: ${budget_state['spent_usd']:.6f} spent so far, "
                f"next call's worst case could reach ${budget_state['spent_usd'] + _worst_case_next_call_cost():.6f}, "
                f"budget is ${BUDGET_USD}."
            )
        kwargs.setdefault("max_tokens", MAX_TOKENS_PER_CALL)
        t0 = time.monotonic()
        response = original_create(self, *args, **kwargs)
        latency_ms = (time.monotonic() - t0) * 1000

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        reasoning_tokens = None
        if usage is not None:
            details = getattr(usage, "completion_tokens_details", None)
            if details is not None:
                reasoning_tokens = getattr(details, "reasoning_tokens", None)

        if prompt_tokens is not None and completion_tokens is not None:
            cost = prompt_tokens * INPUT_RATE_PER_TOKEN + completion_tokens * OUTPUT_RATE_PER_TOKEN
            usage_status = "measured"
        else:
            cost = 0.0  # cannot estimate without real usage -- NOT invented
            usage_status = "unavailable"
        budget_state["spent_usd"] += cost

        call_log.append({
            "model": kwargs.get("model"),
            "latency_ms": round(latency_ms, 1),
            "had_tools_param": "tools" in kwargs,
            "had_json_schema_response_format": bool(
                kwargs.get("response_format") and kwargs["response_format"].get("type") == "json_schema"
            ),
            "returned_tool_calls": bool(getattr(response.choices[0].message, "tool_calls", None)),
            "usage_status": usage_status,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,  # None if the provider doesn't report this field separately
            "estimated_cost_usd": round(cost, 6),
            "cumulative_cost_usd": round(budget_state["spent_usd"], 6),
        })
        return response

    completions_module.Completions.create = tracked_create


def _calls_since(mark: int) -> list[dict]:
    return call_log[mark:]


def run_questions(pipeline: RAGPipeline) -> list[dict]:
    results = []

    def record(label, workflow, question, check_fn):
        mark = len(call_log)
        error = None
        try:
            outcome = check_fn()
        except BudgetExceeded:
            raise  # propagate -- stop the whole run, don't swallow this
        except Exception as exc:  # noqa: BLE001 -- a real-model call failing is a result to report, not a crash
            outcome = {"passed": False, "detail": f"exception: {exc}"}
            error = str(exc)
        calls = _calls_since(mark)
        results.append({
            "label": label, "workflow": workflow, "question": question,
            "passed": outcome.get("passed"), "detail": outcome.get("detail"),
            "error": error, "model_call_count": len(calls), "calls": calls,
            "total_latency_ms": round(sum(c["latency_ms"] for c in calls), 1),
            "total_prompt_tokens": sum(c["prompt_tokens"] or 0 for c in calls),
            "total_completion_tokens": sum(c["completion_tokens"] or 0 for c in calls),
            "total_estimated_cost_usd": round(sum(c["estimated_cost_usd"] for c in calls), 6),
        })
        print(f"[{label}] passed={outcome.get('passed')} calls={len(calls)} "
              f"cost=${sum(c['estimated_cost_usd'] for c in calls):.6f} :: {outcome.get('detail', '')}",
              file=sys.stderr)

    # --- 1. Standard RAG: correct total + valid citation ---
    def q1():
        r = pipeline.answer("What is the total amount on the Example Grocers Inc. receipt?", top_k=5)
        ok = ("42.17" in (r.get("answer") or "")) and any(s["chunk_id"] == "demo_000" for s in r.get("sources", []))
        return {"passed": ok, "detail": f"answer={r.get('answer')!r} sources={[s['chunk_id'] for s in r.get('sources', [])]}"}
    record("1-standard-correct-total", "standard", "What is the total amount on the Example Grocers Inc. receipt?", q1)

    # --- 2. Standard RAG: missing info -> insufficient-evidence fallback ---
    def q2():
        r = pipeline.answer("What payment method was used on the Sample Coffee Co. receipt?", top_k=5)
        ok = r.get("has_sufficient_evidence") is False
        return {"passed": ok, "detail": f"has_sufficient_evidence={r.get('has_sufficient_evidence')} answer={r.get('answer')!r}"}
    record("2-standard-fallback", "standard", "What payment method was used on the Sample Coffee Co. receipt?", q2)

    # --- 3/4. Standard RAG: ambiguous -> clarify -> followup correct ---
    pending = {}

    def q3():
        r = pipeline.answer("What is the total?", top_k=5)
        pending["original_question"] = "What is the total?"
        ok = r.get("route") == "clarify"
        return {"passed": ok, "detail": f"route={r.get('route')} clarification_question={r.get('clarification_question')!r}"}
    record("3-standard-ambiguous-clarify", "standard", "What is the total?", q3)

    def q4():
        resolved = f"{pending['original_question']} Specifically, I mean: The Example Grocers Inc. receipt."
        r = pipeline.answer(resolved, top_k=5, skip_disambiguation=True)
        ok = "42.17" in (r.get("answer") or "")
        return {"passed": ok, "detail": f"resolved_question={resolved!r} answer={r.get('answer')!r}"}
    record("4-standard-clarify-followup-correct", "standard", "(resolved) The Example Grocers Inc. receipt", q4)

    # --- 5. Agent: correct total + citation + genuine tool call + schema-accepted evidence assessment ---
    def q5():
        r = agent_graph.build_graph().invoke(new_state("What is the total amount on the Example Grocers Inc. receipt?", "demo", None))
        used_real_tool = any(t["tool"] == "search_documents" for t in r.get("execution_trace", []))
        ok = (r.get("route") == "answer" and "42.17" in (r.get("answer") or "") and used_real_tool and r.get("error") is None)
        return {"passed": ok, "detail": f"route={r.get('route')} answer={r.get('answer')!r} used_real_tool={used_real_tool} error={r.get('error')}"}
    record("5-agent-correct-total-tool-call", "agent", "What is the total amount on the Example Grocers Inc. receipt?", q5)

    # --- 6. Agent: missing info -> insufficient-evidence fallback ---
    def q6():
        r = agent_graph.build_graph().invoke(new_state("What payment method was used on the Sample Coffee Co. receipt?", "demo", None))
        ok = r.get("route") == "insufficient_evidence"
        return {"passed": ok, "detail": f"route={r.get('route')} answer={r.get('answer')!r}"}
    record("6-agent-fallback", "agent", "What payment method was used on the Sample Coffee Co. receipt?", q6)

    # --- 7/8. Agent: ambiguous -> clarify (schema route=clarify) -> followup correct ---
    agent_pending = {}

    def q7():
        state = new_state("What is the tax?", "demo", None)
        r = agent_graph.build_graph().invoke(state)
        agent_pending["original_question"] = "What is the tax?"
        agent_pending["clarification_question"] = r.get("clarification_question") or ""
        agent_pending["messages"] = r.get("messages", [])
        agent_pending["retrieved_passages"] = r.get("retrieved_passages", {})
        ok = r.get("route") == "clarify" and r.get("error") is None
        return {"passed": ok, "detail": f"route={r.get('route')} clarification_question={r.get('clarification_question')!r}"}
    record("7-agent-ambiguous-clarify-schema", "agent", "What is the tax?", q7)

    def q8():
        resumed = resume_state(
            original_question=agent_pending["original_question"],
            clarification_question=agent_pending["clarification_question"],
            clarification_answer="Fictional Hardware Supply",
            prior_messages=agent_pending["messages"],
            prior_retrieved_passages=agent_pending["retrieved_passages"],
            session_mode="demo", session_id=None,
        )
        r = agent_graph.build_graph().invoke(resumed)
        ok = r.get("route") == "answer" and "8.00" in (r.get("answer") or "")
        return {"passed": ok, "detail": f"route={r.get('route')} answer={r.get('answer')!r}"}
    record("8-agent-clarify-followup-correct", "agent", "(resumed) Fictional Hardware Supply", q8)

    # --- 9. Thinking enabled vs disabled comparison (reuses question 1's shape) ---
    def q9():
        import openai.resources.chat.completions as completions_module
        original_create = completions_module.Completions.create

        def create_with_thinking_enabled(self, *args, **kwargs):
            kwargs["extra_body"] = {"think": True, "reasoning": {"enabled": True}}
            return original_create(self, *args, **kwargs)

        completions_module.Completions.create = create_with_thinking_enabled
        try:
            r = pipeline.answer("What is the total on the Mock Bookstore receipt?", top_k=5)
        finally:
            completions_module.Completions.create = original_create
        ok = "37.45" in (r.get("answer") or "")
        return {"passed": ok, "detail": f"(thinking ENABLED) answer={r.get('answer')!r} -- compare latency/tokens to a thinking-disabled call of similar shape"}
    record("9-thinking-enabled-comparison", "standard", "What is the total on the Mock Bookstore receipt? (thinking enabled)", q9)

    # --- 10. Agent on a different vendor -- consistency check ---
    def q10():
        r = agent_graph.build_graph().invoke(new_state("What is the total at Fictional Hardware Supply?", "demo", None))
        ok = r.get("route") == "answer" and "156.42" in (r.get("answer") or "")
        return {"passed": ok, "detail": f"route={r.get('route')} answer={r.get('answer')!r}"}
    record("10-agent-consistency-check", "agent", "What is the total at Fictional Hardware Supply?", q10)

    return results


def main():
    settings = get_settings()
    if "localhost" in settings.ollama_base_url or "127.0.0.1" in settings.ollama_base_url:
        print(
            "REFUSING TO RUN: OLLAMA_BASE_URL is still pointed at localhost "
            f"({settings.ollama_base_url!r}). This script is specifically for testing a "
            "HOSTED provider -- set OLLAMA_BASE_URL/OLLAMA_MODEL/GENERATION_API_KEY to "
            "Together.ai's values in .env first. See PROGRESS.md's Milestone 5 section "
            "for the exact steps. Not making any API calls.",
            file=sys.stderr,
        )
        sys.exit(1)
    if settings.generation_api_key == "ollama":
        print(
            "REFUSING TO RUN: GENERATION_API_KEY is still the local Ollama placeholder. "
            "Set it to your real Together.ai key in .env first (never paste it in chat). "
            "Not making any API calls.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Target: {settings.ollama_base_url} model={settings.ollama_model}", file=sys.stderr)
    print(f"Budget cap: ${BUDGET_USD} | max_tokens/call: {MAX_TOKENS_PER_CALL}", file=sys.stderr)

    install_usage_tracker()
    pipeline = RAGPipeline(settings)

    try:
        results = run_questions(pipeline)
        stopped_early = False
    except BudgetExceeded as exc:
        print(f"STOPPED EARLY: {exc}", file=sys.stderr)
        results = []  # whatever ran before the exception is still in call_log; the per-question results list may be incomplete
        stopped_early = True

    out = {
        "provider": settings.ollama_base_url,
        "model": settings.ollama_model,
        "budget_usd": BUDGET_USD,
        "total_spent_usd": round(budget_state["spent_usd"], 6),
        "stopped_early_for_budget": stopped_early,
        "results": results,
        "all_calls": call_log,
    }
    out_path = EVAL_DIR / "results" / "together_compat_test.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nTotal spent: ${budget_state['spent_usd']:.6f} of ${BUDGET_USD} budget.", file=sys.stderr)
    print(f"Saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
