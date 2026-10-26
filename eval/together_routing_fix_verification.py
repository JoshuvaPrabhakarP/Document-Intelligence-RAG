"""
Follow-up to eval/together_compat_test.py -- verifies the two routing
fixes made after that test found real issues:
  1. app/disambiguation.py::format_candidates() used to label every demo-
     corpus candidate "synthetic_demo" (the dataset name, identical for
     all documents) instead of something that actually distinguishes
     them -- a disambiguation signal that looks like 4 identical choices
     is not a usable signal. Now falls back to the chunk's own first
     text line (reliably the vendor/business name on a receipt, for ANY
     corpus) when 'source' doesn't distinguish the candidates.
  2. Both the Standard RAG and Agent routing prompts (scripts/
     llm_client.py::build_prompt(), app/agent/llm.py::assess_evidence())
     now explicitly say: if the question ALREADY names a specific
     vendor/document, other comparably-scored documents retrieval
     happens to return are noise, not alternatives -- don't ask which
     one; fall back to insufficient_evidence instead of clarify.

Re-runs the ORIGINAL two failing cases from together_compat_test.py's
report (preserved untouched, not overwritten) plus one fresh paraphrase
of each, against the real Together.ai endpoint. Reuses that script's
budget tracker and seeds it with the ALREADY-SPENT cumulative total from
every real call made in this engagement so far, so the $0.25 cap is
enforced across the whole authorization, not reset to zero here.

Run: venv/bin/python3 eval/together_routing_fix_verification.py
Output: prints a report; saves eval/results/together_routing_fix_verification.json
(together_compat_test.json is never touched by this script).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import graph as agent_graph  # noqa: E402
from app.agent.state import new_state  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.rag_pipeline import RAGPipeline  # noqa: E402
from eval.together_compat_test import BudgetExceeded, budget_state, call_log, install_usage_tracker  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent

# Cumulative spend from every prior real call in this engagement:
# the one-off URL/auth verification call (~$0.0000037) + the full
# 10-question compat test ($0.004190, from eval/results/
# together_compat_test.json's own "total_spent_usd"). Seeding the
# tracker with this means its $0.25 cap check below is against the
# TRUE remaining budget for the whole authorization, not a fresh $0.25.
PRIOR_SPEND_USD = 0.0000037 + 0.004190


def _calls_since(mark: int) -> list[dict]:
    return call_log[mark:]


def run_check(label: str, workflow: str, question: str, run_fn, expect_route: str) -> dict:
    mark = len(call_log)
    try:
        route, detail = run_fn()
        passed = route == expect_route
        error = None
    except BudgetExceeded as exc:
        raise
    except Exception as exc:  # noqa: BLE001 -- captured as a failed result, not a crash
        route, detail, passed, error = None, None, False, str(exc)

    calls = _calls_since(mark)
    result = {
        "label": label,
        "workflow": workflow,
        "question": question,
        "expected_route": expect_route,
        "observed_route": route,
        "passed": passed,
        "detail": detail,
        "error": error,
        "model_call_count": len(calls),
        "calls": calls,
        "total_latency_ms": sum(c["latency_ms"] for c in calls),
        "total_prompt_tokens": sum(c["prompt_tokens"] or 0 for c in calls),
        "total_completion_tokens": sum(c["completion_tokens"] or 0 for c in calls),
        "total_estimated_cost_usd": round(sum(c["estimated_cost_usd"] for c in calls), 6),
    }
    print(f"[{label}] passed={passed} route={route} calls={len(calls)} "
          f"cost=${result['total_estimated_cost_usd']:.6f} :: {detail}")
    return result


def main():
    settings = get_settings()
    if "localhost" in settings.ollama_base_url or "127.0.0.1" in settings.ollama_base_url:
        print(f"REFUSING TO RUN: OLLAMA_BASE_URL is still pointed at localhost "
              f"({settings.ollama_base_url!r}).", file=sys.stderr)
        sys.exit(1)

    install_usage_tracker()
    budget_state["spent_usd"] = PRIOR_SPEND_USD
    print(f"Target: {settings.ollama_base_url} model={settings.ollama_model}", file=sys.stderr)
    print(f"Seeded cumulative spend: ${PRIOR_SPEND_USD:.6f} (budget cap ${0.25})", file=sys.stderr)

    pipeline = RAGPipeline(settings)
    results = []

    def q_standard_clarify_original():
        r = pipeline.answer("What is the total?", top_k=5)
        return r["route"], f"answer={r['answer']!r} clarification_question={r['clarification_question']!r}"

    def q_standard_clarify_paraphrase():
        r = pipeline.answer("Could you tell me the total amount?", top_k=5)
        return r["route"], f"answer={r['answer']!r} clarification_question={r['clarification_question']!r}"

    def q_agent_insufficient_original():
        state = agent_graph.build_graph().invoke(
            new_state("What payment method was used on the Sample Coffee Co. receipt?",
                      session_mode="demo", session_id=None)
        )
        return state["route"], f"answer={state.get('answer')!r} clarification_question={state.get('clarification_question')!r}"

    def q_agent_insufficient_paraphrase():
        state = agent_graph.build_graph().invoke(
            new_state("How was the purchase at Sample Coffee Co. paid for?",
                      session_mode="demo", session_id=None)
        )
        return state["route"], f"answer={state.get('answer')!r} clarification_question={state.get('clarification_question')!r}"

    checks = [
        ("1-standard-clarify-original", "standard", "What is the total?", q_standard_clarify_original, "clarify"),
        ("2-standard-clarify-paraphrase", "standard", "Could you tell me the total amount?", q_standard_clarify_paraphrase, "clarify"),
        ("3-agent-insufficient-original", "agent", "What payment method was used on the Sample Coffee Co. receipt?", q_agent_insufficient_original, "insufficient_evidence"),
        ("4-agent-insufficient-paraphrase", "agent", "How was the purchase at Sample Coffee Co. paid for?", q_agent_insufficient_paraphrase, "insufficient_evidence"),
    ]

    stopped_early = False
    for label, workflow, question, fn, expect in checks:
        try:
            results.append(run_check(label, workflow, question, fn, expect))
        except BudgetExceeded as exc:
            print(f"BUDGET EXCEEDED, stopping: {exc}", file=sys.stderr)
            stopped_early = True
            break

    report = {
        "provider": settings.ollama_base_url,
        "model": settings.ollama_model,
        "budget_usd": 0.25,
        "prior_spend_usd": PRIOR_SPEND_USD,
        "total_spent_usd": round(budget_state["spent_usd"], 6),
        "spent_this_run_usd": round(budget_state["spent_usd"] - PRIOR_SPEND_USD, 6),
        "stopped_early_for_budget": stopped_early,
        "results": results,
    }
    out_path = EVAL_DIR / "results" / "together_routing_fix_verification.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nCumulative spend (whole engagement): ${report['total_spent_usd']:.6f} of $0.25 budget.")
    print(f"Spent this run: ${report['spent_this_run_usd']:.6f}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
