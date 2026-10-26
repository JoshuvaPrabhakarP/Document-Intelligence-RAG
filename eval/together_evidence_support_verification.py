"""
Follow-up to eval/together_routing_fix_verification.py -- verifies the
evidence-support gate added to app/agent/llm.py::assess_evidence() after
that script's paraphrase check found the Agent fabricating an answer
("paid for with cash" / "paid for with a total of $9.75") for a receipt
that names the right vendor but has no payment-method field at all. The
fix requires a verbatim supporting_quote when route="answer" and VERIFIES
it's a real substring of the retrieved text rather than trusting the
model -- see app/agent/llm.py's EVIDENCE_SCHEMA docstring and
tests/test_agent.py's new deterministic tests for the mechanism itself.
This script checks the REAL hosted model against it, which the
deterministic tests (mocked client) cannot.

Against the real Together.ai endpoint:
  1. Both previously-failing payment-method phrasings (original + the
     paraphrase that fabricated an answer last round).
  2. Two FRESH paraphrases never tried before.
  3. A synthetic receipt that EXPLICITLY states a payment method (via the
     same "patch execute_search_documents with a synthetic passage"
     technique tests/test_agent.py already uses for injection testing --
     clearly fabricated test data, not a real document) -- to confirm the
     gate doesn't just reject everything, it still passes genuinely
     supported answers.
  4. Regression checks against the existing total+citation and clarify
     behavior, plus one NEW absent-attribute question in a different
     domain (loyalty rewards, not payment) to confirm the fix is general,
     not special-cased to payment questions.

Reuses eval/together_compat_test.py's budget tracker, seeded with the
cumulative spend from every prior real call in this engagement
(eval/results/together_routing_fix_verification.json's own
"total_spent_usd": 0.006090) so the $0.25 cap is enforced across the
whole authorization. Original reports (together_compat_test.json,
together_routing_fix_verification.json) are never touched -- this saves
to its own eval/results/together_evidence_support_verification.json.

Run: venv/bin/python3 eval/together_evidence_support_verification.py
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import graph as agent_graph  # noqa: E402
from app.agent.state import new_state  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.rag_pipeline import RAGPipeline  # noqa: E402
from eval.together_compat_test import BudgetExceeded, budget_state, call_log, install_usage_tracker  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent

# From eval/results/together_routing_fix_verification.json's own
# "total_spent_usd" -- the true cumulative spend across every real call
# made in this engagement so far (URL/auth check + 10-question compat
# test + routing-fix verification + its one extra diagnostic re-run).
PRIOR_SPEND_USD = 0.006090


def _calls_since(mark: int) -> list[dict]:
    return call_log[mark:]


def run_check(label: str, question: str, run_fn, expect_route: str, expect_contains: str | None = None) -> dict:
    mark = len(call_log)
    try:
        route, answer, detail = run_fn()
        passed = route == expect_route and (expect_contains is None or expect_contains.lower() in (answer or "").lower())
        error = None
    except BudgetExceeded:
        raise
    except Exception as exc:  # noqa: BLE001
        route, answer, detail, passed, error = None, None, None, False, str(exc)

    calls = _calls_since(mark)
    result = {
        "label": label, "question": question, "expected_route": expect_route,
        "expected_contains": expect_contains, "observed_route": route, "observed_answer": answer,
        "passed": passed, "detail": detail, "error": error,
        "model_call_count": len(calls), "calls": calls,
        "total_latency_ms": sum(c["latency_ms"] for c in calls),
        "total_prompt_tokens": sum(c["prompt_tokens"] or 0 for c in calls),
        "total_completion_tokens": sum(c["completion_tokens"] or 0 for c in calls),
        "total_estimated_cost_usd": round(sum(c["estimated_cost_usd"] for c in calls), 6),
    }
    print(f"[{label}] passed={passed} route={route} calls={len(calls)} "
          f"cost=${result['total_estimated_cost_usd']:.6f} :: answer={answer!r} detail={detail}")
    return result


def _run_agent(question: str) -> tuple:
    state = agent_graph.build_graph().invoke(new_state(question, session_mode="demo", session_id=None))
    return state["route"], state.get("answer"), state.get("clarification_question")


def main():
    settings = get_settings()
    if "localhost" in settings.ollama_base_url or "127.0.0.1" in settings.ollama_base_url:
        print(f"REFUSING TO RUN: OLLAMA_BASE_URL is still pointed at localhost "
              f"({settings.ollama_base_url!r}).", file=sys.stderr)
        sys.exit(1)

    install_usage_tracker()
    budget_state["spent_usd"] = PRIOR_SPEND_USD
    print(f"Target: {settings.ollama_base_url} model={settings.ollama_model}", file=sys.stderr)
    print(f"Seeded cumulative spend: ${PRIOR_SPEND_USD:.6f} (budget cap $0.25)", file=sys.stderr)

    results = []
    stopped_early = False

    checks = [
        ("1-payment-original", "What payment method was used on the Sample Coffee Co. receipt?",
         lambda: _run_agent("What payment method was used on the Sample Coffee Co. receipt?"),
         "insufficient_evidence", None),
        ("2-payment-paraphrase-previously-failing", "How was the purchase at Sample Coffee Co. paid for?",
         lambda: _run_agent("How was the purchase at Sample Coffee Co. paid for?"),
         "insufficient_evidence", None),
        ("3-payment-paraphrase-fresh-a", "Did the Sample Coffee Co. customer pay in cash or by card?",
         lambda: _run_agent("Did the Sample Coffee Co. customer pay in cash or by card?"),
         "insufficient_evidence", None),
        ("4-payment-paraphrase-fresh-b", "What was the method of payment at Sample Coffee Co.?",
         lambda: _run_agent("What was the method of payment at Sample Coffee Co.?"),
         "insufficient_evidence", None),
        ("6-regression-total-and-citation", "What is the total amount on the Example Grocers Inc. receipt?",
         lambda: _run_agent("What is the total amount on the Example Grocers Inc. receipt?"),
         "answer", "42.17"),
        ("7-regression-clarify", "What is the tax?",
         lambda: _run_agent("What is the tax?"),
         "clarify", None),
        ("8-different-domain-absent-attribute", "What loyalty rewards discount was applied at Example Grocers Inc.?",
         lambda: _run_agent("What loyalty rewards discount was applied at Example Grocers Inc.?"),
         "insufficient_evidence", None),
    ]

    for label, question, fn, expect_route, expect_contains in checks:
        try:
            results.append(run_check(label, question, fn, expect_route, expect_contains))
        except BudgetExceeded as exc:
            print(f"BUDGET EXCEEDED, stopping: {exc}", file=sys.stderr)
            stopped_early = True
            break

    # Check 5: a synthetic receipt that EXPLICITLY states a payment method
    # -- same "patch execute_search_documents with fabricated test text"
    # technique tests/test_agent.py already uses for injection testing.
    # Confirms the gate still passes real, supported answers.
    if not stopped_early:
        synthetic_text = (
            "SYNTHETIC TEST RECEIPT -- GENERIC MART\n456 Fabricated Lane, Testville, ST 00000\n"
            "Date: 05/01/2026\nWidget 10.00\nGadget 15.00\nPayment Method: Credit Card\nTOTAL 25.00"
        )
        label = "5-supported-answer-synthetic-receipt-with-payment-method"
        question = "What payment method was used on this synthetic test receipt?"
        mark = len(call_log)
        try:
            with patch.object(agent_graph, "execute_search_documents",
                               return_value=[{"chunk_id": "synthetic_payment_test", "source": "synthetic_test_fixture",
                                               "text": synthetic_text, "score": 1.0}]):
                route, answer, detail = _run_agent(question)
            passed = route == "answer" and "credit card" in (answer or "").lower()
            error = None
        except BudgetExceeded as exc:
            print(f"BUDGET EXCEEDED, stopping: {exc}", file=sys.stderr)
            stopped_early = True
            route, answer, detail, passed, error = None, None, None, False, str(exc)
        if not stopped_early or error:
            calls = _calls_since(mark)
            result = {
                "label": label, "question": question, "expected_route": "answer", "expected_contains": "credit card",
                "observed_route": route, "observed_answer": answer, "passed": passed, "detail": detail, "error": error,
                "model_call_count": len(calls), "calls": calls,
                "total_latency_ms": sum(c["latency_ms"] for c in calls),
                "total_prompt_tokens": sum(c["prompt_tokens"] or 0 for c in calls),
                "total_completion_tokens": sum(c["completion_tokens"] or 0 for c in calls),
                "total_estimated_cost_usd": round(sum(c["estimated_cost_usd"] for c in calls), 6),
            }
            print(f"[{label}] passed={passed} route={route} calls={len(calls)} "
                  f"cost=${result['total_estimated_cost_usd']:.6f} :: answer={answer!r} detail={detail}")
            results.append(result)

    report = {
        "provider": settings.ollama_base_url, "model": settings.ollama_model, "budget_usd": 0.25,
        "prior_spend_usd": PRIOR_SPEND_USD, "total_spent_usd": round(budget_state["spent_usd"], 6),
        "spent_this_run_usd": round(budget_state["spent_usd"] - PRIOR_SPEND_USD, 6),
        "stopped_early_for_budget": stopped_early, "results": results,
    }
    out_path = EVAL_DIR / "results" / "together_evidence_support_verification.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nCumulative spend (whole engagement): ${report['total_spent_usd']:.6f} of $0.25 budget.")
    print(f"Spent this run: ${report['spent_this_run_usd']:.6f}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
