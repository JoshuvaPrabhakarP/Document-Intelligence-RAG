"""Continuation of together_clarify_regression_verification.py -- the
first invocation was killed by an outer 280s timeout after completing
A (3/3 runs) and B (2/3 runs). This resumes from B run 3 through E,
seeded with the exact cumulative spend after those 5 completed calls
(0.0167 + 0.000158+0.000179+0.000158+0.000725+0.000744 = 0.018664),
and appends into the SAME results file rather than overwriting it."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import graph as agent_graph  # noqa: E402
from app.agent.state import new_state, resume_state  # noqa: E402
from app.config import get_settings  # noqa: E402
from eval.together_compat_test import BudgetExceeded, budget_state, call_log, install_usage_tracker  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
PRIOR_SPEND_USD = 0.018664
REPEATS = 3


def _calls_since(mark):
    return call_log[mark:]


def run_once(run_fn):
    mark = len(call_log)
    try:
        route, detail = run_fn()
        error = None
    except BudgetExceeded:
        raise
    except Exception as exc:  # noqa: BLE001
        route, detail, error = None, None, str(exc)
    calls = _calls_since(mark)
    return {"route": route, "detail": detail, "error": error,
            "model_call_count": len(calls),
            "total_latency_ms": sum(c["latency_ms"] for c in calls),
            "total_estimated_cost_usd": round(sum(c["estimated_cost_usd"] for c in calls), 6)}


def main():
    settings = get_settings()
    install_usage_tracker()
    budget_state["spent_usd"] = PRIOR_SPEND_USD
    print(f"Seeded cumulative spend: ${PRIOR_SPEND_USD:.6f}", file=sys.stderr)

    def q_explicit_unique():
        state = agent_graph.build_graph().invoke(
            new_state("What is the total at Fictional Hardware Supply?", session_mode="demo", session_id=None))
        return state["route"], f"answer={state.get('answer')!r}"

    def q_missing_info_named_doc():
        state = agent_graph.build_graph().invoke(
            new_state("What payment method was used on the Sample Coffee Co. receipt?", session_mode="demo", session_id=None))
        return state["route"], f"answer={state.get('answer')!r}"

    def q_clarify_followup():
        state1 = agent_graph.build_graph().invoke(new_state("What is the tax?", session_mode="demo", session_id=None))
        if state1["route"] != "clarify":
            return state1["route"], f"(did not reach clarify) clarification_question={state1.get('clarification_question')!r} answer={state1.get('answer')!r}"
        state2 = agent_graph.build_graph().invoke(resume_state(
            original_question="What is the tax?",
            clarification_question=state1["clarification_question"] or "",
            clarification_answer="Fictional Hardware Supply",
            prior_messages=state1["messages"], prior_retrieved_passages=state1["retrieved_passages"],
            session_mode="demo", session_id=None,
        ))
        return state2["route"], f"resumed answer={state2.get('answer')!r}"

    def q_shared_vendor_synthetic():
        synthetic_passages = [
            {"chunk_id": "synthetic_shared_a", "source": "synthetic_test_fixture",
             "text": "SAMPLE COFFEE CO.\nDate: 01/02/2026\nTOTAL 9.75", "score": 1.0},
            {"chunk_id": "synthetic_shared_b", "source": "synthetic_test_fixture",
             "text": "SAMPLE COFFEE CO.\nDate: 03/09/2026\nTOTAL 14.50", "score": 0.95},
        ]
        with patch.object(agent_graph, "execute_search_documents", return_value=synthetic_passages):
            state = agent_graph.build_graph().invoke(
                new_state("What is the total at Sample Coffee Co.?", session_mode="demo", session_id=None))
        return state["route"], f"clarification_question={state.get('clarification_question')!r} answer={state.get('answer')!r}"

    out_path = EVAL_DIR / "results" / "together_clarify_regression_verification.json"
    existing = json.loads(out_path.read_text())
    all_results = existing["results"]

    scenarios = [
        ("B-explicit-unique-receipt", q_explicit_unique, 1),  # 1 remaining run
        ("C-missing-info-named-document", q_missing_info_named_doc, REPEATS),
        ("D-clarify-followup-roundtrip", q_clarify_followup, REPEATS),
        ("E-shared-vendor-synthetic", q_shared_vendor_synthetic, REPEATS),
    ]

    stopped_early = False
    for label, fn, n in scenarios:
        all_results.setdefault(label, {"expectation": "", "runs": []})
        for i in range(n):
            try:
                result = run_once(fn)
            except BudgetExceeded as exc:
                print(f"BUDGET EXCEEDED: {exc}", file=sys.stderr)
                stopped_early = True
                break
            print(f"[{label}] route={result['route']} cost=${result['total_estimated_cost_usd']:.6f} :: {result['detail']}")
            all_results[label]["runs"].append(result)
        if stopped_early:
            break

    report = {
        "provider": settings.ollama_base_url, "model": settings.ollama_model, "budget_usd": 0.25,
        "prior_spend_usd": existing["prior_spend_usd"], "total_spent_usd": round(budget_state["spent_usd"], 6),
        "spent_this_run_usd": round(budget_state["spent_usd"] - existing["prior_spend_usd"], 6),
        "stopped_early_for_budget": stopped_early, "results": all_results,
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nCumulative spend: ${report['total_spent_usd']:.6f} of $0.25. Spent total this investigation: ${report['spent_this_run_usd']:.6f}")


if __name__ == "__main__":
    main()
