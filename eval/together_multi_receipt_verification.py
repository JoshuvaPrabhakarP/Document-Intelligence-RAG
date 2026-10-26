"""
Final verification for the clarify-routing refinement: fixes the one
observed resume failure's investigation findings (preservation of the
original question and selected receipt was confirmed correct across
repeated diagnostics -- see PROGRESS.md; the ~20% resume miss rate is
real-model sampling variance in assess_evidence's own quote production,
not a wiring bug) and the new multi-receipt-answerable rule (clarify
must not fire for an intentional comparison/total across several
receipts).

Four scenarios, 3 real runs each, against the real Together endpoint.
Every attempt is reported, not rerun until green.
"""

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
PRIOR_SPEND_USD = 0.041871  # exact cumulative after this round's diagnostics (tracked precisely)
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
    if "localhost" in settings.ollama_base_url:
        print("REFUSING: localhost", file=sys.stderr)
        sys.exit(1)
    install_usage_tracker()
    budget_state["spent_usd"] = PRIOR_SPEND_USD
    print(f"Seeded cumulative spend: ${PRIOR_SPEND_USD:.6f}", file=sys.stderr)

    def q_followup():
        state1 = agent_graph.build_graph().invoke(new_state("What is the tax?", session_mode="demo", session_id=None))
        if state1["route"] != "clarify":
            return state1["route"], f"(did not reach clarify) answer={state1.get('answer')!r}"
        state2 = agent_graph.build_graph().invoke(resume_state(
            original_question="What is the tax?",
            clarification_question=state1["clarification_question"] or "",
            clarification_answer="Fictional Hardware Supply",
            prior_messages=state1["messages"], prior_retrieved_passages=state1["retrieved_passages"],
            session_mode="demo", session_id=None,
        ))
        return state2["route"], f"resumed answer={state2.get('answer')!r}"

    def q_shared_vendor():
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

    def q_unique_receipt():
        state = agent_graph.build_graph().invoke(
            new_state("What is the total at Fictional Hardware Supply?", session_mode="demo", session_id=None))
        return state["route"], f"answer={state.get('answer')!r}"

    def q_intentional_multi_receipt():
        state = agent_graph.build_graph().invoke(
            new_state("Which receipt has a higher total, Example Grocers Inc. or Mock Bookstore?",
                       session_mode="demo", session_id=None))
        return state["route"], f"answer={state.get('answer')!r} cited={state.get('cited_passage_ids')!r}"

    scenarios = [
        ("1-failed-followup-resume", q_followup, "expect clarify then answer/8.00"),
        ("2-shared-vendor-ambiguity", q_shared_vendor, "expect clarify"),
        ("3-uniquely-identified-receipt", q_unique_receipt, "expect answer/156.42"),
        ("4-intentional-multi-receipt", q_intentional_multi_receipt, "expect answer (NOT clarify), citing both"),
    ]

    all_results = {}
    stopped_early = False
    for label, fn, expectation in scenarios:
        all_results[label] = {"expectation": expectation, "runs": []}
        for i in range(REPEATS):
            try:
                result = run_once(fn)
            except BudgetExceeded as exc:
                print(f"BUDGET EXCEEDED: {exc}", file=sys.stderr)
                stopped_early = True
                break
            print(f"[{label} run {i+1}/{REPEATS}] route={result['route']} cost=${result['total_estimated_cost_usd']:.6f} :: {result['detail']}")
            all_results[label]["runs"].append(result)
        if stopped_early:
            break

    report = {
        "provider": settings.ollama_base_url, "model": settings.ollama_model, "budget_usd": 0.25,
        "prior_spend_usd": PRIOR_SPEND_USD, "total_spent_usd": round(budget_state["spent_usd"], 6),
        "spent_this_run_usd": round(budget_state["spent_usd"] - PRIOR_SPEND_USD, 6),
        "stopped_early_for_budget": stopped_early, "results": all_results,
    }
    out_path = EVAL_DIR / "results" / "together_multi_receipt_verification.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nCumulative spend: ${report['total_spent_usd']:.6f} of $0.25. Spent this run: ${report['spent_this_run_usd']:.6f}")


if __name__ == "__main__":
    main()
