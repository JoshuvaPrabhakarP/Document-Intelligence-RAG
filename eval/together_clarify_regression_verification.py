"""
Follow-up to the container smoke test, which found the agent's
clarification-follow-up check failing to reproduce in 4/4 real attempts
(see PROGRESS.md). Investigation (comparing prompts across commits, not
reproduced here) found two real issues in app/agent/llm.py's route
criteria, introduced/inherited across two earlier rounds, not in this
round's container build itself:
  1. The evidence-support-gate round's "insufficient_evidence" wording
     ("...OR the right document was found but it simply doesn't state
     the specific fact asked about") was broad enough to compete with
     "clarify" for a question that names NO document at all -- a model
     could satisfy itself that "no document states 'the tax' in the
     abstract" without ever checking whether the question should instead
     have triggered "clarify" for not naming a document.
  2. The routing-fix round's "named vendor overrides candidate noise"
     wording assumed naming a vendor always resolves to exactly one
     document -- false when more than one retrieved candidate shares
     that vendor (e.g. two receipts from the same vendor on different
     dates).

Both fixed with a single corrected rule in app/agent/llm.py (mirrored in
scripts/llm_client.py for Standard RAG): clarify applies whenever the
question does not single out EXACTLY ONE retrieved candidate -- whether
because it names no document, or because what it names still matches
more than one candidate; insufficient_evidence only applies once exactly
one candidate IS singled out.

This script verifies the fix with a small FIXED set of 5 real-call
scenarios, each run 3 times, reporting every individual result -- not
rerun until green. Run via the local venv process (NOT inside Docker --
the fix is a pure prompt-text change in files already covered by the
deterministic test suite; Docker vs bare venv doesn't change model
behavior, and the container build itself is accepted/not being
re-verified per instruction, so no rebuild was done for this check).

Reuses eval/together_compat_test.py's budget tracker, seeded with this
engagement's last reported cumulative estimate.
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

# Last reported cumulative ESTIMATE across this whole engagement (the
# container smoke-test calls weren't individually tracked, so this is
# carried forward as an estimate, not a fresh assumption -- see
# PROGRESS.md's Milestone 5 container-smoke-test section).
PRIOR_SPEND_USD = 0.0167

REPEATS = 3


def _calls_since(mark: int) -> list[dict]:
    return call_log[mark:]


def run_once(run_fn) -> dict:
    mark = len(call_log)
    try:
        route, detail = run_fn()
        error = None
    except BudgetExceeded:
        raise
    except Exception as exc:  # noqa: BLE001
        route, detail, error = None, None, str(exc)
    calls = _calls_since(mark)
    return {
        "route": route, "detail": detail, "error": error,
        "model_call_count": len(calls),
        "total_latency_ms": sum(c["latency_ms"] for c in calls),
        "total_estimated_cost_usd": round(sum(c["estimated_cost_usd"] for c in calls), 6),
    }


def main():
    settings = get_settings()
    if "localhost" in settings.ollama_base_url or "127.0.0.1" in settings.ollama_base_url:
        print(f"REFUSING TO RUN: OLLAMA_BASE_URL is still localhost ({settings.ollama_base_url!r}).", file=sys.stderr)
        sys.exit(1)

    install_usage_tracker()
    budget_state["spent_usd"] = PRIOR_SPEND_USD
    pipeline = RAGPipeline(settings)
    print(f"Target: {settings.ollama_base_url} model={settings.ollama_model}", file=sys.stderr)
    print(f"Seeded cumulative spend: ${PRIOR_SPEND_USD:.6f} (budget cap $0.25)", file=sys.stderr)

    def q_ambiguous_no_document():
        r = pipeline.answer("What is the total?", top_k=5)
        return r["route"], f"answer={r['answer']!r} clarification_question={r['clarification_question']!r}"

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
        from app.agent.state import resume_state
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

    scenarios = [
        ("A-ambiguous-no-document", "expect clarify", q_ambiguous_no_document),
        ("B-explicit-unique-receipt", "expect answer/156.42", q_explicit_unique),
        ("C-missing-info-named-document", "expect insufficient_evidence", q_missing_info_named_doc),
        ("D-clarify-followup-roundtrip", "expect clarify then resumed answer/8.00", q_clarify_followup),
        ("E-shared-vendor-synthetic", "expect clarify (NEW fix)", q_shared_vendor_synthetic),
    ]

    all_results = {}
    stopped_early = False
    for label, expectation, fn in scenarios:
        all_results[label] = {"expectation": expectation, "runs": []}
        for i in range(REPEATS):
            try:
                result = run_once(fn)
            except BudgetExceeded as exc:
                print(f"BUDGET EXCEEDED, stopping: {exc}", file=sys.stderr)
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
    out_path = EVAL_DIR / "results" / "together_clarify_regression_verification.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nCumulative spend: ${report['total_spent_usd']:.6f} of $0.25. Spent this run: ${report['spent_this_run_usd']:.6f}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
