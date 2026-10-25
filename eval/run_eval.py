"""
Milestone 4 evaluation harness — runs every case in eval/cases.json against
BOTH Standard RAG (workflow="standard") and the Agent workflow
(workflow="agent"), through the real FastAPI app (fastapi.testclient.TestClient),
exactly as a real client would call it. No internal function is called
directly for the systems under test, so this measures the actual HTTP
behavior, not a shortcut around it.

Design notes (documented here, not just in the report, since this script
IS the methodology):

- Each case gets its own fresh TestClient (its own cookie jar), so no case
  can see another case's session state. Within one case, the SAME
  TestClient/session is reused for the standard-workflow call, the
  agent-workflow call, and (for upload-mode cases) the upload step, which
  is what "same documents and questions" (requirement 3) means in practice.
- Standard RAG has no clarification mechanism. For "ambiguous" category
  cases, this script runs Standard RAG twice: once on the bare ambiguous
  question (recorded as a qualitative/descriptive result, NOT scored
  pass/fail against "clarify" -- that route is structurally impossible for
  Standard RAG), and once on a RESOLVED question built by a single fixed
  template (`_resolve_question`) that appends the clarification follow-up
  to the original question. The resolved-question run IS scored normally
  (it's just an ordinary answerable question once disambiguated). This is
  the fairest literal interpretation of "provide equivalent follow-up
  information" for a system that cannot ask for it itself.
- Scoring is 100% automated substring/exact-match checks against the
  frozen expected_* fields in cases.json -- there is no LLM-judge step and
  no manual review folded into the pass/fail numbers. expected_answer_facts
  (and expected_followup_facts) are OR'd: the answer is correct if ANY ONE
  of the listed acceptable phrasings appears (case-insensitive substring),
  not all of them -- they are alternative acceptable phrasings, not
  required components.
- citation_valid is only scored when the expected route is "answer" --
  there is nothing meaningful to validate a citation against when the
  system is (correctly) abstaining.
- Cold start is paid by one throwaway warm-up call per system/mode
  BEFORE the timed loop, excluded from all aggregate latency numbers.
- A representative subset of held-out cases is repeated 3x each
  (--repeat-cases) to check variability; repeats are recorded individually,
  never averaged into the single-run headline numbers.

Run: venv/bin/python3 eval/run_eval.py [--repeat-subset]
Output: eval/results/raw_results.json, eval/results/config_snapshot.json
"""

import argparse
import io
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pymupdf
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.main import app  # noqa: E402
from app.upload.session import sessions  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
REPEAT_SUBSET_IDS = ["H1", "H7", "H10", "H12"]  # one per category: answerable, absent, ambiguous, adversarial
REPEATS = 3


def make_test_pdf(text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    return doc.tobytes()


def _resolve_question(question: str, followup: str) -> str:
    """The ONE fixed template used for every ambiguous case's 'resolved'
    variant -- defined once, applied uniformly, never adjusted per case."""
    return f"{question} Specifically, I mean: {followup}."


def load_cases(cases_path: Path) -> list[dict]:
    data = json.loads(cases_path.read_text())
    return data["cases"]


def upload_docs(client: TestClient, docs: list[dict]) -> None:
    for doc in docs:
        pdf_bytes = make_test_pdf(doc["text"])
        resp = client.post("/upload", files={"file": (doc["filename"], pdf_bytes, "application/pdf")})
        assert resp.status_code == 200, f"setup failed for {doc['filename']}: {resp.text}"


def call_query(client: TestClient, question: str, mode: str, workflow: str) -> tuple[dict, float]:
    t0 = time.monotonic()
    resp = client.post("/query", json={"question": question, "mode": mode, "workflow": workflow})
    elapsed_ms = (time.monotonic() - t0) * 1000
    assert resp.status_code == 200, f"query failed: {resp.status_code} {resp.text}"
    return resp.json(), elapsed_ms


def score_answer(answer: str | None, expected_facts: list[str]) -> bool | None:
    if not expected_facts:
        return None  # nothing to check (e.g. an abstention case)
    if not answer:
        return False
    answer_l = answer.lower()
    return any(fact.lower() in answer_l for fact in expected_facts)


def score_forbidden(answer: str | None, forbidden: list[str]) -> bool | None:
    if not forbidden:
        return None
    if not answer:
        return True
    answer_l = answer.lower()
    return not any(f.lower() in answer_l for f in forbidden)


def score_citation(sources: list[dict], expected_sources: list[str], expected_route: str) -> bool | None:
    if expected_route != "answer" or not expected_sources:
        return None  # citation validity only meaningful when an answer is expected
    cited = [(s.get("chunk_id") or "", s.get("source") or "") for s in (sources or [])]
    for exp in expected_sources:
        for chunk_id, source in cited:
            if exp.lower() in chunk_id.lower() or exp.lower() in source.lower():
                return True
    return False


def agent_route_analog(standard_result: dict) -> str | None:
    hse = standard_result.get("has_sufficient_evidence")
    if hse is True:
        return "answer"
    if hse is False:
        return "insufficient_evidence"
    return None  # a generation error -- no analog route


def tool_and_model_call_counts(execution_trace: list[dict]) -> dict:
    tool_calls = sum(1 for t in execution_trace if t.get("node") == "tools" and t.get("tool"))
    model_calls = sum(1 for t in execution_trace if t.get("node") in ("agent", "assess_evidence"))
    return {"tool_calls": tool_calls, "model_calls": model_calls}


def run_case(case: dict) -> dict:
    client = TestClient(app)
    mode = case["mode"]
    out = {"id": case["id"], "split": case["split"], "category": case["category"], "mode": mode}

    if mode == "upload":
        upload_docs(client, case["upload_docs"])

    # ---- Standard RAG ----
    std_body, std_ms = call_query(client, case["question"], mode, "standard")
    out["standard"] = {
        "answer": std_body["answer"],
        "sources": std_body["sources"],
        "has_sufficient_evidence": std_body["has_sufficient_evidence"],
        "route_analog": agent_route_analog(std_body),
        "wall_clock_ms": std_ms,
        "retrieval_latency_ms": std_body["retrieval_latency_ms"],
        "rerank_latency_ms": std_body["rerank_latency_ms"],
        "generation_latency_ms": std_body["generation_latency_ms"],
        "error": std_body["error"],
    }

    # ---- Agent workflow ----
    agent_body, agent_ms = call_query(client, case["question"], mode, "agent")
    agent_counts = tool_and_model_call_counts(agent_body.get("execution_trace") or [])
    out["agent"] = {
        "answer": agent_body["answer"],
        "sources": agent_body["sources"],
        "route": agent_body["route"],
        "clarification_question": agent_body["clarification_question"],
        "wall_clock_ms": agent_ms,
        "error": agent_body["error"],
        "execution_trace": agent_body["execution_trace"],
        **agent_counts,
    }

    category = case["category"]

    if category == "ambiguous":
        followup = case["clarification_followup"]

        # Standard RAG has no clarification step. Record the bare-question
        # result as DESCRIPTIVE ONLY (not scored against "clarify" -- that
        # route doesn't exist for this system), then run the one fixed
        # resolved-question template and score THAT normally.
        resolved_q = _resolve_question(case["question"], followup)
        std_resolved_body, std_resolved_ms = call_query(client, resolved_q, mode, "standard")
        out["standard_resolved"] = {
            "question_used": resolved_q,
            "answer": std_resolved_body["answer"],
            "sources": std_resolved_body["sources"],
            "has_sufficient_evidence": std_resolved_body["has_sufficient_evidence"],
            "route_analog": agent_route_analog(std_resolved_body),
            "wall_clock_ms": std_resolved_ms,
            "answer_correct": score_answer(std_resolved_body["answer"], case.get("expected_followup_facts", [])),
            "citation_valid": score_citation(std_resolved_body["sources"], case.get("expected_followup_sources", []),
                                              case.get("expected_followup_route", "answer")),
        }

        # Agent: resume with the SAME follow-up text a real user would type.
        agent_followup_body, agent_followup_ms = call_query(client, followup, mode, "agent")
        agent_followup_counts = tool_and_model_call_counts(agent_followup_body.get("execution_trace") or [])
        out["agent_followup"] = {
            "answer": agent_followup_body["answer"],
            "sources": agent_followup_body["sources"],
            "route": agent_followup_body["route"],
            "wall_clock_ms": agent_followup_ms,
            "error": agent_followup_body["error"],
            "answer_correct": score_answer(agent_followup_body["answer"], case.get("expected_followup_facts", [])),
            "citation_valid": score_citation(agent_followup_body["sources"], case.get("expected_followup_sources", []),
                                              case.get("expected_followup_route", "answer")),
            **agent_followup_counts,
        }

        out["scoring"] = {
            "standard_bare_route_note": "not scored -- Standard RAG cannot clarify; see standard.route_analog / standard.answer for descriptive result only",
            "agent_route_correct": agent_body["route"] == case["expected_route"],
            "agent_followup_route_correct": agent_followup_body["route"] == case.get("expected_followup_route"),
        }
    else:
        out["scoring"] = {
            "standard_route_correct": out["standard"]["route_analog"] == agent_route_analog_for_expected(case["expected_route"]),
            "standard_answer_correct": score_answer(std_body["answer"], case.get("expected_answer_facts", [])),
            "standard_forbidden_ok": score_forbidden(std_body["answer"], case.get("forbidden_in_answer", [])),
            "standard_citation_valid": score_citation(std_body["sources"], case.get("expected_sources", []), case["expected_route"]),
            "agent_route_correct": agent_body["route"] == case["expected_route"],
            "agent_answer_correct": score_answer(agent_body["answer"], case.get("expected_answer_facts", [])),
            "agent_forbidden_ok": score_forbidden(agent_body["answer"], case.get("forbidden_in_answer", [])),
            "agent_citation_valid": score_citation(agent_body["sources"], case.get("expected_sources", []), case["expected_route"]),
        }

    if mode == "upload":
        client.delete("/session")  # cleanup -- never leave synthetic uploads sitting in /tmp between cases

    return out


def agent_route_analog_for_expected(expected_route: str) -> str | None:
    # expected_route is already one of "answer"/"insufficient_evidence" for
    # non-ambiguous categories (clarify is handled in its own branch above).
    return expected_route


def warm_up():
    """Pays cold-start model-load costs (embedding model, reranker, first
    Ollama call, upload-mode's separate embedding device) OUTSIDE the timed
    loop -- same cold/warm distinction this project has used since
    Milestone 1."""
    print("Warming up (cold-start costs paid here, excluded from results)...", file=sys.stderr)
    client = TestClient(app)
    t0 = time.monotonic()
    call_query(client, "What is the total amount on the Example Grocers Inc. receipt?", "demo", "standard")
    call_query(client, "What is the total amount on the Example Grocers Inc. receipt?", "demo", "agent")
    upload_docs(client, [{"filename": "warmup.pdf", "text": "WARMUP VENDOR\nTOTAL 1.00"}])
    call_query(client, "What is the total?", "upload", "standard")
    call_query(client, "What is the total?", "upload", "agent")
    client.delete("/session")
    cold_ms = (time.monotonic() - t0) * 1000
    print(f"Warm-up complete in {cold_ms:.0f}ms.", file=sys.stderr)
    return cold_ms


def ollama_version() -> str | None:
    try:
        resp = httpx.get("http://localhost:11434/api/version", timeout=3.0)
        return resp.json().get("version")
    except Exception:
        return None


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=EVAL_DIR.parent).decode().strip()
    except Exception:
        return None


def build_config_snapshot(cold_ms: float) -> dict:
    settings = get_settings()
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": git_commit(),
        "app_config": settings.app_config,
        "ollama_model": settings.ollama_model,
        "ollama_version": ollama_version(),
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "use_reranker": settings.use_reranker,
        "retrieval_pool_size": settings.retrieval_pool_size,
        "top_k": settings.top_k,
        "cold_start_warmup_ms": cold_ms,
    }


def run_variability_subset(cases_by_id: dict, repeat_ids: list[str]) -> list[dict]:
    print(f"Running variability subset {repeat_ids} x{REPEATS}...", file=sys.stderr)
    runs = []
    for case_id in repeat_ids:
        case = cases_by_id[case_id]
        for rep in range(1, REPEATS + 1):
            print(f"  {case_id} repeat {rep}/{REPEATS}", file=sys.stderr)
            result = run_case(case)
            result["repeat"] = rep
            runs.append(result)
    return runs


def summarize_variability(runs: list[dict]) -> dict:
    by_case: dict[str, list[dict]] = {}
    for r in runs:
        by_case.setdefault(r["id"], []).append(r)

    summary = {}
    for case_id, reps in by_case.items():
        std_latencies = [r["standard"]["wall_clock_ms"] for r in reps]
        agent_latencies = [r["agent"]["wall_clock_ms"] for r in reps]
        std_routes = {r["standard"]["route_analog"] for r in reps}
        agent_routes = {r["agent"]["route"] for r in reps}
        summary[case_id] = {
            "standard_wall_clock_ms": {"values": std_latencies, "mean": statistics.mean(std_latencies),
                                        "stdev": statistics.stdev(std_latencies) if len(std_latencies) > 1 else 0.0},
            "agent_wall_clock_ms": {"values": agent_latencies, "mean": statistics.mean(agent_latencies),
                                     "stdev": statistics.stdev(agent_latencies) if len(agent_latencies) > 1 else 0.0},
            "standard_route_consistent": len(std_routes) == 1,
            "agent_route_consistent": len(agent_routes) == 1,
            "standard_routes_seen": sorted(std_routes),
            "agent_routes_seen": sorted(agent_routes),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-variability", action="store_true")
    parser.add_argument("--cases", default=str(EVAL_DIR / "cases.json"),
                         help="Path to the cases JSON file (default: eval/cases.json, Milestone 4's original set)")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR),
                         help="Where to write raw_results.json/config_snapshot.json (default: eval/results, Milestone 4's original location)")
    parser.add_argument("--repeat-ids", nargs="*", default=REPEAT_SUBSET_IDS,
                         help="Case IDs to repeat for the variability subset (default: Milestone 4's original four)")
    args = parser.parse_args()

    cases_path = Path(args.cases)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    cases = load_cases(cases_path)
    cases_by_id = {c["id"]: c for c in cases}

    cold_ms = warm_up()
    config_snapshot = build_config_snapshot(cold_ms)
    (results_dir / "config_snapshot.json").write_text(json.dumps(config_snapshot, indent=2))
    print(f"Config snapshot saved. ollama_version={config_snapshot['ollama_version']}", file=sys.stderr)

    results = []
    for case in cases:
        print(f"Running {case['id']} ({case['split']}/{case['category']}/{case['mode']})...", file=sys.stderr)
        results.append(run_case(case))

    # Save the primary results NOW, before the optional variability subset --
    # a crash during variability (e.g. GPU memory exhaustion after many
    # sequential real-model calls in one long process, observed once during
    # Milestone 4's post-fix re-run) must not lose work that already
    # finished successfully. The file is overwritten below if variability
    # also completes; if it doesn't, this is what's left on disk, which is
    # still a complete, valid primary-results file.
    out_path = results_dir / "raw_results.json"
    out_path.write_text(json.dumps({
        "config_snapshot": config_snapshot, "cases": results,
        "variability_runs": [], "variability_summary": {},
    }, indent=2))
    print(f"Primary results saved to {out_path} ({len(results)} cases) before attempting variability.", file=sys.stderr)

    variability_runs = []
    if not args.skip_variability:
        try:
            variability_runs = run_variability_subset(cases_by_id, args.repeat_ids)
        except Exception as exc:  # noqa: BLE001 -- primary results above must survive this either way
            print(f"Variability subset failed ({exc!r}); primary results already saved are unaffected.", file=sys.stderr)
    variability_summary = summarize_variability(variability_runs) if variability_runs else {}

    out_path.write_text(json.dumps({
        "config_snapshot": config_snapshot,
        "cases": results,
        "variability_runs": variability_runs,
        "variability_summary": variability_summary,
    }, indent=2))

    print(f"Done. {len(results)} cases run, {len(variability_runs)} variability runs. Results saved to {out_path}", file=sys.stderr)
    # Leftover upload sessions (demo-mode cases set a session cookie too,
    # for agent-workflow clarification state) are per-TestClient in-memory
    # cookie jars that vanish with the process -- but the server-side
    # SessionManager entries and on-disk session dirs they created are not
    # swept until TTL expiry, so clear them explicitly here.
    for sid in list(sessions._sessions.keys()):
        sessions.clear_session(sid)


if __name__ == "__main__":
    main()
