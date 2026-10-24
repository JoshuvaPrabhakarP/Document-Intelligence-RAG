"""Reads eval/results/raw_results.json and prints the aggregate comparison
tables used in eval/EVALUATION_REPORT.md. Pure read/aggregate -- never
re-runs or re-scores anything, so re-running this script never changes a
result, only how it's summarized."""

import json
import statistics
from pathlib import Path

RESULTS = json.loads((Path(__file__).parent / "results" / "raw_results.json").read_text())


def held_out(cases):
    return [c for c in cases if c["split"] == "held_out"]


def pct(n, d):
    return f"{100*n/d:.0f}%" if d else "n/a"


def main():
    cases = held_out(RESULTS["cases"])
    print(f"Held-out cases: {len(cases)}\n")

    # ---- Route correctness ----
    print("== Route correctness (held-out only) ==")
    agent_route_ok = sum(1 for c in cases if c["scoring"].get("agent_route_correct"))
    print(f"Agent route correct: {agent_route_ok}/{len(cases)} ({pct(agent_route_ok, len(cases))})")
    non_ambig = [c for c in cases if c["category"] != "ambiguous"]
    std_route_ok = sum(1 for c in non_ambig if c["scoring"].get("standard_route_correct"))
    print(f"Standard route(analog) correct (non-ambiguous only, n={len(non_ambig)}): {std_route_ok}/{len(non_ambig)} ({pct(std_route_ok, len(non_ambig))})")
    ambig = [c for c in cases if c["category"] == "ambiguous"]
    for c in ambig:
        print(f"  [{c['id']}] standard bare -> route_analog={c['standard']['route_analog']!r} answer={c['standard']['answer']!r} (descriptive only, not scored)")
        print(f"        standard resolved -> correct={c['standard_resolved']['answer_correct']} answer={c['standard_resolved']['answer']!r}")
        print(f"        agent clarify -> route_correct={c['scoring']['agent_route_correct']} followup_correct={c['scoring']['agent_followup_route_correct']}")

    print()
    # ---- Answer correctness (only where expected_answer_facts non-empty, i.e. answer cases) ----
    print("== Answer correctness (held-out, route=answer cases only) ==")
    answerable = [c for c in cases if c["category"] in ("answerable", "adversarial")]
    std_correct = sum(1 for c in answerable if c["scoring"].get("standard_answer_correct"))
    agent_correct = sum(1 for c in answerable if c["scoring"].get("agent_answer_correct"))
    print(f"Standard: {std_correct}/{len(answerable)} ({pct(std_correct, len(answerable))})")
    print(f"Agent:    {agent_correct}/{len(answerable)} ({pct(agent_correct, len(answerable))})")
    for c in answerable:
        print(f"  [{c['id']}] standard_correct={c['scoring'].get('standard_answer_correct')} agent_correct={c['scoring'].get('agent_answer_correct')}"
              f" | std_ans={c['standard']['answer']!r} | agent_ans={c['agent']['answer']!r}")

    print()
    # ---- Adversarial: forbidden-content check ----
    print("== Adversarial injection resistance (held-out) ==")
    adversarial = [c for c in cases if c["category"] == "adversarial"]
    for c in adversarial:
        print(f"  [{c['id']}] standard_forbidden_ok={c['scoring'].get('standard_forbidden_ok')} agent_forbidden_ok={c['scoring'].get('agent_forbidden_ok')}")

    print()
    # ---- Citation validity ----
    print("== Citation validity (held-out, route=answer cases only) ==")
    cite_cases = [c for c in cases if c["scoring"].get("standard_citation_valid") is not None]
    std_cite_ok = sum(1 for c in cite_cases if c["scoring"].get("standard_citation_valid"))
    agent_cite_cases = [c for c in cases if c["scoring"].get("agent_citation_valid") is not None]
    agent_cite_ok = sum(1 for c in agent_cite_cases if c["scoring"].get("agent_citation_valid"))
    print(f"Standard: {std_cite_ok}/{len(cite_cases)} ({pct(std_cite_ok, len(cite_cases))})")
    print(f"Agent:    {agent_cite_ok}/{len(agent_cite_cases)} ({pct(agent_cite_ok, len(agent_cite_cases))})")

    print()
    # ---- Abstention correctness (category=absent) ----
    print("== Correct abstention (held-out, category=absent) ==")
    absent = [c for c in cases if c["category"] == "absent"]
    std_abstain_ok = sum(1 for c in absent if c["standard"]["route_analog"] == "insufficient_evidence")
    agent_abstain_ok = sum(1 for c in absent if c["agent"]["route"] == "insufficient_evidence")
    print(f"Standard correctly abstained: {std_abstain_ok}/{len(absent)}")
    print(f"Agent correctly abstained:    {agent_abstain_ok}/{len(absent)}")
    for c in absent:
        print(f"  [{c['id']}] std_route_analog={c['standard']['route_analog']} agent_route={c['agent']['route']}")

    print()
    # ---- Latency / calls ----
    print("== Latency (wall-clock, ms) — held-out main run (single pass, not variability) ==")
    std_lat = [c["standard"]["wall_clock_ms"] for c in cases]
    agent_lat = [c["agent"]["wall_clock_ms"] for c in cases]
    print(f"Standard: mean={statistics.mean(std_lat):.0f} median={statistics.median(std_lat):.0f} min={min(std_lat):.0f} max={max(std_lat):.0f}")
    print(f"Agent:    mean={statistics.mean(agent_lat):.0f} median={statistics.median(agent_lat):.0f} min={min(agent_lat):.0f} max={max(agent_lat):.0f}")

    print()
    print("== Agent tool/model call counts (held-out) ==")
    tool_calls = [c["agent"]["tool_calls"] for c in cases]
    model_calls = [c["agent"]["model_calls"] for c in cases]
    print(f"tool_calls: mean={statistics.mean(tool_calls):.1f} min={min(tool_calls)} max={max(tool_calls)}")
    print(f"model_calls: mean={statistics.mean(model_calls):.1f} min={min(model_calls)} max={max(model_calls)}")

    print()
    # ---- Errors ----
    print("== Errors/timeouts/limit hits (held-out) ==")
    for c in cases:
        if c["standard"].get("error") or c["agent"].get("error"):
            print(f"  [{c['id']}] standard_error={c['standard'].get('error')!r} agent_error={c['agent'].get('error')!r}")
    if not any(c["standard"].get("error") or c["agent"].get("error") for c in cases):
        print("  none")

    print()
    # ---- Variability ----
    print("== Variability summary ==")
    print(json.dumps(RESULTS["variability_summary"], indent=2))


if __name__ == "__main__":
    main()
