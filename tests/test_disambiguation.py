"""
Deterministic tests for the document-ambiguity fix: a question that could
have a genuinely different correct answer depending on WHICH of several
retrieved documents it refers to must trigger a clarifying question
instead of silently picking one (Milestone 4's H10 finding). Covers the
four required scenarios for BOTH Standard RAG and the Agent:
  1. Two receipts sharing a date (distinct documents, same date field)
  2. Two receipts sharing a vendor (distinct documents, same vendor text)
  3. A uniquely identified receipt (one dominant match -- must NOT
     falsely trigger clarification; ordinary single-document questions
     must keep working)
  4. A clarification follow-up selecting one candidate (resolves to a
     normal answer, not another clarify)

All mocked/deterministic -- no real model calls, no network. The shared
detection logic (app/disambiguation.py) is unit-tested directly; Standard
RAG's and the Agent's own fixed-prefix/field protocol for turning that
signal into a route is tested by mocking only the generation/evidence-
assessment call, so the REAL wiring (app/rag_pipeline.py's
CLARIFICATION_NEEDED parsing, app/agent/graph.py's candidate computation)
runs for real.
"""

from contextlib import contextmanager
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from app.agent import graph as agent_graph
from app.agent.state import new_state
from app.disambiguation import cluster_candidate_documents, format_candidates
from app.main import app, pipeline
from app.upload.session import sessions

# ---------------------------------------------------------------------------
# 1-3: the shared detection function itself (both "sharing a date" and
# "sharing a vendor" reduce to the same question -- are these two
# DIFFERENT documents with comparable relevance? -- grouping is by
# document identity, never by parsing the question or vendor text).
# ---------------------------------------------------------------------------

def test_two_receipts_sharing_a_date_are_both_candidates():
    # Two distinct documents (different chunk_ids/vendors) that both
    # happen to be dated the same -- the date itself plays no role in
    # the detection logic, only the retrieval scores and document identity.
    pool = [
        {"chunk_id": "demo_000", "source": "Example Grocers Inc.", "text": "Date: 03/14/2026 ...", "rerank_score": 2.66},
        {"chunk_id": "demo_003", "source": "Demo Pizza Kitchen", "text": "Date: 03/14/2026 ...", "rerank_score": 0.21},
        {"chunk_id": "demo_001", "source": "Sample Coffee Co.", "text": "Date: 01/02/2026 ...", "rerank_score": -7.1},
    ]
    candidates = cluster_candidate_documents(pool)
    assert {c["chunk_id"] for c in candidates} == {"demo_000", "demo_003"}


def test_two_receipts_sharing_a_vendor_are_still_two_candidates():
    # SAME vendor text, but genuinely different documents (different
    # files/chunk_ids, different totals) -- must not be collapsed into
    # one candidate just because the vendor name string matches.
    pool = [
        {"chunk_id": "fileA_p1", "source": "Example Grocers Inc.", "text": "TOTAL 42.17", "score": 10.0},
        {"chunk_id": "fileB_p1", "source": "Example Grocers Inc.", "text": "TOTAL 58.00", "score": 9.2},
    ]
    candidates = cluster_candidate_documents(pool)
    assert len(candidates) == 2
    assert {c["chunk_id"] for c in candidates} == {"fileA_p1", "fileB_p1"}


def test_uniquely_identified_receipt_is_a_single_candidate():
    pool = [
        {"chunk_id": "demo_002", "source": "Fictional Hardware Supply", "text": "TOTAL 156.42", "rerank_score": 1.96},
        {"chunk_id": "demo_008", "source": "x", "text": "irrelevant", "rerank_score": -5.22},
    ]
    candidates = cluster_candidate_documents(pool)
    assert len(candidates) == 1
    assert candidates[0]["chunk_id"] == "demo_002"


def test_multipage_upload_same_document_collapses_to_one_candidate():
    pool = [
        {"chunk_id": "fileA_p1", "source": "receipt.pdf", "text": "page 1", "score": 5.0},
        {"chunk_id": "fileA_p2", "source": "receipt.pdf", "text": "page 2", "score": 4.9},
    ]
    assert len(cluster_candidate_documents(pool)) == 1


def test_format_candidates_includes_page_for_upload_chunks():
    pool = [{"chunk_id": "fileA_p2", "source": "receipt.pdf", "text": "x", "score": 1.0}]
    labels = format_candidates(pool)
    assert labels[0]["label"] == "receipt.pdf (page 2)"


def test_format_candidates_falls_back_to_text_when_source_is_identical():
    """Regression test for a real bug found during Milestone 5's hosted-
    model test: the demo/research corpus sets 'source' to a fixed dataset
    identifier ("synthetic_demo") for EVERY document, not a per-document
    name -- the old code used 'source' unconditionally, so a genuinely
    ambiguous question produced a disambiguation prompt listing 4
    IDENTICAL "synthetic_demo" choices, a signal that looks broken rather
    than helpful. This is exactly why the hosted model failed to ask for
    clarification on "What is the total?" even though the ambiguity
    signal itself (cluster_candidate_documents) correctly fired -- the
    label it was given was useless. Must fall back to a per-document label
    derived from the chunk's own text (reliably the vendor name on a
    receipt) whenever 'source' doesn't distinguish the candidates."""
    candidates = [
        {"chunk_id": "demo_008", "source": "synthetic_demo", "text": "DEMO PIZZA KITCHEN\nTOTAL 12.00"},
        {"chunk_id": "demo_003", "source": "synthetic_demo", "text": "MOCK BOOKSTORE\nTOTAL 37.45"},
    ]
    labels = [c["label"] for c in format_candidates(candidates)]
    assert labels == ["DEMO PIZZA KITCHEN", "MOCK BOOKSTORE"]
    assert len(set(labels)) == len(labels)  # never identical choices


def test_format_candidates_still_prefers_source_when_it_distinguishes():
    """Upload mode's 'source' is already the original filename -- a good,
    distinguishing label -- so the text-fallback must NOT override it
    (that would lose the explicit "filename, and page" requirement for
    multi-page uploads, where the first line of page 2's text is not
    the vendor name)."""
    candidates = [
        {"chunk_id": "fileA", "source": "receipt_a.pdf", "text": "irrelevant first line A"},
        {"chunk_id": "fileB", "source": "receipt_b.pdf", "text": "irrelevant first line B"},
    ]
    labels = [c["label"] for c in format_candidates(candidates)]
    assert labels == ["receipt_a.pdf", "receipt_b.pdf"]


# ---------------------------------------------------------------------------
# Standard RAG: real wiring (app/rag_pipeline.py), mocked reranker/provider.
# ---------------------------------------------------------------------------

@contextmanager
def _mock_pipeline(rerank_return, generate_return):
    """Bypasses the lazy _reranker/_provider properties with direct mocks
    -- no real model load, and the REAL answer_from_pool() control flow
    (disambiguation detection, CLARIFICATION_NEEDED parsing) still runs.

    A context manager, not a bare assignment: `pipeline` is the SAME
    module-level singleton tests/test_integration.py's real-model tests
    use later in the same pytest session -- a bare `pipeline._reranker =
    Mock()` with no teardown would permanently replace the real
    reranker/provider for the rest of the run (a real bug this project
    hit: those real-model tests failed, passing only in isolation, until
    this was fixed to restore the original attributes afterward)."""
    original_reranker, original_provider = pipeline._reranker, pipeline._provider
    pipeline._reranker = Mock()
    pipeline._reranker.rerank.return_value = (rerank_return, 12.0)
    pipeline._provider = Mock()
    pipeline._provider.generate.return_value = (generate_return, 345.0)
    try:
        yield
    finally:
        pipeline._reranker, pipeline._provider = original_reranker, original_provider


def test_standard_two_documents_sharing_a_date_triggers_clarify():
    pool = [
        {"chunk_id": "demo_000", "source": "Example Grocers Inc.", "text": "t1", "score": 1.0},
        {"chunk_id": "demo_003", "source": "Demo Pizza Kitchen", "text": "t2", "score": 0.9},
    ]
    reranked = [{**pool[0], "rerank_score": 2.66}, {**pool[1], "rerank_score": 0.21}]
    with _mock_pipeline(reranked, "CLARIFICATION_NEEDED: Which receipt do you mean -- Example Grocers Inc. or Demo Pizza Kitchen?"):
        result = pipeline.answer_from_pool("What was purchased on 03/14/2026?", pool, retrieval_ms=5.0)
        call_kwargs = pipeline._provider.generate.call_args.kwargs

    assert result["route"] == "clarify"
    assert result["answer"] is None
    assert "Example Grocers Inc." in result["clarification_question"]
    assert len(call_kwargs["candidates"]) == 2


def test_standard_two_documents_sharing_a_vendor_triggers_clarify():
    pool = [
        {"chunk_id": "fileA_p1", "source": "Example Grocers Inc.", "text": "t1", "score": 1.0},
        {"chunk_id": "fileB_p1", "source": "Example Grocers Inc.", "text": "t2", "score": 0.9},
    ]
    reranked = [{**pool[0], "rerank_score": 1.5}, {**pool[1], "rerank_score": 1.2}]
    with _mock_pipeline(reranked, "CLARIFICATION_NEEDED: Which Example Grocers Inc. receipt do you mean?"):
        result = pipeline.answer_from_pool("What is the total?", pool, retrieval_ms=5.0)
        call_kwargs = pipeline._provider.generate.call_args.kwargs

    assert result["route"] == "clarify"
    assert len(call_kwargs["candidates"]) == 2


def test_standard_uniquely_identified_receipt_answers_normally():
    """Regression protection: an ordinary single-document question must
    NOT be treated as ambiguous just because more than one chunk was
    retrieved (only one of them is actually comparably relevant here)."""
    pool = [
        {"chunk_id": "demo_002", "source": "Fictional Hardware Supply", "text": "TOTAL 156.42", "score": 1.0},
        {"chunk_id": "demo_008", "source": "x", "text": "irrelevant", "score": 0.1},
    ]
    reranked = [{**pool[0], "rerank_score": 1.96}, {**pool[1], "rerank_score": -5.22}]
    with _mock_pipeline(reranked, "The total at Fictional Hardware Supply is $156.42."):
        result = pipeline.answer_from_pool("What is the total at Fictional Hardware Supply?", pool, retrieval_ms=5.0)
        call_kwargs = pipeline._provider.generate.call_args.kwargs

    assert result["route"] == "answer"
    assert result["answer"] == "The total at Fictional Hardware Supply is $156.42."
    assert call_kwargs["candidates"] is None  # no ambiguity signal sent to the model at all


def test_standard_clarification_followup_selecting_one_candidate_resolves():
    """End-to-end through the real API: ask an ambiguous question, answer
    the clarification by naming one candidate, confirm it resolves to a
    normal answer (not another clarify) and the pending state is cleared."""
    client = TestClient(app)
    pool = [
        {"chunk_id": "demo_000", "source": "Example Grocers Inc.", "text": "t1", "score": 1.0},
        {"chunk_id": "demo_003", "source": "Demo Pizza Kitchen", "text": "t2", "score": 0.9},
    ]
    reranked = [{**pool[0], "rerank_score": 2.66}, {**pool[1], "rerank_score": 0.21}]

    with patch("app.main.pipeline.answer") as mock_answer:
        mock_answer.side_effect = [
            {"answer": None, "sources": [], "has_sufficient_evidence": None,
             "retrieval_latency_ms": 5.0, "rerank_latency_ms": 1.0, "generation_latency_ms": 2.0,
             "route": "clarify", "clarification_question": "Example Grocers Inc. or Demo Pizza Kitchen?"},
        ]
        resp1 = client.post("/query", json={"question": "What was purchased on 03/14/2026?",
                                             "mode": "demo", "workflow": "standard"})
        body1 = resp1.json()
        assert body1["route"] == "clarify"
        session_id = client.cookies.get("session_id")
        assert sessions.get_pending_clarification(session_id) is not None
        assert sessions.get_pending_clarification(session_id).workflow == "standard"

        mock_answer.side_effect = [
            {"answer": "Bread, Milk, Eggs, Apples, Cheese.", "sources": [], "has_sufficient_evidence": True,
             "retrieval_latency_ms": 5.0, "rerank_latency_ms": 1.0, "generation_latency_ms": 2.0,
             "route": "answer", "clarification_question": None},
        ]
        resp2 = client.post("/query", json={"question": "Example Grocers Inc.", "mode": "demo", "workflow": "standard"})
        body2 = resp2.json()

    client.delete("/session")

    assert body2["route"] == "answer"
    assert "Bread" in body2["answer"]
    # The resolved question actually sent to pipeline.answer() must combine
    # both pieces -- this is the exact bug class fixed for the agent
    # workflow in the clarification-resume fix, now verified for Standard
    # RAG's own (much simpler) resume path too.
    resolved_question_sent = mock_answer.call_args.args[0]
    assert "What was purchased on 03/14/2026?" in resolved_question_sent
    assert "Example Grocers Inc." in resolved_question_sent
    assert sessions.get_pending_clarification(session_id) is None


def test_standard_and_agent_pending_clarifications_do_not_cross_resume():
    """A pending clarification left by one workflow must not be consumed
    as a resume by the other -- each workflow's follow-up starts fresh if
    the OTHER workflow is the one with state pending."""
    client = TestClient(app)
    with patch("app.main.pipeline.answer") as mock_answer:
        mock_answer.return_value = {
            "answer": None, "sources": [], "has_sufficient_evidence": None,
            "retrieval_latency_ms": 1.0, "rerank_latency_ms": None, "generation_latency_ms": 1.0,
            "route": "clarify", "clarification_question": "Which one?",
        }
        client.post("/query", json={"question": "What is the total?", "mode": "demo", "workflow": "standard"})

    session_id = client.cookies.get("session_id")
    assert sessions.get_pending_clarification(session_id).workflow == "standard"

    with patch.object(agent_graph, "call_with_tools") as mock_llm, \
         patch.object(agent_graph, "assess_evidence") as mock_assess:
        mock_llm.return_value = {"content": "unsure", "tool_calls": [], "malformed_tool_calls": []}
        mock_assess.return_value = {"route": "insufficient_evidence", "answer": None,
                                     "clarification_question": None, "cited_passage_ids": [], "reasoning": "x"}
        resp = client.post("/query", json={"question": "Example Grocers Inc.", "mode": "demo", "workflow": "agent"})

    client.delete("/session")
    # The agent call must NOT have been treated as resuming standard's
    # pending clarify -- it got a fresh new_state(), which this mock setup
    # routes to insufficient_evidence, confirming no cross-workflow resume
    # was attempted (a resume would have gone through resume_state()/the
    # mocked assess_evidence differently and raised if misused).
    assert resp.json()["route"] == "insufficient_evidence"


# ---------------------------------------------------------------------------
# Routing-rule fixes (Milestone 5 hosted-model test): two real failures
# found against Together.ai's Qwen3.5-9B --
#   (a) a genuinely ambiguous question ("What is the total?", 4 receipts)
#       was answered "I don't know" instead of triggering clarify -- traced
#       to format_candidates() labeling every demo-corpus candidate
#       identically ("synthetic_demo"), a signal that looks broken rather
#       than useful (see the format_candidates tests above).
#   (b) a question that already names a specific vendor ("...Sample Coffee
#       Co. receipt?") but asks about an attribute absent from it got
#       routed to "clarify" instead of "insufficient_evidence" -- the
#       model was asking WHICH document despite the question already
#       saying which one, because other retrieved documents happened to
#       score comparably (the exact "off-topic questions cluster closely
#       too" trap app/disambiguation.py's own docstring warns about).
# Both prompts (scripts/llm_client.py::build_prompt(),
# app/agent/llm.py::assess_evidence()) now say explicitly: a question
# that already names a document overrides a retrieval-coincidence-based
# candidate signal -- these tests check that guidance is actually present
# in the real prompt sent to the model, deterministically (no network).
# ---------------------------------------------------------------------------

import sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "scripts"))
from llm_client import build_prompt  # noqa: E402

from app.agent.llm import assess_evidence  # noqa: E402


def test_standard_prompt_distinguishes_identically_sourced_candidates():
    """The exact scenario that caused failure (a): demo-corpus candidates
    whose 'source' is identical. The disambiguation block in the real
    prompt sent to the model must show distinguishable labels, not the
    same string repeated, and must carry the "already names a document"
    guard."""
    candidates_raw = [
        {"chunk_id": "demo_008", "source": "synthetic_demo", "text": "DEMO PIZZA KITCHEN\nTOTAL 12.00"},
        {"chunk_id": "demo_003", "source": "synthetic_demo", "text": "MOCK BOOKSTORE\nTOTAL 37.45"},
    ]
    prompt_candidates = format_candidates(candidates_raw)
    prompt = build_prompt("What is the total?", context_chunks=candidates_raw, candidates=prompt_candidates)

    assert "DEMO PIZZA KITCHEN" in prompt
    assert "MOCK BOOKSTORE" in prompt
    assert prompt.count("synthetic_demo") == 0  # the useless identical label must not leak into the prompt
    assert "single out exactly ONE" in prompt


def test_standard_prompt_does_not_treat_a_shared_vendor_name_as_uniquely_identifying():
    """Milestone 5 regression: a vendor name does NOT by itself single out
    one document when more than one retrieved candidate shares that same
    vendor (e.g. two Sample Coffee Co. receipts on different dates) --
    the prompt must say so explicitly, not just "if the question names a
    vendor, stop asking" (the wording this replaced, which incorrectly
    suppressed clarification in exactly this case)."""
    candidates_raw = [
        {"chunk_id": "demo_010", "source": "synthetic_demo", "text": "SAMPLE COFFEE CO.\nDate: 01/02/2026\nTOTAL 9.75"},
        {"chunk_id": "demo_011", "source": "synthetic_demo", "text": "SAMPLE COFFEE CO.\nDate: 03/09/2026\nTOTAL 14.50"},
    ]
    prompt_candidates = format_candidates(candidates_raw)
    prompt = build_prompt("What is the total at Sample Coffee Co.?", context_chunks=candidates_raw, candidates=prompt_candidates)

    assert "does NOT by itself single out which one is meant" in prompt
    assert "MORE THAN ONE of the documents listed" in prompt


def test_agent_prompt_tells_model_named_vendor_overrides_candidate_noise():
    """The exact scenario that caused failure (b): mocks only the model
    call (same scope as the rest of this file) and inspects the REAL
    prompt app/agent/llm.py::assess_evidence() builds -- confirms the
    "named vendor overrides retrieval noise" guidance is actually present
    for both the route-criteria text and the candidate_note, and that the
    candidates shown are the real (non-identical) labels."""
    candidates_raw = [
        {"chunk_id": "demo_001", "source": "synthetic_demo", "text": "SAMPLE COFFEE CO.\nTOTAL 9.75"},
        {"chunk_id": "demo_005", "source": "synthetic_demo", "text": "DEMO PIZZA KITCHEN\nTOTAL 12.00"},
    ]
    prompt_candidates = format_candidates(candidates_raw)

    fake_response = Mock()
    fake_response.choices = [Mock(message=Mock(content=(
        '{"route": "insufficient_evidence", "answer": null, "clarification_question": null, '
        '"cited_passage_ids": [], "reasoning": "payment method not stated"}'
    )))]
    fake_client = Mock()
    fake_client.chat.completions.create.return_value = fake_response

    with patch("app.agent.llm._get_client", return_value=fake_client), \
         patch("app.agent.llm._get_model", return_value="Qwen/Qwen3.5-9B"):
        result = assess_evidence(
            "What payment method was used on the Sample Coffee Co. receipt?",
            gathered_context="[passage_id=demo_001] (synthetic_demo)\nSAMPLE COFFEE CO.\nTOTAL 9.75",
            candidates=prompt_candidates,
        )

    sent_prompt = fake_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "single out exactly ONE" in sent_prompt
    assert "SAMPLE COFFEE CO." in sent_prompt
    assert "DEMO PIZZA KITCHEN" in sent_prompt
    # The candidate_note's choice list (not gathered_context, which
    # legitimately carries the real "(synthetic_demo)" source tag) must
    # not list the same useless label twice.
    assert "- synthetic_demo" not in sent_prompt
    assert result["route"] == "insufficient_evidence"


def test_agent_prompt_does_not_treat_a_shared_vendor_name_as_uniquely_identifying():
    """Milestone 5 regression, agent side: the four real clarification
    failures investigated this round showed the model choosing
    "insufficient_evidence" over "clarify" for a document-selection
    question -- traced to the route criteria text not clearly excluding
    the case where a named vendor still matches more than one candidate.
    Confirms the corrected wording reaches the real prompt."""
    candidates_raw = [
        {"chunk_id": "demo_010", "source": "synthetic_demo", "text": "SAMPLE COFFEE CO.\nDate: 01/02/2026\nTOTAL 9.75"},
        {"chunk_id": "demo_011", "source": "synthetic_demo", "text": "SAMPLE COFFEE CO.\nDate: 03/09/2026\nTOTAL 14.50"},
    ]
    prompt_candidates = format_candidates(candidates_raw)

    fake_response = Mock()
    fake_response.choices = [Mock(message=Mock(content=(
        '{"route": "clarify", "answer": null, '
        '"clarification_question": "Which Sample Coffee Co. receipt -- 01/02/2026 or 03/09/2026?", '
        '"cited_passage_ids": [], "supporting_quote": null, "reasoning": "vendor matches two candidates"}'
    )))]
    fake_client = Mock()
    fake_client.chat.completions.create.return_value = fake_response

    with patch("app.agent.llm._get_client", return_value=fake_client), \
         patch("app.agent.llm._get_model", return_value="Qwen/Qwen3.5-9B"):
        result = assess_evidence(
            "What is the total at Sample Coffee Co.?",
            gathered_context="[passage_id=demo_010] (synthetic_demo)\nSAMPLE COFFEE CO.\nDate: 01/02/2026\nTOTAL 9.75"
                              "\n\n[passage_id=demo_011] (synthetic_demo)\nSAMPLE COFFEE CO.\nDate: 03/09/2026\nTOTAL 14.50",
            candidates=prompt_candidates,
        )

    sent_prompt = fake_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "does NOT by itself single out which one is meant" in sent_prompt
    assert "MORE THAN ONE of the retrieved candidates" in sent_prompt
    assert result["route"] == "clarify"
