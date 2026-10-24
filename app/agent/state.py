"""
Typed state for the LangGraph agent workflow.

session_mode/session_id are set ONCE, server-side, when the graph is
invoked (from the authenticated request's session cookie) and never
changed by any node. Tool implementations (tools.py) read these directly
from state rather than accepting them as model-controlled arguments — this
is the actual mechanism that makes "neither tool arguments nor the model
may override session identity" true, not just a comment.
"""

import time
from typing import TypedDict


class AgentState(TypedDict):
    question: str
    session_mode: str  # "demo" or "upload" — server-bound, read-only to nodes
    session_id: str | None  # upload session id — server-bound, read-only to nodes

    messages: list[dict]  # OpenAI-style message history for the tool-calling loop
    tool_call_count: int
    step_count: int
    started_at: float

    # passage_id -> {chunk_id, source, text, score} — accumulated across every
    # search_documents call this run. fetch_source can only return passages
    # already in here, which is what keeps it scoped to this session's own
    # retrieval results instead of being an unscoped global lookup.
    retrieved_passages: dict[str, dict]

    route: str | None  # "answer" | "clarify" | "insufficient_evidence"
    answer: str | None
    clarification_question: str | None
    cited_passage_ids: list[str]

    execution_trace: list[dict]  # [{node, tool, duration_ms, detail}, ...] — metadata only, no content
    error: str | None  # set on a bounded-execution abort or unrecoverable failure

    # Internal scratch field: tool calls requested by the latest agent_node
    # turn, consumed by tools_node / the routing function immediately after.
    # Not part of the public graph contract, but declared here (rather than
    # left as an undeclared dict key) so its purpose and lifetime are documented.
    _pending_tool_calls: list[dict]


def new_state(question: str, session_mode: str, session_id: str | None) -> AgentState:
    return AgentState(
        question=question,
        session_mode=session_mode,
        session_id=session_id,
        messages=[],
        tool_call_count=0,
        step_count=0,
        started_at=time.monotonic(),
        retrieved_passages={},
        route=None,
        answer=None,
        clarification_question=None,
        cited_passage_ids=[],
        execution_trace=[],
        error=None,
        _pending_tool_calls=[],
    )


def resume_state(
    original_question: str,
    clarification_answer: str,
    prior_messages: list[dict],
    prior_retrieved_passages: dict,
    session_mode: str,
    session_id: str | None,
) -> AgentState:
    """Resumes a paused clarify run. Carries over the message history and
    already-retrieved passages from app/upload/session.py's PendingClarification
    (so the agent doesn't re-search from scratch) and appends the user's
    follow-up as a new turn. Gets its OWN fresh step/tool-call/time budget —
    each user-facing turn is bounded independently, not against the
    original request's already-spent budget.

    This is a plain re-invocation from START with a non-empty `messages`
    list (agent_node only does its one-time system-prompt setup when
    `messages` is empty) rather than LangGraph checkpointer-based resume —
    a deliberate simplicity choice: the session itself is already the
    persistence boundary for this feature, so durable graph checkpointing
    across process restarts isn't needed here.
    """
    messages = list(prior_messages) + [{"role": "user", "content": f"Clarification: {clarification_answer}"}]
    return AgentState(
        question=original_question,
        session_mode=session_mode,
        session_id=session_id,
        messages=messages,
        tool_call_count=0,
        step_count=0,
        started_at=time.monotonic(),
        retrieved_passages=dict(prior_retrieved_passages),
        route=None,
        answer=None,
        clarification_question=None,
        cited_passage_ids=[],
        execution_trace=[],
        error=None,
        _pending_tool_calls=[],
    )
