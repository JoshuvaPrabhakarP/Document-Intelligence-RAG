"""
Per-browser-session upload storage. Each session gets its own directory
OUTSIDE the git repo entirely (under the OS temp dir by default), so
uploaded files and extracted text can never end up committed — there's no
.gitignore rule to maintain or forget, the files simply never live inside
the project directory.

A session is identified by an opaque UUID stored in an HTTP-only cookie
(app/main.py sets it). Nothing here trusts the client for anything beyond
that one random ID — file paths are always server-generated.

Expiry: lazy, checked on every session-touching request (list_sessions()
callers call sweep_expired() first) rather than a background thread, so
behavior is deterministic and testable without timing-dependent flakiness.
"""

import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class UploadedFile:
    file_id: str
    original_filename: str
    saved_path: str
    content_type: str
    page_count: int
    extracted_text: str
    chunk_ids: list[str] = field(default_factory=list)
    warning: str | None = None  # e.g. "OCR unavailable, page 2 has no extractable text"


@dataclass
class PendingClarification:
    """A paused agent run awaiting the user's answer to a clarifying
    question — Milestone 3. Lives inside the SAME per-session dict as
    uploaded files, so clearing or expiring a session (both already-existing
    code paths) wipes this for free, with no separate cleanup mechanism to
    forget."""
    original_question: str
    clarification_question: str
    messages: list[dict]  # the agent's message history at the point it paused
    retrieved_passages: dict  # passages already gathered, so a resume doesn't re-search from scratch


class SessionManager:
    def __init__(self, base_dir: str, ttl_minutes: int):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_minutes * 60
        # session_id -> {"created_at": float, "files": {file_id: UploadedFile},
        #                "pending_clarification": PendingClarification | None}
        self._sessions: dict[str, dict] = {}

    def _session_dir(self, session_id: str) -> Path:
        return self.base_dir / session_id

    def touch_or_create(self, session_id: str) -> None:
        self.sweep_expired()
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "created_at": time.time(), "files": {}, "pending_clarification": None,
            }
            (self._session_dir(session_id) / "files").mkdir(parents=True, exist_ok=True)

    def is_known(self, session_id: str) -> bool:
        self.sweep_expired()
        return session_id in self._sessions

    def add_file(self, session_id: str, uploaded: UploadedFile) -> None:
        self._sessions[session_id]["files"][uploaded.file_id] = uploaded

    def list_files(self, session_id: str) -> list[UploadedFile]:
        self.sweep_expired()
        return list(self._sessions.get(session_id, {}).get("files", {}).values())

    def chunks_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "chunks.jsonl"

    def index_dir(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "index"

    def files_dir(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "files"

    def clear_session(self, session_id: str) -> bool:
        """'Clear uploaded receipts' — deletes all files/index for this
        session from disk and forgets it in memory. Returns whether there
        was anything to clear."""
        existed = session_id in self._sessions
        self._sessions.pop(session_id, None)
        session_dir = self._session_dir(session_id)
        if session_dir.exists():
            shutil.rmtree(session_dir, ignore_errors=True)
        return existed

    def sweep_expired(self) -> list[str]:
        """Removes any session older than the TTL. Called automatically by
        the methods above, and also exposed directly for an explicit
        periodic sweep / tests."""
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s["created_at"] > self.ttl_seconds
        ]
        for sid in expired:
            self.clear_session(sid)
        return expired

    # ---- Milestone 3: clarification state, isolated per session ----

    def set_pending_clarification(self, session_id: str, pending: "PendingClarification") -> None:
        self.touch_or_create(session_id)
        self._sessions[session_id]["pending_clarification"] = pending

    def get_pending_clarification(self, session_id: str) -> "PendingClarification | None":
        self.sweep_expired()
        session = self._sessions.get(session_id)
        return session["pending_clarification"] if session else None

    def clear_pending_clarification(self, session_id: str) -> None:
        if session_id in self._sessions:
            self._sessions[session_id]["pending_clarification"] = None


def new_session_id() -> str:
    return str(uuid.uuid4())


# Module-level singleton. Deliberately instantiated HERE (not in app/main.py)
# so app/agent/tools.py can import it directly without creating a circular
# import (main.py -> agent/graph.py -> agent/tools.py -> main.py would be
# circular; main.py -> agent/... -> upload/session.py is not, since this
# module imports nothing from app.main or app.agent).
def _build_default_manager() -> SessionManager:
    from app.config import get_settings

    settings = get_settings()
    return SessionManager(base_dir=settings.upload_dir, ttl_minutes=settings.upload_session_ttl_minutes)


sessions = _build_default_manager()
