"""
Verifies session/upload expiry actually deletes files from disk (not
just forgets the in-memory record) -- this had no test coverage before
Milestone 5, despite being a real privacy requirement ("uploads and
extracted content" must not linger). Deterministic: no real TTL wait,
the session's created_at timestamp is set directly to simulate expiry
(app/upload/session.py's own docstring: expiry is lazy and checked on
every session-touching call, specifically so this is testable without
timing-dependent flakiness).
"""

from pathlib import Path

import pytest

from app.upload.session import PendingClarification, SessionManager, UploadedFile


@pytest.fixture
def manager(tmp_path):
    return SessionManager(base_dir=str(tmp_path), ttl_minutes=30)


def test_sweep_expired_deletes_the_session_directory_from_disk(manager):
    sid = "expiring-session"
    manager.touch_or_create(sid)
    manager.add_file(sid, UploadedFile(
        file_id="f1", original_filename="f1.pdf", saved_path=str(manager.files_dir(sid) / "f1.pdf"),
        content_type="application/pdf", page_count=1, extracted_text="SYNTHETIC TEST CONTENT",
    ))
    # Simulate an actual uploaded file on disk, the way app/main.py's
    # upload() endpoint does.
    manager.files_dir(sid).mkdir(parents=True, exist_ok=True)
    saved_file = manager.files_dir(sid) / "f1.pdf"
    saved_file.write_bytes(b"fake pdf bytes")
    assert saved_file.exists()

    manager._sessions[sid]["created_at"] -= manager.ttl_seconds + 1  # simulate TTL having passed

    expired = manager.sweep_expired()

    assert expired == [sid]
    assert not saved_file.exists()
    assert not manager._session_dir(sid).exists()
    assert manager.is_known(sid) is False
    assert manager.list_files(sid) == []


def test_sweep_expired_also_clears_pending_clarification(manager):
    sid = "expiring-with-clarify"
    manager.set_pending_clarification(sid, PendingClarification(
        original_question="What is the total?", clarification_question="Which receipt?",
        messages=[], retrieved_passages={},
    ))
    assert manager.get_pending_clarification(sid) is not None

    manager._sessions[sid]["created_at"] -= manager.ttl_seconds + 1

    manager.sweep_expired()

    assert manager.get_pending_clarification(sid) is None


def test_non_expired_sessions_are_left_alone(manager):
    sid = "fresh-session"
    manager.touch_or_create(sid)
    manager.add_file(sid, UploadedFile(
        file_id="f1", original_filename="f1.pdf", saved_path="/tmp/x",
        content_type="application/pdf", page_count=1, extracted_text="x",
    ))

    expired = manager.sweep_expired()

    assert expired == []
    assert manager.is_known(sid) is True
    assert len(manager.list_files(sid)) == 1


def test_clear_session_deletes_files_immediately_without_waiting_for_ttl(manager):
    sid = "manually-cleared-session"
    manager.touch_or_create(sid)
    manager.files_dir(sid).mkdir(parents=True, exist_ok=True)
    saved_file = manager.files_dir(sid) / "f1.pdf"
    saved_file.write_bytes(b"fake pdf bytes")

    existed = manager.clear_session(sid)

    assert existed is True
    assert not saved_file.exists()
    assert not manager._session_dir(sid).exists()
