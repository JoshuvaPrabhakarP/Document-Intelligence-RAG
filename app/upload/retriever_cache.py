"""
Caches one HybridRetriever (scripts/fusion.py, UNMODIFIED) per upload
session, so the embedding/BM25 models load once per session rather than
once per query. Evicted explicitly when a session is cleared or expires —
never silently reused across sessions, which is the actual isolation
guarantee ("questions in upload mode must search only that session's
receipts").
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from fusion import HybridRetriever  # noqa: E402

_cache: dict[str, HybridRetriever] = {}


def get_retriever(session_id: str, index_dir: str, chunks_path: str) -> HybridRetriever:
    if session_id not in _cache:
        _cache[session_id] = HybridRetriever(index_dir=index_dir, chunks_path=chunks_path)
    return _cache[session_id]


def invalidate(session_id: str) -> None:
    """Called whenever a session's chunks change (new upload) or the
    session is cleared/expired — forces the next query to rebuild against
    current data instead of serving a stale cached retriever."""
    _cache.pop(session_id, None)
