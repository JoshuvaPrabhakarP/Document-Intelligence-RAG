"""
Practical, simple, single-process limits for a public demo deployment —
deliberately NOT a general-purpose rate-limiting library. This app runs
as one uvicorn worker with no shared external state (see Dockerfile's
comment on why not multiple workers), so a plain in-memory counter is
the right amount of machinery, not an underpowered simplification of a
"real" solution. If this app is ever scaled to multiple instances behind
a load balancer, per-IP counts would need to move to shared storage
(e.g. Redis) — noted here so that's not a silent gap later.

Three limits, each answering a different abuse/cost scenario:
  - per-IP request rate (RateLimiter): one client hammering the API
  - global concurrency (ConcurrencyLimiter): many clients at once, each
    legitimate alone, collectively exceeding what one small container's
    CPU can do (embedding + reranking + an LLM call per request)
  - per-session upload count (checked in app/main.py directly against
    app.config.Settings.upload_max_files_per_session): one session
    looping uploads to grow its own index/disk usage without bound
"""

import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    """Fixed-window-ish per-key request counter (sliding via a deque of
    timestamps, so it's not susceptible to the classic fixed-window
    boundary-burst problem). Thread-safe: uvicorn's default worker can
    still run sync path-operation functions in a thread pool, and this
    is checked from exactly those."""

    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - 60.0
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] < cutoff:
                hits.popleft()
            if len(hits) >= self.max_per_minute:
                return False
            hits.append(now)
            return True


class ConcurrencyLimiter:
    """A plain counting semaphore with a non-blocking try-acquire, so a
    request over the limit gets an immediate, honest 503 rather than
    queuing silently (queuing would just move the problem to a request
    timeout with no useful error message)."""

    def __init__(self, max_concurrent: int):
        self._semaphore = threading.Semaphore(max_concurrent)

    def try_acquire(self) -> bool:
        return self._semaphore.acquire(blocking=False)

    def release(self) -> None:
        self._semaphore.release()
