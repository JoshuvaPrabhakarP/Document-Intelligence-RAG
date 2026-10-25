"""
Shared, autouse test fixtures. Resets app/main.py's module-level rate/
concurrency limiter state before every test -- without this, running many
tests in one pytest session (all using TestClient, which presents the
same client address every time) would trip the real per-IP rate limits
meant for an actual deployment, failing unrelated tests for a reason that
has nothing to do with what they're testing. The PRODUCTION limits
themselves (app/config.py's rate_limit_*/max_concurrent_heavy_requests)
are left exactly as configured -- this only makes each test start with a
clean slate, the same test-isolation principle already applied to the
shared RAGPipeline singleton (see tests/test_disambiguation.py).
"""

import threading

import pytest

from app.config import get_settings
from app.main import heavy_request_limiter, query_rate_limiter, upload_rate_limiter


@pytest.fixture(autouse=True)
def _reset_rate_limiters():
    query_rate_limiter._hits.clear()
    upload_rate_limiter._hits.clear()
    # Semaphores self-balance (every try_acquire() is matched by a
    # release() in a finally block), but replacing it outright is a
    # cheap, certain way to guarantee no test starts already partially
    # throttled by a previous test's in-flight state.
    heavy_request_limiter._semaphore = threading.Semaphore(get_settings().max_concurrent_heavy_requests)
    yield
