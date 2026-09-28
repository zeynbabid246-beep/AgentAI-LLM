"""Tests for the in-memory sliding-window rate limiter."""

from main import SlidingWindowLimiter


class TestSlidingWindowLimiter:
    def test_allows_up_to_limit(self):
        limiter = SlidingWindowLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            allowed, _ = limiter.check("s1")
            assert allowed

    def test_blocks_over_limit(self):
        limiter = SlidingWindowLimiter(max_requests=2, window_seconds=60)
        limiter.check("s1")
        limiter.check("s1")
        allowed, retry = limiter.check("s1")
        assert not allowed
        assert retry >= 1

    def test_keys_are_independent(self):
        limiter = SlidingWindowLimiter(max_requests=1, window_seconds=60)
        assert limiter.check("a")[0]
        assert limiter.check("b")[0]
        assert not limiter.check("a")[0]

    def test_window_expiry(self):
        limiter = SlidingWindowLimiter(max_requests=1, window_seconds=0.05)
        assert limiter.check("s1")[0]
        import time

        time.sleep(0.06)
        assert limiter.check("s1")[0]
