"""
Rate Limiter - Control API Call Frequency
"""
import time
from typing import Optional


class RateLimiter:
    """API Call Rate Limiter"""

    def __init__(self, requests_per_min: int = 60, tokens_per_min: int = 150000):
        self.request_limit = requests_per_min
        self.token_limit = tokens_per_min

        self._request_count = 0
        self._token_count = 0
        self._last_reset = time.time()

    def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire call permission

        Args:
            tokens: Number of tokens consumed

        Returns:
            Whether successfully acquired
        """
        now = time.time()

        # Reset counters every minute
        if now - self._last_reset > 60:
            self._request_count = 0
            self._token_count = 0
            self._last_reset = now

        # Check if limit exceeded
        if self._request_count >= self.request_limit:
            return False

        if self._token_count + tokens > self.token_limit:
            return False

        # Update counters
        self._request_count += 1
        self._token_count += tokens

        return True

    def wait_if_needed(self):
        """Wait until a call can be made, if needed"""
        while not self.acquire():
            time.sleep(1.0)
