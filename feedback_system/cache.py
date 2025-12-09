"""LRU cache for feedback analysis results."""
import hashlib
import time
from typing import Optional, Dict, Any
from collections import OrderedDict
from config import config
from schemas import AnalysisResult


class FeedbackCache:
    """LRU cache with TTL for analysis results.

    Design decisions:
    - LRU eviction to cap memory usage
    - TTL to prevent stale results
    - Hash-based keys for consistent lookup
    """

    def __init__(self, max_size: int = None, ttl_seconds: int = None):
        """Initialize the cache.

        Args:
            max_size: Maximum number of entries (default from config)
            ttl_seconds: Time-to-live in seconds (default from config)
        """
        self.max_size = max_size or config.CACHE_MAX_SIZE
        self.ttl_seconds = ttl_seconds or config.CACHE_TTL_SECONDS
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._stats = {"hits": 0, "misses": 0}

    def _generate_key(self, feedback_text: str) -> str:
        """Generate a cache key from feedback text.

        Uses hash to:
        - Normalize similar texts
        - Keep keys a consistent size
        - Privacy (don't store raw feedback in keys)
        """
        return hashlib.sha256(feedback_text.encode()).hexdigest()

    def get(self, feedback_text: str) -> Optional[AnalysisResult]:
        """Retrieve cached result if available and not expired.

        Args:
            feedback_text: The feedback to look up

        Returns:
            AnalysisResult if cached and valid, None otherwise
        """
        key = self._generate_key(feedback_text)

        if key not in self._cache:
            self._stats["misses"] += 1
            return None

        entry = self._cache[key]
        current_time = time.time()

        # Check if expired
        if current_time - entry["timestamp"] > self.ttl_seconds:
            del self._cache[key]
            self._stats["misses"] += 1
            return None

        # Move to end (LRU)
        self._cache.move_to_end(key)
        self._stats["hits"] += 1

        return AnalysisResult(**entry["result"])

    def set(self, feedback_text: str, result: AnalysisResult) -> None:
        """Store analysis result in cache.

        Args:
            feedback_text: The feedback that was analyzed
            result: The analysis result to cache
        """
        key = self._generate_key(feedback_text)

        # If at capacity, remove oldest entry (LRU)
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._cache.popitem(last=False)

        self._cache[key] = {
            "result": result.model_dump(),
            "timestamp": time.time()
        }

        # Move to end (most recent)
        self._cache.move_to_end(key)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._stats = {"hits": 0, "misses": 0}

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with hits, misses, size, and hit rate
        """
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = (
            self._stats["hits"] / total_requests if total_requests > 0 else 0
        )

        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "size": len(self._cache),
            "max_size": self.max_size,
            "hit_rate": round(hit_rate, 3)
        }
