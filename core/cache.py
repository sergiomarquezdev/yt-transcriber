"""Simple LLM response cache with content-based keys.

This module provides intelligent caching for LLM responses based on:
1. Model name
2. Prompt template version
3. Input content hash (video-specific)

This ensures that:
- Same video + same prompt → cache hit (saves cost/time)
- Different video → cache miss (generates new response)
- Updated prompt version → cache miss (regenerates with new prompt)
"""

import hashlib
import json
import logging
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default maximum entries in memory cache (LRU eviction when exceeded)
DEFAULT_MAX_MEM_CACHE_SIZE = 1000


class LLMCache:
    """Simple file-based LLM response cache with intelligent keying."""

    def __init__(
        self,
        cache_dir: Path,
        ttl_days: int = 7,
        max_mem_cache_size: int = DEFAULT_MAX_MEM_CACHE_SIZE,
    ):
        """Initialize cache.

        Args:
            cache_dir: Directory to store cache files
            ttl_days: Time-to-live in days for cached responses
            max_mem_cache_size: Maximum entries in memory cache (LRU eviction)
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(days=ttl_days)
        self._max_mem_cache_size = max_mem_cache_size
        # In-memory LRU cache using OrderedDict for efficient eviction
        self._mem_cache: OrderedDict[str, dict] = OrderedDict()
        logger.debug(
            f"LLM cache initialized at {cache_dir} with TTL={ttl_days} days, "
            f"max_mem_cache_size={max_mem_cache_size}"
        )

    def _make_key(self, model: str, prompt_version: str, inputs: dict) -> str:
        """Generate cache key from model, prompt version, and input content."""
        # Single pass hashing for efficiency (Issue #9)
        payload = {"m": model, "v": prompt_version, "i": inputs}
        # Dump with sort_keys to ensure consistency
        payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        cache_key = hashlib.sha256(payload_bytes).hexdigest()
        return cache_key

    def _update_mem_cache(self, cache_key: str, data: Any, expires: datetime) -> None:
        """Update memory cache with LRU eviction.

        Args:
            cache_key: Cache key
            data: Data to store
            expires: Expiration datetime
        """
        # If key exists, remove it first to update position
        if cache_key in self._mem_cache:
            del self._mem_cache[cache_key]

        # Evict oldest entries if at capacity
        while len(self._mem_cache) >= self._max_mem_cache_size:
            oldest_key = next(iter(self._mem_cache))
            del self._mem_cache[oldest_key]
            logger.debug(f"LRU eviction: removed {oldest_key[:8]}...")

        # Add new entry at end (most recently used)
        self._mem_cache[cache_key] = {"data": data, "expires": expires}

    def get(self, model: str, prompt_version: str, inputs: dict) -> dict | None:
        """Get cached response if it exists and is not expired."""
        cache_key = self._make_key(model, prompt_version, inputs)

        # 1. Check in-memory cache first (Issue #6 - reduce sync I/O)
        if cache_key in self._mem_cache:
            entry = self._mem_cache[cache_key]
            # Check expiry (fast in-memory check)
            if datetime.now() < entry["expires"]:
                # Move to end for LRU behavior (most recently used)
                self._mem_cache.move_to_end(cache_key)
                logger.debug(f"Memory Cache HIT: {cache_key[:8]}...")
                return entry["data"]
            else:
                del self._mem_cache[cache_key]

        # 2. Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, encoding="utf-8") as f:
                cached: dict[str, Any] = json.load(f)

            # Check expiration
            expires_at = datetime.fromisoformat(cached["expires_at"])
            if datetime.now() > expires_at:
                logger.debug(f"Cache EXPIRED - deleting: {cache_key[:16]}...")
                cache_file.unlink(missing_ok=True)
                return None

            logger.info(
                f"✅ Disk Cache HIT - Reusing response (saved ${cached.get('cost', 0):.4f} + {cached.get('time_saved', 0):.1f}s)"
            )
            response_obj = cached.get("response")

            if isinstance(response_obj, dict):
                # Populate memory cache with LRU eviction
                self._update_mem_cache(cache_key, response_obj, expires_at)
                return response_obj

            # If response isn't a dict, treat as corrupted
            logger.warning("Cache 'response' is not a dict; deleting corrupted cache file")
            cache_file.unlink(missing_ok=True)
            return None

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Cache file corrupted, deleting: {e}")
            cache_file.unlink(missing_ok=True)
            return None

    def set(
        self,
        model: str,
        prompt_version: str,
        inputs: dict,
        response: Any,
        cost: float = 0.0,
        time_saved: float = 0.0,
    ):
        """Save response to cache."""
        cache_key = self._make_key(model, prompt_version, inputs)
        cache_file = self.cache_dir / f"{cache_key}.json"

        now = datetime.now()
        expires_at = now + self.ttl

        cached_data = {
            "model": model,
            "prompt_version": prompt_version,
            "response": response,
            "cost": cost,
            "time_saved": time_saved,
            "created_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
        }

        # Update disk (sync I/O)
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cached_data, f, indent=2, ensure_ascii=False)
            logger.debug(
                f"Cache saved: {cache_key[:16]}... (expires: {expires_at.strftime('%Y-%m-%d %H:%M')})"
            )

            # Update memory cache with LRU eviction
            self._update_mem_cache(cache_key, response, expires_at)

            # Probabilistic Cleanup (Issue #13)
            # 5% chance to clean up expired files on write
            import random

            if random.random() < 0.05:
                self.clear_expired()

        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def clear_expired(self) -> int:
        """Clear all expired cache files.

        Returns:
            Number of files deleted
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, encoding="utf-8") as f:
                    cached = json.load(f)
                expires_at = datetime.fromisoformat(cached["expires_at"])
                if datetime.now() > expires_at:
                    cache_file.unlink()
                    count += 1
            except Exception:
                # Corrupted file - delete it
                cache_file.unlink()
                count += 1

        if count > 0:
            logger.info(f"Cleared {count} expired cache files")
        return count


__all__ = ["LLMCache"]
