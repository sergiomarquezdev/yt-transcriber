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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


class LLMCache:
    """Simple file-based LLM response cache with intelligent keying."""

    def __init__(self, cache_dir: Path, ttl_days: int = 7):
        """Initialize cache.

        Args:
            cache_dir: Directory to store cache files
            ttl_days: Time-to-live in days for cached responses
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(days=ttl_days)
        logger.debug(f"LLM cache initialized at {cache_dir} with TTL={ttl_days} days")

    def _make_key(self, model: str, prompt_version: str, inputs: dict) -> str:
        """Generate cache key from model, prompt version, and input content.

        The key is a hash of:
        - model name (e.g., "gemini-1.5-flash-8b")
        - prompt_version (e.g., "v1.0")
        - inputs hash (e.g., hash of transcript + title)

        This ensures:
        - Different videos → different keys (no incorrect reuse)
        - Same video + updated prompt → different keys (regenerates)
        - Same video + same prompt → same key (cache hit)

        Args:
            model: Model name
            prompt_version: Prompt template version
            inputs: Dictionary of input data (transcript, title, etc.)

        Returns:
            SHA256 hash string
        """
        # Sort inputs to ensure consistent hashing
        inputs_str = json.dumps(inputs, sort_keys=True)
        inputs_hash = hashlib.sha256(inputs_str.encode()).hexdigest()

        # Combine all components
        key_data = f"{model}:{prompt_version}:{inputs_hash}"
        cache_key = hashlib.sha256(key_data.encode()).hexdigest()

        logger.debug(
            f"Cache key generated: {cache_key[:16]}... (inputs_hash: {inputs_hash[:16]}...)"
        )
        return cache_key

    def get(self, model: str, prompt_version: str, inputs: dict) -> dict | None:
        """Get cached response if it exists and is not expired.

        Args:
            model: Model name
            prompt_version: Prompt template version
            inputs: Dictionary of input data

        Returns:
            Cached response dict or None if cache miss
        """
        cache_key = self._make_key(model, prompt_version, inputs)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            logger.debug(f"Cache MISS - file not found: {cache_key[:16]}...")
            return None

        try:
            with open(cache_file, encoding="utf-8") as f:
                cached: dict[str, Any] = json.load(f)

            # Check expiration
            expires_at = datetime.fromisoformat(cached["expires_at"])
            if datetime.now() > expires_at:
                logger.debug(f"Cache EXPIRED - deleting: {cache_key[:16]}...")
                cache_file.unlink()
                return None

            logger.info(
                f"✅ Cache HIT - Reusing response (saved ${cached.get('cost', 0):.4f} + {cached.get('time_saved', 0):.1f}s)"
            )
            response_obj = cached.get("response")
            if isinstance(response_obj, dict):
                return response_obj
            # If response isn't a dict, treat as corrupted and invalidate this cache entry
            logger.warning("Cache 'response' is not a dict; deleting corrupted cache file")
            cache_file.unlink(missing_ok=True)
            return None

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Cache file corrupted, deleting: {e}")
            cache_file.unlink()
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
        """Save response to cache.

        Args:
            model: Model name
            prompt_version: Prompt template version
            inputs: Dictionary of input data
            response: Response to cache (will be JSON-serialized)
            cost: Estimated cost saved by cache hit (for logging)
            time_saved: Estimated time saved in seconds (for logging)
        """
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

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cached_data, f, indent=2, ensure_ascii=False)
            logger.debug(
                f"Cache saved: {cache_key[:16]}... (expires: {expires_at.strftime('%Y-%m-%d %H:%M')})"
            )
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
