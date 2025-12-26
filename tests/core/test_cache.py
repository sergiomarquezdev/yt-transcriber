"""Tests for core.cache module."""

import json
import time
from datetime import timedelta
from unittest.mock import patch

import pytest

from core.cache import LLMCache


class TestLLMCache:
    """Tests for LLMCache class."""

    @pytest.fixture
    def cache(self, temp_cache_dir):
        """Create a cache instance for testing."""
        return LLMCache(cache_dir=temp_cache_dir, ttl_days=7)

    @pytest.fixture
    def sample_inputs(self):
        """Sample inputs for cache operations."""
        return {
            "transcript": "This is a sample transcript for testing.",
            "video_title": "Test Video",
            "task": "summarization",
        }

    # =========================================================================
    # INITIALIZATION TESTS
    # =========================================================================

    def test_init_creates_directory(self, temp_dir):
        """Test that cache directory is created on init."""
        cache_dir = temp_dir / "new_cache"
        assert not cache_dir.exists()

        LLMCache(cache_dir=cache_dir, ttl_days=7)

        assert cache_dir.exists()
        assert cache_dir.is_dir()

    def test_init_with_existing_directory(self, temp_cache_dir):
        """Test init with existing directory doesn't fail."""
        cache = LLMCache(cache_dir=temp_cache_dir, ttl_days=7)
        assert cache.cache_dir == temp_cache_dir

    def test_init_sets_ttl(self, temp_cache_dir):
        """Test that TTL is set correctly."""
        cache = LLMCache(cache_dir=temp_cache_dir, ttl_days=14)
        assert cache.ttl == timedelta(days=14)

    # =========================================================================
    # KEY GENERATION TESTS
    # =========================================================================

    def test_make_key_deterministic(self, cache, sample_inputs):
        """Test that same inputs produce same key."""
        key1 = cache._make_key("model", "v1.0", sample_inputs)
        key2 = cache._make_key("model", "v1.0", sample_inputs)
        assert key1 == key2

    def test_make_key_different_model(self, cache, sample_inputs):
        """Test that different models produce different keys."""
        key1 = cache._make_key("model_a", "v1.0", sample_inputs)
        key2 = cache._make_key("model_b", "v1.0", sample_inputs)
        assert key1 != key2

    def test_make_key_different_version(self, cache, sample_inputs):
        """Test that different versions produce different keys."""
        key1 = cache._make_key("model", "v1.0", sample_inputs)
        key2 = cache._make_key("model", "v2.0", sample_inputs)
        assert key1 != key2

    def test_make_key_different_inputs(self, cache):
        """Test that different inputs produce different keys."""
        inputs1 = {"transcript": "First transcript"}
        inputs2 = {"transcript": "Second transcript"}

        key1 = cache._make_key("model", "v1.0", inputs1)
        key2 = cache._make_key("model", "v1.0", inputs2)
        assert key1 != key2

    def test_make_key_input_order_independent(self, cache):
        """Test that input dict order doesn't affect key."""
        inputs1 = {"a": "1", "b": "2", "c": "3"}
        inputs2 = {"c": "3", "a": "1", "b": "2"}

        key1 = cache._make_key("model", "v1.0", inputs1)
        key2 = cache._make_key("model", "v1.0", inputs2)
        assert key1 == key2

    # =========================================================================
    # SET/GET TESTS
    # =========================================================================

    def test_set_and_get_basic(self, cache, sample_inputs):
        """Test basic set and get operations."""
        response = {"text": "Generated response"}

        cache.set("model", "v1.0", sample_inputs, response)
        result = cache.get("model", "v1.0", sample_inputs)

        assert result == response

    def test_get_returns_none_for_missing(self, cache, sample_inputs):
        """Test that get returns None for missing cache."""
        result = cache.get("model", "v1.0", sample_inputs)
        assert result is None

    def test_get_different_version_returns_none(self, cache, sample_inputs):
        """Test that different version returns None (cache miss)."""
        response = {"text": "Response"}
        cache.set("model", "v1.0", sample_inputs, response)

        # Get with different version
        result = cache.get("model", "v2.0", sample_inputs)
        assert result is None

    def test_set_with_cost_and_time(self, cache, sample_inputs, temp_cache_dir):
        """Test that cost and time_saved are stored."""
        response = {"text": "Response"}
        cache.set(
            "model",
            "v1.0",
            sample_inputs,
            response,
            cost=0.05,
            time_saved=2.5,
        )

        # Read the cache file directly to verify
        cache_files = list(temp_cache_dir.glob("*.json"))
        assert len(cache_files) == 1

        with open(cache_files[0]) as f:
            data = json.load(f)

        assert data["cost"] == 0.05
        assert data["time_saved"] == 2.5

    def test_cache_file_structure(self, cache, sample_inputs, temp_cache_dir):
        """Test that cache file has correct structure."""
        response = {"text": "Response"}
        cache.set("model", "v1.0", sample_inputs, response)

        cache_files = list(temp_cache_dir.glob("*.json"))
        with open(cache_files[0]) as f:
            data = json.load(f)

        assert "model" in data
        assert "prompt_version" in data
        assert "response" in data
        assert "created_at" in data
        assert "expires_at" in data

    # =========================================================================
    # EXPIRATION TESTS
    # =========================================================================

    def test_expired_cache_returns_none(self, temp_cache_dir, sample_inputs):
        """Test that expired cache entries return None."""
        # Create cache with 0 days TTL (expired immediately)
        cache = LLMCache(cache_dir=temp_cache_dir, ttl_days=0)
        response = {"text": "Response"}

        cache.set("model", "v1.0", sample_inputs, response)

        # Wait a tiny bit to ensure expiration
        time.sleep(0.1)

        result = cache.get("model", "v1.0", sample_inputs)
        assert result is None

    def test_expired_cache_file_deleted(self, temp_cache_dir, sample_inputs):
        """Test that expired cache file is deleted on get."""
        cache = LLMCache(cache_dir=temp_cache_dir, ttl_days=0)
        response = {"text": "Response"}

        # Mock random to prevent probabilistic cleanup during set()
        with patch("random.random", return_value=1.0):
            cache.set("model", "v1.0", sample_inputs, response)
        cache_files_before = list(temp_cache_dir.glob("*.json"))
        assert len(cache_files_before) == 1

        time.sleep(0.1)
        cache.get("model", "v1.0", sample_inputs)

        cache_files_after = list(temp_cache_dir.glob("*.json"))
        assert len(cache_files_after) == 0

    # =========================================================================
    # CLEAR EXPIRED TESTS
    # =========================================================================

    def test_clear_expired_removes_old_files(self, temp_cache_dir):
        """Test that clear_expired removes expired files."""
        cache = LLMCache(cache_dir=temp_cache_dir, ttl_days=0)

        # Mock random to prevent probabilistic cleanup during set()
        with patch("random.random", return_value=1.0):
            # Create multiple cache entries
            cache.set("model", "v1.0", {"a": "1"}, {"text": "1"})
            cache.set("model", "v1.0", {"b": "2"}, {"text": "2"})
            cache.set("model", "v1.0", {"c": "3"}, {"text": "3"})

        assert len(list(temp_cache_dir.glob("*.json"))) == 3

        time.sleep(0.1)
        count = cache.clear_expired()

        assert count == 3
        assert len(list(temp_cache_dir.glob("*.json"))) == 0

    def test_clear_expired_keeps_valid_files(self, temp_cache_dir):
        """Test that clear_expired keeps non-expired files."""
        cache = LLMCache(cache_dir=temp_cache_dir, ttl_days=7)

        cache.set("model", "v1.0", {"a": "1"}, {"text": "1"})
        cache.set("model", "v1.0", {"b": "2"}, {"text": "2"})

        count = cache.clear_expired()

        assert count == 0
        assert len(list(temp_cache_dir.glob("*.json"))) == 2

    # =========================================================================
    # EDGE CASES AND ERROR HANDLING
    # =========================================================================

    def test_corrupted_json_handled(self, cache, temp_cache_dir, sample_inputs):
        """Test that corrupted JSON files are handled gracefully."""
        # Create a corrupted cache file
        cache.set("model", "v1.0", sample_inputs, {"text": "Response"})

        cache_files = list(temp_cache_dir.glob("*.json"))
        # Corrupt the file
        cache_files[0].write_text("{ invalid json }")

        # Clear memory cache to force disk read
        cache._mem_cache.clear()

        # Should return None and delete corrupted file
        result = cache.get("model", "v1.0", sample_inputs)
        assert result is None

        # File should be deleted
        assert not cache_files[0].exists()

    def test_missing_keys_in_cache_file(self, cache, temp_cache_dir, sample_inputs):
        """Test handling of cache file missing required keys."""
        cache.set("model", "v1.0", sample_inputs, {"text": "Response"})

        cache_files = list(temp_cache_dir.glob("*.json"))
        # Write incomplete data
        cache_files[0].write_text('{"model": "test"}')  # Missing expires_at

        # Clear memory cache to force disk read
        cache._mem_cache.clear()

        result = cache.get("model", "v1.0", sample_inputs)
        assert result is None

    def test_empty_inputs_handled(self, cache):
        """Test that empty inputs dict is handled."""
        response = {"text": "Response"}
        cache.set("model", "v1.0", {}, response)

        result = cache.get("model", "v1.0", {})
        assert result == response

    def test_complex_inputs_handled(self, cache):
        """Test handling of complex nested inputs."""
        complex_inputs = {
            "nested": {
                "level1": {
                    "level2": "value",
                },
            },
            "list": [1, 2, 3],
            "unicode": "español 日本語",
        }
        response = {"text": "Response"}

        cache.set("model", "v1.0", complex_inputs, response)
        result = cache.get("model", "v1.0", complex_inputs)

        assert result == response

    def test_response_not_dict_handled(self, cache, temp_cache_dir, sample_inputs):
        """Test that non-dict response in cache is handled."""
        cache.set("model", "v1.0", sample_inputs, {"text": "Response"})

        # Manually modify to have non-dict response
        cache_files = list(temp_cache_dir.glob("*.json"))
        with open(cache_files[0]) as f:
            data = json.load(f)
        data["response"] = "not a dict"
        with open(cache_files[0], "w") as f:
            json.dump(data, f)

        # Clear memory cache to force disk read
        cache._mem_cache.clear()

        result = cache.get("model", "v1.0", sample_inputs)
        assert result is None

    def test_large_response_stored(self, cache, sample_inputs):
        """Test that large responses can be stored and retrieved."""
        large_text = "x" * 100_000  # 100KB of text
        response = {"text": large_text}

        cache.set("model", "v1.0", sample_inputs, response)
        result = cache.get("model", "v1.0", sample_inputs)

        assert result == response
        assert len(result["text"]) == 100_000
