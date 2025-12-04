"""Shared utilities for LLM caching across modules.

Now supports multi-provider dispatch based on the model name prefix:

- "openai:<model>"  -> OpenAI Chat Completions API
- "anthropic:<model>" -> Anthropic Messages API
- "gemini:<model>"  -> Google Gemini API
- "<model>"         -> Defaults to Gemini (backwards compatible)

All existing call sites keep using `call_gemini_with_cache`, which will
intelligently route to the correct provider while preserving cache keys
and error semantics.
"""

import logging
import os
import time
from collections import deque

import google.generativeai as genai

# OpenAI and Anthropic are optional; we'll import them lazily inside the call
# when needed to avoid adding hard dependencies for users who only use Gemini.
from core.cache import LLMCache
from core.settings import settings


logger = logging.getLogger(__name__)

# Global cache instance
_cache: LLMCache | None = None
_call_times: deque[float] = deque()  # timestamps (seconds) for simple QPS limiting

# Simple pricing tables (USD per 1M tokens) for rough cost estimation
_PRICING = {
    "gemini": {
        "gemini-2.5-pro": (1.25, 10.0),
        "gemini-2.5-flash": (0.30, 2.50),
        "gemini-2.5-flash-lite": (0.10, 0.40),
        "gemini-2.0-flash": (0.30, 2.50),
        "gemini-2.0-flash-lite": (0.10, 0.40),
    },
    "openai": {
        "gpt-5": (1.25, 10.0),
        "gpt-5-mini": (0.25, 2.0),
        "gpt-5-nano": (0.05, 0.40),
        "gpt-4o-mini": (0.15, 0.60),
    },
    "anthropic": {
        "claude-4.5-sonnet": (3.0, 15.0),
        "claude-4-sonnet": (3.0, 15.0),
        "claude-3.7-sonnet": (3.0, 15.0),
        "claude-3.5-haiku": (0.80, 4.0),
        "claude-3-haiku": (0.25, 1.25),
    },
}


def _estimate_tokens(text: str) -> int:
    """Rough token estimation from text length.

    Assumes ~4 chars per token on average for Latin text.
    """
    return max(1, int(len(text) / 4))


def _get_model_pricing(provider: str, model_name: str) -> tuple[float, float] | None:
    pricing = _PRICING.get(provider, {})
    # Exact match first
    if model_name in pricing:
        return pricing[model_name]
    # Fallback: partial match
    for key, value in pricing.items():
        if key in model_name:
            return value
    return None


def get_llm_cache() -> LLMCache:
    """Get or create the global LLM cache instance."""
    global _cache
    if _cache is None:
        _cache = LLMCache(
            cache_dir=settings.LLM_CACHE_DIR,
            ttl_days=settings.LLM_CACHE_TTL_DAYS,
        )
    return _cache


def call_gemini_with_cache(
    prompt: str,
    model_name: str,
    prompt_version: str,
    inputs: dict,
    temperature: float = 0.7,
    error_class: type[Exception] = Exception,
) -> str:
    """Call an LLM (Gemini by default, OpenAI/Anthropic if prefixed) with intelligent caching.

    This is a shared utility function used across all modules that call LLMs.
    It provides consistent caching behavior based on content hashing.

    Cache key = hash(model_name + prompt_version + inputs)

    This ensures:
    - Same inputs + same prompt version → cache HIT (instant + free)
    - Different inputs → cache MISS (new generation)
    - Updated prompt version → cache MISS (regenerates with new prompt)

    Args:
        prompt: Full prompt text
        model_name: Model name with optional provider prefix (e.g., "gemini-1.5-flash-8b", "openai:gpt-4o-mini")
        prompt_version: Prompt template version for cache invalidation (e.g., "v1.0")
        inputs: Dictionary of input data (transcript, title, etc.)
        temperature: Generation temperature (default: 0.7)
        error_class: Exception class to raise on errors (default: Exception)

    Returns:
        Generated text from LLM

    Raises:
        error_class: If API call fails
    """
    # Determine provider + normalized model id (supports provider prefixes)
    provider = "gemini"
    normalized_model = model_name
    if ":" in model_name:
        maybe_provider, remainder = model_name.split(":", 1)
        if maybe_provider.lower() in {"openai", "gemini", "anthropic"}:
            provider = maybe_provider.lower()
            normalized_model = remainder

    if not settings.LLM_CACHE_ENABLED:
        # Cache disabled - direct API call to the selected provider
        logger.debug(f"Cache disabled - calling {provider} API directly")
        return _call_llm_direct(
            provider=provider,
            model_name=normalized_model,
            prompt=prompt,
            temperature=temperature,
            error_class=error_class,
        )

    # Try cache first
    cache = get_llm_cache()
    cached = cache.get(model_name, prompt_version, inputs)

    if cached:
        # Cache hit!
        cached_text: str = cached["text"]
        return cached_text

    # Cache miss - call the selected provider API
    logger.debug(f"Cache MISS - Calling {provider.upper()} API ({normalized_model})...")
    start_time = time.time()

    # Simple rate limiting (requests per minute)
    try:
        if settings.LLM_QPS and settings.LLM_QPS > 0:
            window = 60.0
            now = time.time()
            while _call_times and now - _call_times[0] > window:
                _call_times.popleft()
            if len(_call_times) >= settings.LLM_QPS:
                sleep_time = window - (now - _call_times[0]) + 0.01
                if sleep_time > 0:
                    time.sleep(sleep_time)
            _call_times.append(time.time())
    except Exception:
        # Never fail due to rate limiter
        pass

    try:
        response_text: str = _call_llm_direct(
            provider=provider,
            model_name=normalized_model,
            prompt=prompt,
            temperature=temperature,
            error_class=error_class,
        )
        elapsed = time.time() - start_time

        # Rough cost estimation based on prompt/response sizes and pricing tables
        est_cost = 0.0
        try:
            pricing = _get_model_pricing(provider, normalized_model)
            if pricing:
                in_price, out_price = pricing
                in_tokens = _estimate_tokens(prompt)
                out_tokens = _estimate_tokens(response_text)
                est_cost = (in_tokens / 1_000_000) * in_price + (out_tokens / 1_000_000) * out_price
        except Exception:
            est_cost = 0.0

        # Save to cache (store cost estimate and elapsed time)
        cache.set(
            model_name,  # keep original (with prefix) to partition caches by provider
            prompt_version,
            inputs,
            response={"text": response_text},
            cost=est_cost,
            time_saved=elapsed,
        )

        return response_text

    except Exception as e:
        logger.error(f"LLM API call failed ({provider}): {e}")
        raise error_class(f"LLM API call failed ({provider}): {e}") from e


def _call_llm_direct(
    provider: str,
    model_name: str,
    prompt: str,
    temperature: float,
    error_class: type[Exception],
) -> str:
    """Call the specific provider API without caching.

    Args:
        provider: "gemini" | "openai" | "anthropic"
        model_name: Provider-specific model identifier (no prefix)
        prompt: Full prompt text
        temperature: Generation temperature
        error_class: Exception class to raise on errors
    """
    if provider == "gemini":
        if not settings.GOOGLE_API_KEY:
            raise error_class(
                "GOOGLE_API_KEY no configurada. Establece la variable de entorno para usar modelos Gemini."
            )
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt, generation_config={"temperature": temperature})
        return response.text

    if provider == "openai":
        if not getattr(settings, "OPENAI_API_KEY", None) and not os.environ.get("OPENAI_API_KEY"):
            raise error_class(
                "OPENAI_API_KEY no configurada. Establece la variable de entorno para usar modelos OpenAI."
            )
        try:
            from openai import OpenAI  # type: ignore
        except Exception as ie:
            raise error_class(
                "Paquete 'openai' no instalado. Añade 'openai' a requirements.txt e instala dependencias."
            ) from ie

        api_key = getattr(settings, "OPENAI_API_KEY", None) or os.environ.get("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        # Use Chat Completions with a single user message.
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        content = resp.choices[0].message.content
        # content can be str or list of content parts; normalize to str
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
            return "".join(text_parts)
        return content or ""

    if provider == "anthropic":
        if not getattr(settings, "ANTHROPIC_API_KEY", None) and not os.environ.get(
            "ANTHROPIC_API_KEY"
        ):
            raise error_class(
                "ANTHROPIC_API_KEY no configurada. Establece la variable de entorno para usar modelos Anthropic."
            )
        try:
            from anthropic import Anthropic  # type: ignore
        except Exception as ie:
            raise error_class(
                "Paquete 'anthropic' no instalado. Añade 'anthropic' a requirements.txt e instala dependencias."
            ) from ie

        api_key = getattr(settings, "ANTHROPIC_API_KEY", None) or os.environ.get(
            "ANTHROPIC_API_KEY"
        )
        client = Anthropic(api_key=api_key)
        # Use Messages API with a single user message
        msg = client.messages.create(
            model=model_name,
            max_tokens=4096,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        # Concatenate text content from blocks
        parts = []
        for block in getattr(msg, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts)

    # Fallback defensivo
    raise error_class(f"Proveedor LLM desconocido: {provider}")


# Backwards-compatible alias (now supports provider prefixes)
call_llm_with_cache = call_gemini_with_cache


def is_model_configured(model_name: str) -> tuple[bool, str | None]:
    """Check whether the required API key is configured for a provider-prefixed model.

    Returns (ok, reason_if_not_ok).
    """
    provider = "gemini"
    if ":" in model_name:
        pfx, _ = model_name.split(":", 1)
        provider = pfx.lower()

    if provider == "gemini":
        return (bool(settings.GOOGLE_API_KEY), "Falta GOOGLE_API_KEY")
    if provider == "openai":
        import os as _os

        has_key = bool(getattr(settings, "OPENAI_API_KEY", "") or _os.environ.get("OPENAI_API_KEY"))
        return (has_key, "Falta OPENAI_API_KEY")
    if provider == "anthropic":
        import os as _os

        has_key = bool(
            getattr(settings, "ANTHROPIC_API_KEY", "") or _os.environ.get("ANTHROPIC_API_KEY")
        )
        return (has_key, "Falta ANTHROPIC_API_KEY")

    return False, f"Proveedor desconocido: {provider}"


__all__ = [
    "call_gemini_with_cache",
    "call_llm_with_cache",
    "get_llm_cache",
    "is_model_configured",
]
