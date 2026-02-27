"""Claude CLI wrapper for LLM calls via subprocess.

Replaces multi-provider SDK dispatch with a single function that invokes
the `claude` CLI. Auth is handled by the CLI's own subscription (Max/Pro).
"""

import logging
import os
import shutil
import subprocess

from core.settings import settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base exception for LLM-related errors."""


class LLMProviderError(LLMError):
    """Error from LLM provider (CLI failure, timeout, etc.)."""


class LLMConfigurationError(LLMError):
    """Error in LLM configuration (CLI not found, etc.)."""


def call_llm(
    prompt: str,
    model: str | None = None,
    timeout: int | None = None,
) -> str:
    """Call Claude CLI and return the response text.

    Args:
        prompt: Full prompt text (sent via stdin)
        model: Claude model name ("sonnet", "haiku", "opus"). Defaults to settings.DEFAULT_LLM_MODEL
        timeout: Timeout in seconds. Defaults to settings.CLAUDE_CLI_TIMEOUT

    Returns:
        Response text from Claude CLI

    Raises:
        LLMConfigurationError: If Claude CLI is not found in PATH
        LLMProviderError: If CLI returns non-zero exit, times out, or returns empty response
    """
    model = model if model is not None else settings.DEFAULT_LLM_MODEL
    timeout = timeout if timeout is not None else settings.CLAUDE_CLI_TIMEOUT
    cli_path = settings.CLAUDE_CLI_PATH

    if not shutil.which(cli_path):
        raise LLMConfigurationError(f"Claude CLI not found: '{cli_path}'")

    args = [
        cli_path,
        "-p",
        "--model",
        model,
        "--output-format",
        "text",
        "--max-turns",
        "1",
    ]

    # Clean env to allow running from within a Claude Code session
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)

    try:
        result = subprocess.run(
            args,
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        raise LLMProviderError(f"Claude CLI timed out after {timeout}s") from e
    except OSError as e:
        raise LLMProviderError(f"Failed to execute Claude CLI: {e}") from e

    if result.returncode != 0:
        raise LLMProviderError(
            f"Claude CLI failed (exit {result.returncode}): {result.stderr.strip()}"
        )

    response = result.stdout.strip()
    if not response:
        raise LLMProviderError("Claude CLI returned empty response")

    return response


def is_model_configured(model_name: str) -> tuple[bool, str]:
    """Check if Claude CLI is available.

    Args:
        model_name: Model name (unused, kept for API compatibility)

    Returns:
        Tuple of (is_configured, reason_if_not)
    """
    if shutil.which(settings.CLAUDE_CLI_PATH):
        return True, ""
    return False, f"Claude CLI not found at '{settings.CLAUDE_CLI_PATH}'"


__all__ = [
    "call_llm",
    "is_model_configured",
    "LLMError",
    "LLMProviderError",
    "LLMConfigurationError",
]
