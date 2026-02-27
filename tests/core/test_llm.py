"""Tests for core.llm module (Claude CLI wrapper)."""

import subprocess
from unittest.mock import patch

import pytest

from core.llm import (
    LLMConfigurationError,
    LLMProviderError,
    call_llm,
    is_model_configured,
)


class TestCallLlm:
    """Tests for call_llm function."""

    @pytest.fixture(autouse=True)
    def _mock_settings(self):
        """Mock settings for all tests."""
        with patch("core.llm.settings") as mock_settings:
            mock_settings.DEFAULT_LLM_MODEL = "sonnet"
            mock_settings.CLAUDE_CLI_TIMEOUT = 180
            mock_settings.CLAUDE_CLI_PATH = "claude"
            self.mock_settings = mock_settings
            yield

    def test_call_llm_success(self):
        """Test successful CLI call returns stripped stdout."""
        with patch("core.llm.shutil.which", return_value="/usr/bin/claude"):
            with patch("core.llm.subprocess.run") as mock_run:
                mock_run.return_value = subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="  Response text  \n", stderr=""
                )

                result = call_llm("Test prompt")

                assert result == "Response text"

    def test_call_llm_uses_stdin(self):
        """Test that prompt is sent via stdin (input= parameter)."""
        with patch("core.llm.shutil.which", return_value="/usr/bin/claude"):
            with patch("core.llm.subprocess.run") as mock_run:
                mock_run.return_value = subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="OK", stderr=""
                )

                call_llm("My prompt text")

                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["input"] == "My prompt text"

    def test_call_llm_default_model(self):
        """Test that default model from settings is used."""
        with patch("core.llm.shutil.which", return_value="/usr/bin/claude"):
            with patch("core.llm.subprocess.run") as mock_run:
                mock_run.return_value = subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="OK", stderr=""
                )

                call_llm("prompt")

                call_args = mock_run.call_args[0][0]
                model_idx = call_args.index("--model")
                assert call_args[model_idx + 1] == "sonnet"

    def test_call_llm_custom_model(self):
        """Test that explicit model overrides default."""
        with patch("core.llm.shutil.which", return_value="/usr/bin/claude"):
            with patch("core.llm.subprocess.run") as mock_run:
                mock_run.return_value = subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="OK", stderr=""
                )

                call_llm("prompt", model="opus")

                call_args = mock_run.call_args[0][0]
                model_idx = call_args.index("--model")
                assert call_args[model_idx + 1] == "opus"

    def test_call_llm_timeout(self):
        """Test subprocess.TimeoutExpired raises LLMProviderError."""
        with patch("core.llm.shutil.which", return_value="/usr/bin/claude"):
            with patch("core.llm.subprocess.run") as mock_run:
                mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=180)

                with pytest.raises(LLMProviderError, match="timed out"):
                    call_llm("prompt")

    def test_call_llm_nonzero_exit(self):
        """Test non-zero exit code raises LLMProviderError."""
        with patch("core.llm.shutil.which", return_value="/usr/bin/claude"):
            with patch("core.llm.subprocess.run") as mock_run:
                mock_run.return_value = subprocess.CompletedProcess(
                    args=[], returncode=1, stdout="", stderr="Error: something went wrong"
                )

                with pytest.raises(LLMProviderError, match="exit 1"):
                    call_llm("prompt")

    def test_call_llm_empty_response(self):
        """Test empty stdout raises LLMProviderError."""
        with patch("core.llm.shutil.which", return_value="/usr/bin/claude"):
            with patch("core.llm.subprocess.run") as mock_run:
                mock_run.return_value = subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="   \n", stderr=""
                )

                with pytest.raises(LLMProviderError, match="empty response"):
                    call_llm("prompt")

    def test_call_llm_cli_not_found(self):
        """Test missing CLI raises LLMConfigurationError."""
        with patch("core.llm.shutil.which", return_value=None):
            with pytest.raises(LLMConfigurationError, match="not found"):
                call_llm("prompt")


class TestIsModelConfigured:
    """Tests for is_model_configured function."""

    def test_found(self):
        """Test CLI in PATH returns (True, "")."""
        with patch("core.llm.settings") as mock_settings:
            mock_settings.CLAUDE_CLI_PATH = "claude"
            with patch("core.llm.shutil.which", return_value="/usr/bin/claude"):
                ok, reason = is_model_configured("sonnet")
                assert ok is True
                assert reason == ""

    def test_missing(self):
        """Test CLI not in PATH returns (False, message)."""
        with patch("core.llm.settings") as mock_settings:
            mock_settings.CLAUDE_CLI_PATH = "claude"
            with patch("core.llm.shutil.which", return_value=None):
                ok, reason = is_model_configured("sonnet")
                assert ok is False
                assert "not found" in reason
