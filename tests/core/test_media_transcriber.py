"""Tests for core.media_transcriber module."""

from unittest.mock import MagicMock

import pytest

from core.media_transcriber import (
    TranscriptionError,
    TranscriptionResult,
    transcribe_audio_file,
)


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""

    def test_basic_creation(self):
        """Test basic TranscriptionResult creation."""
        result = TranscriptionResult(
            text="Hello world, this is a test.",
            language="en",
        )

        assert result.text == "Hello world, this is a test."
        assert result.language == "en"

    def test_creation_without_language(self):
        """Test TranscriptionResult without language."""
        result = TranscriptionResult(text="Some text")

        assert result.text == "Some text"
        assert result.language is None


class TestTranscribeAudioFile:
    """Tests for transcribe_audio_file function."""

    def test_successful_transcription(self, sample_audio_path, mock_whisper_model):
        """Test successful audio transcription."""
        result = transcribe_audio_file(
            audio_path=sample_audio_path,
            model=mock_whisper_model,
        )

        assert result.text == "This is a sample transcription."
        assert result.language == "en"
        mock_whisper_model.transcribe.assert_called_once()

    def test_transcription_with_language(self, sample_audio_path, mock_whisper_model):
        """Test transcription with specified language."""
        transcribe_audio_file(
            audio_path=sample_audio_path,
            model=mock_whisper_model,
            language="es",
        )

        # Verify language was passed to transcribe
        call_kwargs = mock_whisper_model.transcribe.call_args[1]
        assert call_kwargs.get("language") == "es"

    def test_file_not_found_raises_error(self, temp_dir, mock_whisper_model):
        """Test that non-existent file raises TranscriptionError."""
        non_existent = temp_dir / "does_not_exist.wav"

        with pytest.raises(TranscriptionError, match="no encontrado"):
            transcribe_audio_file(
                audio_path=non_existent,
                model=mock_whisper_model,
            )

    def test_empty_transcription_handled(self, sample_audio_path, mock_whisper_model):
        """Test that empty transcription is handled."""
        mock_whisper_model.transcribe.return_value = {
            "text": "",
            "language": "en",
        }

        result = transcribe_audio_file(
            audio_path=sample_audio_path,
            model=mock_whisper_model,
        )

        assert result.text == ""

    def test_whitespace_trimmed(self, sample_audio_path, mock_whisper_model):
        """Test that whitespace is trimmed from transcription."""
        mock_whisper_model.transcribe.return_value = {
            "text": "  Some text with spaces  ",
            "language": "en",
        }

        result = transcribe_audio_file(
            audio_path=sample_audio_path,
            model=mock_whisper_model,
        )

        assert result.text == "Some text with spaces"

    def test_whisper_error_wrapped(self, sample_audio_path, mock_whisper_model):
        """Test that Whisper errors are wrapped in TranscriptionError."""
        mock_whisper_model.transcribe.side_effect = RuntimeError("Whisper crashed")

        with pytest.raises(TranscriptionError, match="inesperado"):
            transcribe_audio_file(
                audio_path=sample_audio_path,
                model=mock_whisper_model,
            )

    def test_fp16_enabled_on_cuda(self, sample_audio_path):
        """Test that FP16 is enabled when model is on CUDA."""
        mock_model = MagicMock()
        mock_model.device = MagicMock()
        mock_model.device.type = "cuda"
        mock_model.transcribe.return_value = {"text": "Test", "language": "en"}

        transcribe_audio_file(
            audio_path=sample_audio_path,
            model=mock_model,
        )

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs.get("fp16") is True

    def test_fp16_disabled_on_cpu(self, sample_audio_path):
        """Test that FP16 is disabled when model is on CPU."""
        mock_model = MagicMock()
        mock_model.device = MagicMock()
        mock_model.device.type = "cpu"
        mock_model.transcribe.return_value = {"text": "Test", "language": "en"}

        transcribe_audio_file(
            audio_path=sample_audio_path,
            model=mock_model,
        )

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs.get("fp16") is False

    def test_detected_language_returned(self, sample_audio_path, mock_whisper_model):
        """Test that detected language is returned."""
        mock_whisper_model.transcribe.return_value = {
            "text": "Hola mundo",
            "language": "es",
        }

        result = transcribe_audio_file(
            audio_path=sample_audio_path,
            model=mock_whisper_model,
        )

        assert result.language == "es"

    def test_path_converted_to_string(self, sample_audio_path, mock_whisper_model):
        """Test that Path is converted to string for Whisper."""
        transcribe_audio_file(
            audio_path=sample_audio_path,
            model=mock_whisper_model,
        )

        call_args = mock_whisper_model.transcribe.call_args[0]
        # First positional argument should be string path
        assert isinstance(call_args[0], str)

    def test_long_audio_handled(self, temp_dir, mock_whisper_model):
        """Test handling of long audio transcription."""
        audio_path = temp_dir / "long_audio.wav"
        audio_path.touch()

        # Simulate long transcription
        long_text = "Word " * 10000
        mock_whisper_model.transcribe.return_value = {
            "text": long_text,
            "language": "en",
        }

        result = transcribe_audio_file(
            audio_path=audio_path,
            model=mock_whisper_model,
        )

        assert len(result.text.split()) == 10000

    def test_special_characters_preserved(self, sample_audio_path, mock_whisper_model):
        """Test that special characters are preserved."""
        special_text = "Hello! ¿Cómo estás? 日本語 🎉"
        mock_whisper_model.transcribe.return_value = {
            "text": special_text,
            "language": "en",
        }

        result = transcribe_audio_file(
            audio_path=sample_audio_path,
            model=mock_whisper_model,
        )

        assert "¿Cómo" in result.text
        assert "日本語" in result.text
