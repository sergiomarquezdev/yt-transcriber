"""Tests for core.media_transcriber module."""

from unittest.mock import MagicMock, patch

import pytest

from core.media_transcriber import (
    TranscriptionError,
    TranscriptionResult,
    transcribe_audio_file,
)


def _make_segments(*texts):
    """Helper to create mock segments from text strings."""
    segments = []
    for text in texts:
        seg = MagicMock()
        seg.text = text
        segments.append(seg)
    return segments


def _make_info(language="en"):
    """Helper to create a mock transcription info object."""
    info = MagicMock()
    info.language = language
    return info


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

        call_kwargs = mock_whisper_model.transcribe.call_args[1]
        assert call_kwargs.get("language") == "es"

    def test_file_not_found_raises_error(self, temp_dir, mock_whisper_model):
        """Test that non-existent file raises TranscriptionError."""
        non_existent = temp_dir / "does_not_exist.wav"

        with pytest.raises(TranscriptionError, match="not found"):
            transcribe_audio_file(
                audio_path=non_existent,
                model=mock_whisper_model,
            )

    def test_empty_transcription_handled(self, sample_audio_path, mock_whisper_model):
        """Test that empty transcription is handled."""
        mock_whisper_model.transcribe.return_value = (_make_segments(""), _make_info())

        result = transcribe_audio_file(
            audio_path=sample_audio_path,
            model=mock_whisper_model,
        )

        assert result.text == ""

    def test_whitespace_trimmed(self, sample_audio_path, mock_whisper_model):
        """Test that whitespace is trimmed from transcription."""
        mock_whisper_model.transcribe.return_value = (
            _make_segments("  Some text with spaces  "),
            _make_info(),
        )

        result = transcribe_audio_file(
            audio_path=sample_audio_path,
            model=mock_whisper_model,
        )

        assert result.text == "Some text with spaces"

    def test_whisper_error_wrapped(self, sample_audio_path, mock_whisper_model):
        """Test that Whisper errors are wrapped in TranscriptionError."""
        mock_whisper_model.transcribe.side_effect = RuntimeError("Whisper crashed")

        with pytest.raises(TranscriptionError, match="Unexpected"):
            transcribe_audio_file(
                audio_path=sample_audio_path,
                model=mock_whisper_model,
            )

    def test_detected_language_returned(self, sample_audio_path, mock_whisper_model):
        """Test that detected language is returned."""
        mock_whisper_model.transcribe.return_value = (
            _make_segments("Hola mundo"),
            _make_info("es"),
        )

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
        assert isinstance(call_args[0], str)

    def test_long_audio_handled(self, temp_dir, mock_whisper_model):
        """Test handling of long audio transcription."""
        audio_path = temp_dir / "long_audio.wav"
        audio_path.touch()

        words = ["Word"] * 10000
        segments = _make_segments(*words)
        mock_whisper_model.transcribe.return_value = (segments, _make_info())

        result = transcribe_audio_file(
            audio_path=audio_path,
            model=mock_whisper_model,
        )

        assert len(result.text.split()) == 10000

    def test_special_characters_preserved(self, sample_audio_path, mock_whisper_model):
        """Test that special characters are preserved."""
        mock_whisper_model.transcribe.return_value = (
            _make_segments("Hello! ¿Cómo estás? 日本語"),
            _make_info(),
        )

        result = transcribe_audio_file(
            audio_path=sample_audio_path,
            model=mock_whisper_model,
        )

        assert "¿Cómo" in result.text
        assert "日本語" in result.text

    def test_beam_size_passed(self, sample_audio_path, mock_whisper_model):
        """Test that beam_size from settings is passed to transcribe."""
        with patch("core.media_transcriber.settings") as mock_settings:
            mock_settings.WHISPER_BEAM_SIZE = 3
            mock_settings.WHISPER_VAD_FILTER = True

            transcribe_audio_file(
                audio_path=sample_audio_path,
                model=mock_whisper_model,
            )

            call_kwargs = mock_whisper_model.transcribe.call_args[1]
            assert call_kwargs["beam_size"] == 3

    def test_vad_filter_passed(self, sample_audio_path, mock_whisper_model):
        """Test that vad_filter from settings is passed to transcribe."""
        with patch("core.media_transcriber.settings") as mock_settings:
            mock_settings.WHISPER_BEAM_SIZE = 5
            mock_settings.WHISPER_VAD_FILTER = False

            transcribe_audio_file(
                audio_path=sample_audio_path,
                model=mock_whisper_model,
            )

            call_kwargs = mock_whisper_model.transcribe.call_args[1]
            assert call_kwargs["vad_filter"] is False

    def test_multiple_segments_joined(self, sample_audio_path, mock_whisper_model):
        """Test that multiple segments are joined with spaces."""
        mock_whisper_model.transcribe.return_value = (
            _make_segments("Hello world.", "How are you?", "Fine thanks."),
            _make_info(),
        )

        result = transcribe_audio_file(
            audio_path=sample_audio_path,
            model=mock_whisper_model,
        )

        assert result.text == "Hello world. How are you? Fine thanks."
