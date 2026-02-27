"""Shared Whisper transcription utilities.

Centralizes the transcription logic to be reused across pipelines.
"""

import dataclasses
import logging
from pathlib import Path
from typing import Any

from core.settings import settings

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TranscriptionResult:
    """Holds the result of a transcription."""

    text: str
    language: str | None = None


class TranscriptionError(Exception):
    """Custom exception for transcription errors."""

    pass


def transcribe_audio_file(
    audio_path: Path,
    model: Any,
    language: str | None = None,
) -> TranscriptionResult:
    """Transcribe an audio file using a preloaded faster-whisper model.

    Args:
        audio_path: Path to WAV audio file
        model: Preloaded WhisperModel instance
        language: Optional language code (e.g., 'en', 'es'); None to auto-detect

    Returns:
        TranscriptionResult with transcribed text and detected language

    Raises:
        TranscriptionError: If the audio file does not exist or transcription fails
    """
    logger.info(f"Starting transcription for: {audio_path} with preloaded Whisper model.")

    if not audio_path.exists():
        logger.error(f"Transcription error: Audio file not found at {audio_path}")
        raise TranscriptionError(f"Audio file not found: {audio_path}")

    try:
        logger.info(f"Transcribing file: {audio_path} language='{language}'")

        transcribe_options: dict = {
            "beam_size": settings.WHISPER_BEAM_SIZE,
            "vad_filter": settings.WHISPER_VAD_FILTER,
        }
        if language:
            transcribe_options["language"] = language

        segments, info = model.transcribe(str(audio_path), **transcribe_options)

        transcribed_text = " ".join(segment.text.strip() for segment in segments).strip()
        if not transcribed_text:
            logger.warning("Transcription returned empty text.")

        detected_language = info.language

        logger.info(
            f"Transcription complete. Detected language: {detected_language}. Length: {len(transcribed_text)} chars."
        )
        return TranscriptionResult(text=transcribed_text, language=detected_language)

    except Exception as e:
        logger.error(
            f"Unexpected error during transcription of '{audio_path}': {e}",
            exc_info=True,
        )
        raise TranscriptionError(f"Unexpected Whisper error: {e}") from e
