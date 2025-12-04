"""Shared Whisper transcription utilities.

Centralizes the transcription logic to be reused across pipelines.
"""

import dataclasses
import logging
from pathlib import Path

import whisper


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
    model: whisper.Whisper,
    language: str | None = None,
) -> TranscriptionResult:
    """Transcribe an audio file using a preloaded Whisper model.

    Args:
        audio_path: Path to WAV audio file
        model: Preloaded Whisper model
        language: Optional language code (e.g., 'en', 'es'); None to auto-detect

    Returns:
        TranscriptionResult with transcribed text and detected language

    Raises:
        TranscriptionError: If the audio file does not exist or Whisper fails
    """
    logger.info(f"Starting transcription for: {audio_path} with preloaded Whisper model.")

    if not audio_path.exists():
        logger.error(f"Error de transcripción: Archivo de audio no encontrado en {audio_path}")
        # Spanish message to match legacy behavior/tests
        raise TranscriptionError(f"Archivo de audio no encontrado: {audio_path}")

    try:
        logger.info(f"Transcribing file: {audio_path} language='{language}'")

        transcribe_options: dict = {
            "fp16": getattr(model, "device", None) and model.device.type == "cuda"
        }
        if language:
            transcribe_options["language"] = language

        result = model.transcribe(str(audio_path), **transcribe_options)

        transcribed_text = result.get("text", "").strip()
        if not transcribed_text:
            # Spanish message to match tests
            logger.warning("La transcripción ha devuelto un texto vacío.")

        detected_language = result.get("language")

        logger.info(
            f"Transcripción completada. Idioma detectado: {detected_language}. Longitud: {len(transcribed_text)} chars."
        )
        return TranscriptionResult(text=transcribed_text, language=detected_language)

    except Exception as e:
        logger.error(
            f"Error inesperado durante transcripción de '{audio_path}': {e}",
            exc_info=True,
        )
        # Spanish message for compatibility
        raise TranscriptionError(f"Error inesperado en Whisper: {e}") from e
