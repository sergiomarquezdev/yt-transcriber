"""Shared test fixtures and configuration."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# =============================================================================
# DIRECTORY FIXTURES
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    path = Path(tempfile.mkdtemp())
    yield path
    # Cleanup after test
    if path.exists():
        shutil.rmtree(path)


@pytest.fixture
def temp_output_dir(temp_dir):
    """Create a temporary output directory."""
    output_dir = temp_dir / "output"
    output_dir.mkdir(parents=True)
    return output_dir


# =============================================================================
# AUDIO/VIDEO FIXTURES
# =============================================================================


@pytest.fixture
def sample_audio_path(temp_dir):
    """Create a sample audio file path (not real audio)."""
    audio_path = temp_dir / "sample_audio.wav"
    # Create empty file to simulate audio
    audio_path.touch()
    return audio_path


@pytest.fixture
def sample_video_path(temp_dir):
    """Create a sample video file path (not real video)."""
    video_path = temp_dir / "sample_video.mp4"
    video_path.touch()
    return video_path


# =============================================================================
# MOCK WHISPER MODEL
# =============================================================================


@pytest.fixture
def mock_whisper_model():
    """Create a mock faster-whisper model."""
    model = MagicMock()
    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 2.5
    mock_segment.text = "This is a sample transcription."
    mock_info = MagicMock()
    mock_info.language = "en"
    model.transcribe.return_value = ([mock_segment], mock_info)
    return model


# =============================================================================
# SAMPLE DATA FIXTURES
# =============================================================================


@pytest.fixture
def sample_transcript():
    """Sample transcript text for testing."""
    return """
    Welcome to this video about Python programming.
    Today we will learn about testing with pytest.
    Testing is an important part of software development.
    It helps us catch bugs early and ensures code quality.
    Let's dive into the basics of unit testing.
    """


@pytest.fixture
def sample_transcript_es():
    """Sample Spanish transcript for testing."""
    return """
    Bienvenidos a este video sobre programacion en Python.
    Hoy aprenderemos sobre testing con pytest.
    El testing es una parte importante del desarrollo de software.
    Nos ayuda a detectar errores temprano y asegura la calidad del codigo.
    Vamos a ver los conceptos basicos de pruebas unitarias.
    """


# =============================================================================
# ENVIRONMENT FIXTURES
# =============================================================================


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing."""
    env_vars = {
        "WHISPER_MODEL_NAME": "tiny",
        "WHISPER_DEVICE": "cpu",
        "LOG_LEVEL": "DEBUG",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars


# =============================================================================
# YOUTUBE URL FIXTURES
# =============================================================================


@pytest.fixture
def youtube_urls():
    """Sample YouTube URLs for testing."""
    return {
        "standard": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "short": "https://youtu.be/dQw4w9WgXcQ",
        "with_timestamp": "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120",
        "invalid": "https://example.com/not-a-video",
    }


@pytest.fixture
def google_drive_urls():
    """Sample Google Drive URLs for testing."""
    return {
        "file_d": "https://drive.google.com/file/d/1ABC123xyz/view",
        "open_id": "https://drive.google.com/open?id=1ABC123xyz",
        "docs": "https://docs.google.com/file/d/1ABC123xyz/edit",
        "invalid": "https://drive.google.com/invalid/format",
    }
