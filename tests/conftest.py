"""Shared test fixtures and configuration."""

import shutil
import tempfile
from datetime import datetime
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
def temp_cache_dir(temp_dir):
    """Create a temporary cache directory."""
    cache_dir = temp_dir / "cache"
    cache_dir.mkdir(parents=True)
    return cache_dir


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


@pytest.fixture
def sample_video_summary():
    """Create a sample VideoSummary for testing."""
    from core.models import TimestampedSection, VideoSummary

    return VideoSummary(
        video_url="https://youtube.com/watch?v=test123",
        video_title="Test Video Title",
        video_id="test123",
        executive_summary="This is a test executive summary.",
        key_points=[
            "First key point",
            "Second key point",
            "Third key point",
        ],
        timestamps=[
            TimestampedSection(
                timestamp="00:00",
                description="Introduction",
                importance=3,
            ),
            TimestampedSection(
                timestamp="05:30",
                description="Main content",
                importance=5,
            ),
        ],
        conclusion="This is the test conclusion.",
        action_items=["Action 1", "Action 2"],
        word_count=500,
        estimated_duration_minutes=3.5,
        language="en",
        generated_at=datetime(2024, 1, 1, 12, 0, 0),
    )


# =============================================================================
# ENVIRONMENT FIXTURES
# =============================================================================


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing."""
    env_vars = {
        "GOOGLE_API_KEY": "test_google_api_key",
        "OPENAI_API_KEY": "test_openai_api_key",
        "ANTHROPIC_API_KEY": "test_anthropic_api_key",
        "WHISPER_MODEL_NAME": "tiny",
        "WHISPER_DEVICE": "cpu",
        "LOG_LEVEL": "DEBUG",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars


@pytest.fixture
def clean_env(monkeypatch):
    """Remove all API key environment variables."""
    keys_to_remove = [
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ]
    for key in keys_to_remove:
        monkeypatch.delenv(key, raising=False)


# =============================================================================
# LLM RESPONSE FIXTURES
# =============================================================================


@pytest.fixture
def sample_llm_summary_response():
    """Sample LLM response for summary generation."""
    return """
# 📹 Summary: Test Video

## 🎯 Executive Summary
This video covers the basics of Python testing with pytest.

## 🔑 Key Points
1. **Testing basics**: Understanding unit tests
2. **Pytest features**: Fixtures and assertions
3. **Best practices**: Writing maintainable tests

## ⏱️ Important Moments
- **00:00** - Introduction to testing
- **05:30** - Setting up pytest
- **10:00** - Writing your first test

## 💡 Conclusion
Testing is essential for quality software.

## ✅ Action Items
1. Install pytest
2. Write your first test
3. Practice TDD
"""


@pytest.fixture
def sample_linkedin_response():
    """Sample LLM response for LinkedIn post generation."""
    return """
Hook: Python testing has never been easier

Intro: After years of writing untested code, pytest changed everything.

Insight1: 🔹 Simple syntax: Write tests that read like documentation
Insight2: 🔹 Powerful fixtures: Reusable test setup without boilerplate
Insight3: 🔹 Plugin ecosystem: Extend functionality for any use case
Insight4: 🔹 Fast execution: Run thousands of tests in seconds

WhyItMatters: Quality software starts with quality tests

CTA: What testing framework do you use?

Tags: #Python #Testing #SoftwareDevelopment
"""


@pytest.fixture
def sample_twitter_response():
    """Sample LLM response for Twitter thread generation."""
    return """
1. Just discovered pytest and my mind is blown 🚀
2. Simple syntax makes testing enjoyable, not a chore
3. Fixtures eliminate repetitive setup code entirely
4. Parametrize lets you test multiple scenarios in one function
5. The plugin system is incredibly powerful
6. Coverage reports show exactly what needs testing
7. Watch mode reruns tests on file changes
8. Integration with CI/CD is seamless
9. This is how testing should have always been
10. Ready to level up your Python testing game?

Hashtags: Python, Testing, DevTools
"""


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
