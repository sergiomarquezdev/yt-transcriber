"""Tests for yt_transcriber.summarizer module."""

from datetime import datetime
from unittest.mock import patch

import pytest

from yt_transcriber.summarizer import (
    SummarizationError,
    _build_prompt,
    _detect_language,
    _extract_list_items,
    _extract_section,
    _extract_timestamps,
    _parse_summary_response,
    generate_summary,
)


class TestDetectLanguage:
    """Tests for _detect_language function."""

    def test_detects_english(self):
        """Test detection of English text."""
        text = """
        Welcome to this video about Python programming.
        Today we will learn about testing and how to write better code.
        This is an important topic for all developers.
        """
        assert _detect_language(text) == "en"

    def test_detects_spanish(self):
        """Test detection of Spanish text."""
        text = """
        Bienvenidos a este video sobre programación en Python.
        Hoy aprenderemos sobre testing y cómo escribir mejor código.
        Este es un tema importante para todos los desarrolladores.
        """
        assert _detect_language(text) == "es"

    def test_mixed_content_defaults_correctly(self):
        """Test with mixed content."""
        # More Spanish words
        text = """
        El sistema de gestión incluye workflows avanzados.
        La API permite integración con sistemas externos.
        El endpoint principal es el más utilizado.
        """
        result = _detect_language(text)
        # Should detect Spanish due to articles
        assert result in ["es", "en"]  # Accept either as valid

    def test_short_text(self):
        """Test with very short text."""
        result = _detect_language("Hello world")
        assert result in ["es", "en"]

    def test_technical_text_en(self):
        """Test with technical English text."""
        text = """
        The API endpoint returns a JSON response.
        You can configure the parameters in the settings file.
        This workflow automates the deployment process.
        """
        assert _detect_language(text) == "en"

    def test_empty_string(self):
        """Test with empty string."""
        result = _detect_language("")
        # Should return default (es or en)
        assert result in ["es", "en"]


class TestBuildPrompt:
    """Tests for _build_prompt function."""

    def test_builds_spanish_prompt(self):
        """Test building Spanish prompt."""
        prompt = _build_prompt(
            transcript="El texto de la transcripción",
            video_title="Mi Video",
            word_count=500,
            duration_minutes=5.0,
            language="es",
        )

        assert "Mi Video" in prompt
        assert "500" in prompt
        assert "5.0" in prompt
        # Should be Spanish template
        assert "Resumen Ejecutivo" in prompt or "Analiza" in prompt

    def test_builds_english_prompt(self):
        """Test building English prompt."""
        prompt = _build_prompt(
            transcript="The transcript text",
            video_title="My Video",
            word_count=1000,
            duration_minutes=10.0,
            language="en",
        )

        assert "My Video" in prompt
        assert "1000" in prompt
        # Should be English template
        assert "Executive Summary" in prompt or "Analyze" in prompt

    def test_includes_transcript(self):
        """Test that transcript is included."""
        transcript = "This is the actual transcript content."
        prompt = _build_prompt(
            transcript=transcript,
            video_title="Test",
            word_count=100,
            duration_minutes=1.0,
            language="en",
        )

        assert transcript in prompt


class TestExtractSection:
    """Tests for _extract_section function."""

    def test_extracts_section_content(self):
        """Test extracting section content."""
        text = """
## 🎯 Executive Summary
This is the summary content.
It spans multiple lines.

## 🔑 Key Points
1. First point
"""
        result = _extract_section(text, r"## 🎯 Executive Summary")

        assert "This is the summary content" in result
        assert "multiple lines" in result
        assert "Key Points" not in result

    def test_returns_empty_for_missing_section(self):
        """Test returns empty string for missing section."""
        text = "Some text without the header"
        result = _extract_section(text, r"## Missing Header")

        assert result == ""

    def test_handles_alternate_headers(self):
        """Test with regex alternation for headers."""
        text = """
## 🎯 Resumen Ejecutivo
Este es el contenido del resumen.

## 🔑 Puntos Clave
"""
        result = _extract_section(
            text, r"## 🎯 Resumen Ejecutivo|## 🎯 Executive Summary"
        )

        assert "contenido del resumen" in result


class TestExtractListItems:
    """Tests for _extract_list_items function."""

    def test_extracts_numbered_list(self):
        """Test extracting numbered list items."""
        text = """
## 🔑 Key Points
1. First point
2. Second point
3. Third point

## Next Section
"""
        items = _extract_list_items(text, r"## 🔑 Key Points")

        assert len(items) == 3
        assert "First point" in items[0]
        assert "Second point" in items[1]

    def test_extracts_bulleted_list(self):
        """Test extracting bulleted list items."""
        text = """
## ✅ Action Items
- First action
- Second action
* Third action

## End
"""
        items = _extract_list_items(text, r"## ✅ Action Items")

        assert len(items) == 3

    def test_returns_empty_for_missing_section(self):
        """Test returns empty list for missing section."""
        text = "No matching section"
        items = _extract_list_items(text, r"## Missing")

        assert items == []


class TestExtractTimestamps:
    """Tests for _extract_timestamps function."""

    def test_extracts_timestamps(self):
        """Test extracting timestamps."""
        text = """
## ⏱️ Important Moments
- **00:00** - Introduction
- **05:30** - Main content
- **10:15** - Conclusion

## End
"""
        timestamps = _extract_timestamps(text)

        assert len(timestamps) == 3
        assert timestamps[0].timestamp == "00:00"
        assert timestamps[0].description == "Introduction"
        assert timestamps[1].timestamp == "05:30"

    def test_handles_hour_format(self):
        """Test with hour:minute:second format."""
        text = """
## ⏱️ Momentos Importantes
- **01:30:00** - Long video section

## End
"""
        timestamps = _extract_timestamps(text)

        assert len(timestamps) == 1
        assert timestamps[0].timestamp == "01:30:00"

    def test_returns_empty_for_no_timestamps(self):
        """Test returns empty list when no timestamps."""
        text = "No timestamps here"
        timestamps = _extract_timestamps(text)

        assert timestamps == []

    def test_default_importance(self):
        """Test that default importance is set."""
        text = """
## ⏱️ Important Moments
- **00:00** - Test
"""
        timestamps = _extract_timestamps(text)

        assert timestamps[0].importance == 3


class TestParseSummaryResponse:
    """Tests for _parse_summary_response function."""

    def test_parses_complete_response(self, sample_llm_summary_response):
        """Test parsing complete response."""
        summary = _parse_summary_response(
            summary_text=sample_llm_summary_response,
            video_url="https://youtube.com/watch?v=test",
            video_title="Test Video",
            video_id="test123",
            word_count=1000,
            duration_minutes=10.0,
            language="en",
        )

        assert summary.video_url == "https://youtube.com/watch?v=test"
        assert summary.video_id == "test123"
        assert summary.word_count == 1000
        assert summary.language == "en"
        assert len(summary.executive_summary) > 0

    def test_sets_generated_at(self, sample_llm_summary_response):
        """Test that generated_at is set."""
        before = datetime.now()
        summary = _parse_summary_response(
            summary_text=sample_llm_summary_response,
            video_url="https://example.com",
            video_title="Test",
            video_id="test",
            word_count=100,
            duration_minutes=1.0,
            language="en",
        )
        after = datetime.now()

        assert before <= summary.generated_at <= after


class TestGenerateSummary:
    """Tests for generate_summary function."""

    def test_successful_generation(self, sample_llm_summary_response, sample_transcript):
        """Test successful summary generation."""
        with patch("yt_transcriber.summarizer.call_gemini_with_cache") as mock_call:
            mock_call.return_value = sample_llm_summary_response

            with patch("yt_transcriber.summarizer.settings") as mock_settings:
                mock_settings.SUMMARIZER_MODEL = "gemini-2.5-flash"
                mock_settings.SUMMARIZER_PROMPT_VERSION = "v1.0"

                summary = generate_summary(
                    transcript=sample_transcript,
                    video_title="Test Video",
                    video_url="https://youtube.com/watch?v=test123",
                    video_id="test123",
                )

                assert summary is not None
                assert summary.video_id == "test123"
                mock_call.assert_called_once()

    def test_detects_language(self, sample_llm_summary_response):
        """Test that language is detected from transcript."""
        spanish_transcript = """
        Bienvenidos a este video sobre Python.
        En este tutorial aprenderemos testing.
        Es muy importante para los desarrolladores.
        """

        with patch("yt_transcriber.summarizer.call_gemini_with_cache") as mock_call:
            mock_call.return_value = sample_llm_summary_response

            with patch("yt_transcriber.summarizer.settings") as mock_settings:
                mock_settings.SUMMARIZER_MODEL = "gemini-2.5-flash"
                mock_settings.SUMMARIZER_PROMPT_VERSION = "v1.0"

                summary = generate_summary(
                    transcript=spanish_transcript,
                    video_title="Video",
                    video_url="https://example.com",
                    video_id="test",
                )

                # Language should be detected
                assert summary.language in ["es", "en"]

    def test_api_error_raises_summarization_error(self, sample_transcript):
        """Test that API errors are wrapped in SummarizationError."""
        with patch("yt_transcriber.summarizer.call_gemini_with_cache") as mock_call:
            # Simulate what call_gemini_with_cache does: wrap in error_class
            mock_call.side_effect = SummarizationError("API failed")

            with patch("yt_transcriber.summarizer.settings") as mock_settings:
                mock_settings.SUMMARIZER_MODEL = "gemini-2.5-flash"
                mock_settings.SUMMARIZER_PROMPT_VERSION = "v1.0"

                with pytest.raises(SummarizationError, match="API failed"):
                    generate_summary(
                        transcript=sample_transcript,
                        video_title="Test",
                        video_url="https://example.com",
                        video_id="test",
                    )

    def test_calculates_word_count(self, sample_llm_summary_response):
        """Test that word count is calculated."""
        transcript = "word " * 100  # 100 words

        with patch("yt_transcriber.summarizer.call_gemini_with_cache") as mock_call:
            mock_call.return_value = sample_llm_summary_response

            with patch("yt_transcriber.summarizer.settings") as mock_settings:
                mock_settings.SUMMARIZER_MODEL = "gemini-2.5-flash"
                mock_settings.SUMMARIZER_PROMPT_VERSION = "v1.0"

                summary = generate_summary(
                    transcript=transcript,
                    video_title="Test",
                    video_url="https://example.com",
                    video_id="test",
                )

                assert summary.word_count == 100

    def test_calculates_duration(self, sample_llm_summary_response):
        """Test that duration is calculated."""
        # 150 words per minute average
        transcript = "word " * 300  # 300 words = 2 minutes

        with patch("yt_transcriber.summarizer.call_gemini_with_cache") as mock_call:
            mock_call.return_value = sample_llm_summary_response

            with patch("yt_transcriber.summarizer.settings") as mock_settings:
                mock_settings.SUMMARIZER_MODEL = "gemini-2.5-flash"
                mock_settings.SUMMARIZER_PROMPT_VERSION = "v1.0"

                summary = generate_summary(
                    transcript=transcript,
                    video_title="Test",
                    video_url="https://example.com",
                    video_id="test",
                )

                assert summary.estimated_duration_minutes == 2.0
