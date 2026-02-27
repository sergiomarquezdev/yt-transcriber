"""Tests for core.translator module."""

from datetime import datetime
from unittest.mock import patch

import pytest

from core.models import TimestampedSection, VideoSummary
from core.translator import ScriptTranslator


class TestScriptTranslator:
    """Tests for ScriptTranslator class."""

    @pytest.fixture
    def translator(self):
        """Create a ScriptTranslator instance for testing."""
        with patch("core.translator.settings") as mock_settings:
            mock_settings.TRANSLATOR_MODEL = "haiku"
            mock_settings.SUMMARIZER_MODEL = "sonnet"
            return ScriptTranslator(use_translation_model=True)

    @pytest.fixture
    def sample_summary(self):
        """Create a sample VideoSummary for testing."""
        return VideoSummary(
            video_url="https://youtube.com/watch?v=test123",
            video_title="Test Video Title",
            video_id="test123",
            executive_summary="This video explains Python testing with pytest.",
            key_points=[
                "Unit tests help catch bugs early",
                "Fixtures provide reusable test setup",
                "Mocking isolates code under test",
            ],
            timestamps=[
                TimestampedSection("00:00", "Introduction to testing", 3),
                TimestampedSection("05:30", "Writing your first test", 5),
                TimestampedSection("10:00", "Advanced fixtures", 4),
            ],
            conclusion="Testing is essential for quality software development.",
            action_items=[
                "Install pytest in your project",
                "Write tests for existing code",
            ],
            word_count=500,
            estimated_duration_minutes=10.5,
            language="en",
            generated_at=datetime(2024, 1, 15, 12, 0, 0),
        )

    # =========================================================================
    # INITIALIZATION TESTS
    # =========================================================================

    def test_init_with_translation_model(self):
        """Test initialization with translation model."""
        with patch("core.translator.settings") as mock_settings:
            mock_settings.TRANSLATOR_MODEL = "haiku"
            mock_settings.SUMMARIZER_MODEL = "sonnet"

            translator = ScriptTranslator(use_translation_model=True)
            assert translator.model_name == "haiku"

    def test_init_with_summarizer_model(self):
        """Test initialization with summarizer model (for scripts)."""
        with patch("core.translator.settings") as mock_settings:
            mock_settings.TRANSLATOR_MODEL = "haiku"
            mock_settings.SUMMARIZER_MODEL = "sonnet"

            translator = ScriptTranslator(use_translation_model=False)
            assert translator.model_name == "sonnet"

    # =========================================================================
    # TRANSLATE SUMMARY TESTS
    # =========================================================================

    def test_translate_summary_structure_preserved(self, translator, sample_summary):
        """Test that summary structure is preserved after translation."""
        with patch("core.translator.call_llm") as mock_call:
            mock_call.return_value = "Texto traducido al español"

            result = translator.translate_summary(sample_summary)

            # Structure should be preserved
            assert result.video_url == sample_summary.video_url
            assert result.video_title == sample_summary.video_title
            assert result.video_id == sample_summary.video_id
            assert result.word_count == sample_summary.word_count
            assert len(result.key_points) == len(sample_summary.key_points)
            assert len(result.timestamps) == len(sample_summary.timestamps)
            assert len(result.action_items) == len(sample_summary.action_items)

    def test_translate_summary_language_set_to_spanish(self, translator, sample_summary):
        """Test that translated summary has language set to 'es'."""
        with patch("core.translator.call_llm") as mock_call:
            mock_call.return_value = "Texto traducido"

            result = translator.translate_summary(sample_summary)

            assert result.language == "es"

    def test_translate_summary_timestamps_preserved(self, translator, sample_summary):
        """Test that timestamp values are preserved (only description translated)."""
        with patch("core.translator.call_llm") as mock_call:
            mock_call.return_value = "Descripción traducida"

            result = translator.translate_summary(sample_summary)

            # Timestamps should be same
            assert result.timestamps[0].timestamp == "00:00"
            assert result.timestamps[1].timestamp == "05:30"
            assert result.timestamps[2].timestamp == "10:00"

            # Importance should be preserved
            assert result.timestamps[0].importance == 3
            assert result.timestamps[1].importance == 5

    def test_translate_summary_calls_llm_for_each_field(self, translator, sample_summary):
        """Test that LLM is called for each translatable field."""
        with patch("core.translator.call_llm") as mock_call:
            mock_call.return_value = "Traducido"

            translator.translate_summary(sample_summary)

            # Should be called for:
            # - executive_summary (1)
            # - key_points (3)
            # - timestamp descriptions (3)
            # - conclusion (1)
            # - action_items (2)
            # Total: 10 calls
            assert mock_call.call_count >= 10

    def test_translate_summary_graceful_degradation(self, translator, sample_summary):
        """Test that translate_summary uses original text on API errors (graceful degradation)."""
        with patch("core.translator.call_llm") as mock_call:
            mock_call.side_effect = Exception("API Error")

            # Should NOT raise - uses graceful degradation
            result = translator.translate_summary(sample_summary)

            # Original text should be preserved when translation fails
            assert result.executive_summary == sample_summary.executive_summary
            assert result.language == "es"  # Language still set to Spanish

    def test_translate_summary_new_timestamp(self, translator, sample_summary):
        """Test that translation creates new generated_at timestamp."""
        with patch("core.translator.call_llm") as mock_call:
            mock_call.return_value = "Traducido"

            result = translator.translate_summary(sample_summary)

            assert result.generated_at != sample_summary.generated_at
            assert result.generated_at > sample_summary.generated_at

    # =========================================================================
    # TRANSLATE TEXT BLOCK TESTS
    # =========================================================================

    def test_translate_text_block_empty_fallback(self, translator):
        """Test that empty translation returns original text."""
        with patch("core.translator.call_llm") as mock_call:
            mock_call.return_value = ""  # Empty response

            result = translator._translate_text_block(
                text="Original text",
                block_type="test",
                video_title="Test Video",
            )

            assert result == "Original text"

    def test_translate_text_block_error_fallback(self, translator):
        """Test that translation error returns original text."""
        with patch("core.translator.call_llm") as mock_call:
            mock_call.side_effect = Exception("Translation failed")

            result = translator._translate_text_block(
                text="Original text",
                block_type="test",
                video_title="Test Video",
            )

            # Should return original on error
            assert result == "Original text"

    def test_translate_text_block_strips_whitespace(self, translator):
        """Test that translated text is stripped."""
        with patch("core.translator.call_llm") as mock_call:
            mock_call.return_value = "  Texto traducido  \n"

            result = translator._translate_text_block(
                text="Original",
                block_type="test",
                video_title="Test",
            )

            assert result == "Texto traducido"

    # =========================================================================
    # SEO TRANSLATION TESTS
    # =========================================================================

    def test_translate_seo_title_removes_quotes(self, translator):
        """Test that quotes are removed from translated titles."""
        with patch("core.translator.call_llm") as mock_call:
            mock_call.return_value = '"Título traducido"'

            result = translator._translate_seo_title("Original Title")

            assert result == "Título traducido"

    def test_translate_seo_title_fallback_on_error(self, translator):
        """Test that original title is returned on error."""
        with patch("core.translator.call_llm") as mock_call:
            mock_call.side_effect = Exception("Error")

            result = translator._translate_seo_title("Original Title")

            assert result == "Original Title"

    def test_translate_seo_description_fallback(self, translator):
        """Test SEO description fallback on error."""
        with patch("core.translator.call_llm") as mock_call:
            mock_call.side_effect = Exception("Error")

            result = translator._translate_seo_description("Original description")

            assert result == "Original description"

    # =========================================================================
    # TAG ADAPTATION TESTS
    # =========================================================================

    def test_adapt_seo_tags_preserves_original(self, translator):
        """Test that original tags are preserved."""
        original_tags = ["python", "testing", "pytest"]

        result = translator._adapt_seo_tags(original_tags)

        for tag in original_tags:
            assert tag in result

    def test_adapt_seo_tags_adds_spanish_variants(self, translator):
        """Test that Spanish variants are added."""
        original_tags = ["tutorial", "guide", "installation"]

        result = translator._adapt_seo_tags(original_tags)

        # Should contain Spanish variants
        assert "guía" in result or "configuración" in result or "instalación" in result

    def test_adapt_seo_tags_limits_to_30(self, translator):
        """Test that tags are limited to 30."""
        many_tags = [f"tag{i}" for i in range(50)]

        result = translator._adapt_seo_tags(many_tags)

        assert len(result) <= 30

    def test_adapt_seo_tags_no_duplicates(self, translator):
        """Test that no duplicate tags are added."""
        original_tags = ["tutorial", "guía"]  # Spanish already present

        result = translator._adapt_seo_tags(original_tags)

        # Count occurrences of "guía"
        count = sum(1 for tag in result if tag == "guía")
        assert count == 1

    # =========================================================================
    # PROMPT CONFIGURATION TESTS
    # =========================================================================

    def test_translation_passes_model(self, translator, sample_summary):
        """Test that translations pass the correct model."""
        with patch("core.translator.call_llm") as mock_call:
            mock_call.return_value = "Traducido"

            translator.translate_summary(sample_summary)

            # All calls should pass model kwarg
            for call in mock_call.call_args_list:
                kwargs = call[1]
                assert "model" in kwargs

    def test_translation_passes_prompt(self, translator):
        """Test that translation passes prompt with text to translate."""
        with patch("core.translator.call_llm") as mock_call:
            mock_call.return_value = "Traducido"

            translator._translate_text_block(
                text="Some text to translate",
                block_type="key point",
                video_title="Test Video",
            )

            call_kwargs = mock_call.call_args[1]
            assert "prompt" in call_kwargs
            assert "Some text to translate" in call_kwargs["prompt"]
