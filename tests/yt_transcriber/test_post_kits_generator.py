"""Tests for yt_transcriber.post_kits_generator module."""

from unittest.mock import patch

import pytest

from yt_transcriber.models import LinkedInPost, PostKits, TwitterThread
from yt_transcriber.post_kits_generator import (
    PostKitsError,
    _build_linkedin_prompt,
    _build_twitter_prompt,
    _generate_linkedin_post,
    _generate_twitter_thread,
    _parse_linkedin_response,
    _parse_twitter_response,
    _translate_linkedin_post,
    _translate_twitter_thread,
    generate_post_kits,
)


class TestBuildLinkedInPrompt:
    """Tests for _build_linkedin_prompt function."""

    def test_includes_video_title(self, sample_video_summary):
        """Test that prompt includes video title."""
        prompt = _build_linkedin_prompt(sample_video_summary, "Test Video Title")
        assert "Test Video Title" in prompt

    def test_includes_executive_summary(self, sample_video_summary):
        """Test that prompt includes executive summary."""
        prompt = _build_linkedin_prompt(sample_video_summary, "Test")
        assert sample_video_summary.executive_summary in prompt

    def test_includes_key_points(self, sample_video_summary):
        """Test that prompt includes key points."""
        prompt = _build_linkedin_prompt(sample_video_summary, "Test")
        for point in sample_video_summary.key_points:
            assert point in prompt

    def test_has_output_format_section(self, sample_video_summary):
        """Test that prompt has output format section."""
        prompt = _build_linkedin_prompt(sample_video_summary, "Test")
        assert "OUTPUT FORMAT" in prompt
        assert "Hook:" in prompt
        assert "Intro:" in prompt
        assert "Insight" in prompt


class TestBuildTwitterPrompt:
    """Tests for _build_twitter_prompt function."""

    def test_includes_video_title(self, sample_video_summary):
        """Test that prompt includes video title."""
        prompt = _build_twitter_prompt(sample_video_summary, "Test Video")
        assert "Test Video" in prompt

    def test_includes_executive_summary(self, sample_video_summary):
        """Test that prompt includes executive summary."""
        prompt = _build_twitter_prompt(sample_video_summary, "Test")
        assert sample_video_summary.executive_summary in prompt

    def test_has_tweet_format(self, sample_video_summary):
        """Test that prompt specifies tweet format."""
        prompt = _build_twitter_prompt(sample_video_summary, "Test")
        assert "8-12 tweets" in prompt or "tweets" in prompt
        assert "Hashtags" in prompt


class TestParseLinkedInResponse:
    """Tests for _parse_linkedin_response function."""

    def test_parses_valid_response(self, sample_linkedin_response):
        """Test parsing valid LinkedIn response."""
        with patch("yt_transcriber.post_kits_generator.settings") as mock_settings:
            mock_settings.POST_KITS_LINKEDIN_MIN_INSIGHTS = 4
            mock_settings.POST_KITS_LINKEDIN_MAX_INSIGHTS = 8

            post = _parse_linkedin_response(sample_linkedin_response)

            assert isinstance(post, LinkedInPost)
            assert len(post.hook) > 0
            assert len(post.insights) >= 4

    def test_raises_on_missing_fields(self):
        """Test raises error on missing required fields."""
        incomplete = """
Hook: Test hook

Insight1: Some insight
"""
        with patch("yt_transcriber.post_kits_generator.settings") as mock_settings:
            mock_settings.POST_KITS_LINKEDIN_MIN_INSIGHTS = 4
            mock_settings.POST_KITS_LINKEDIN_MAX_INSIGHTS = 8

            with pytest.raises(PostKitsError, match="missing fields"):
                _parse_linkedin_response(incomplete)

    def test_raises_on_too_few_insights(self):
        """Test raises error with too few insights."""
        response = """
Hook: Test hook

Intro: Test intro

Insight1: Only one insight

WhyItMatters: Test value

CTA: Test call to action

Tags: #Test
"""
        with patch("yt_transcriber.post_kits_generator.settings") as mock_settings:
            mock_settings.POST_KITS_LINKEDIN_MIN_INSIGHTS = 4
            mock_settings.POST_KITS_LINKEDIN_MAX_INSIGHTS = 8

            with pytest.raises(PostKitsError, match="insights found"):
                _parse_linkedin_response(response)

    def test_extracts_tags(self, sample_linkedin_response):
        """Test that tags are extracted."""
        with patch("yt_transcriber.post_kits_generator.settings") as mock_settings:
            mock_settings.POST_KITS_LINKEDIN_MIN_INSIGHTS = 4
            mock_settings.POST_KITS_LINKEDIN_MAX_INSIGHTS = 8

            post = _parse_linkedin_response(sample_linkedin_response)
            assert post.tags is not None


class TestParseTwitterResponse:
    """Tests for _parse_twitter_response function."""

    def test_parses_valid_response(self, sample_twitter_response):
        """Test parsing valid Twitter response."""
        thread = _parse_twitter_response(sample_twitter_response)

        assert isinstance(thread, TwitterThread)
        assert len(thread.tweets) >= 8

    def test_parses_numbered_format(self):
        """Test parsing numbered tweet format."""
        response = """
1. First tweet here
2. Second tweet here
3. Third tweet here
4. Fourth tweet here
5. Fifth tweet here
6. Sixth tweet here
7. Seventh tweet here
8. Eighth tweet here

Hashtags: Python, Testing
"""
        thread = _parse_twitter_response(response)
        assert len(thread.tweets) == 8

    def test_parses_slash_format(self):
        """Test parsing slash tweet format (1/)."""
        response = """
1/ First tweet
2/ Second tweet
3/ Third tweet
4/ Fourth tweet
5/ Fifth tweet
6/ Sixth tweet
7/ Seventh tweet
8/ Eighth tweet

Hashtags: AI, Tech
"""
        thread = _parse_twitter_response(response)
        assert len(thread.tweets) == 8

    def test_extracts_hashtags(self, sample_twitter_response):
        """Test that hashtags are extracted."""
        thread = _parse_twitter_response(sample_twitter_response)
        assert len(thread.hashtags) > 0

    def test_raises_on_too_few_tweets(self):
        """Test raises error with too few tweets."""
        response = """
1. Only one tweet

Hashtags: Test
"""
        with pytest.raises(PostKitsError, match="tweets found"):
            _parse_twitter_response(response)


class TestGenerateLinkedInPost:
    """Tests for _generate_linkedin_post function."""

    def test_calls_llm_api(self, sample_video_summary, sample_linkedin_response):
        """Test that LLM API is called."""
        with patch("yt_transcriber.post_kits_generator.call_gemini_with_cache") as mock_call:
            mock_call.return_value = sample_linkedin_response

            with patch("yt_transcriber.post_kits_generator.settings") as mock_settings:
                mock_settings.SUMMARIZER_MODEL = "gemini-2.5-flash"
                mock_settings.POST_KITS_PROMPT_VERSION = "v1.0"
                mock_settings.POST_KITS_LINKEDIN_MIN_INSIGHTS = 4
                mock_settings.POST_KITS_LINKEDIN_MAX_INSIGHTS = 8

                post = _generate_linkedin_post(sample_video_summary, "Test Video")

                mock_call.assert_called_once()
                assert isinstance(post, LinkedInPost)

    def test_api_error_raises_postkits_error(self, sample_video_summary):
        """Test that API errors are wrapped."""
        with patch("yt_transcriber.post_kits_generator.call_gemini_with_cache") as mock_call:
            mock_call.side_effect = PostKitsError("API failed")

            with patch("yt_transcriber.post_kits_generator.settings") as mock_settings:
                mock_settings.SUMMARIZER_MODEL = "gemini-2.5-flash"
                mock_settings.POST_KITS_PROMPT_VERSION = "v1.0"

                with pytest.raises(PostKitsError):
                    _generate_linkedin_post(sample_video_summary, "Test")


class TestGenerateTwitterThread:
    """Tests for _generate_twitter_thread function."""

    def test_calls_llm_api(self, sample_video_summary, sample_twitter_response):
        """Test that LLM API is called."""
        with patch("yt_transcriber.post_kits_generator.call_gemini_with_cache") as mock_call:
            mock_call.return_value = sample_twitter_response

            with patch("yt_transcriber.post_kits_generator.settings") as mock_settings:
                mock_settings.SUMMARIZER_MODEL = "gemini-2.5-flash"
                mock_settings.POST_KITS_PROMPT_VERSION = "v1.0"

                thread = _generate_twitter_thread(sample_video_summary, "Test Video")

                mock_call.assert_called_once()
                assert isinstance(thread, TwitterThread)


class TestTranslateLinkedInPost:
    """Tests for _translate_linkedin_post function."""

    def test_translates_post(self, sample_linkedin_response):
        """Test that post is translated."""
        with patch("yt_transcriber.post_kits_generator.settings") as mock_settings:
            mock_settings.POST_KITS_LINKEDIN_MIN_INSIGHTS = 4
            mock_settings.POST_KITS_LINKEDIN_MAX_INSIGHTS = 8

            original = _parse_linkedin_response(sample_linkedin_response)

            with patch("yt_transcriber.post_kits_generator.call_gemini_with_cache") as mock_call:
                # Return Spanish-like response
                mock_call.return_value = sample_linkedin_response.replace(
                    "Python testing", "pruebas en Python"
                )

                mock_settings.SUMMARIZER_MODEL = "gemini-2.5-flash"
                mock_settings.POST_KITS_PROMPT_VERSION = "v1.0"

                translated = _translate_linkedin_post(original, "Test Video")

                assert isinstance(translated, LinkedInPost)

    def test_fallback_on_error(self, sample_linkedin_response):
        """Test that original is returned on translation error."""
        with patch("yt_transcriber.post_kits_generator.settings") as mock_settings:
            mock_settings.POST_KITS_LINKEDIN_MIN_INSIGHTS = 4
            mock_settings.POST_KITS_LINKEDIN_MAX_INSIGHTS = 8

            original = _parse_linkedin_response(sample_linkedin_response)

            with patch("yt_transcriber.post_kits_generator.call_gemini_with_cache") as mock_call:
                mock_call.side_effect = PostKitsError("Translation failed")

                mock_settings.SUMMARIZER_MODEL = "gemini-2.5-flash"
                mock_settings.POST_KITS_PROMPT_VERSION = "v1.0"

                with pytest.raises(PostKitsError):
                    _translate_linkedin_post(original, "Test Video")


class TestTranslateTwitterThread:
    """Tests for _translate_twitter_thread function."""

    def test_translates_thread(self, sample_twitter_response):
        """Test that thread is translated."""
        original = _parse_twitter_response(sample_twitter_response)

        with patch("yt_transcriber.post_kits_generator.call_gemini_with_cache") as mock_call:
            mock_call.return_value = sample_twitter_response

            with patch("yt_transcriber.post_kits_generator.settings") as mock_settings:
                mock_settings.SUMMARIZER_MODEL = "gemini-2.5-flash"
                mock_settings.POST_KITS_PROMPT_VERSION = "v1.0"

                translated = _translate_twitter_thread(original, "Test Video")

                assert isinstance(translated, TwitterThread)


class TestGeneratePostKits:
    """Tests for generate_post_kits main function."""

    def test_generates_complete_kits(
        self, sample_video_summary, sample_linkedin_response, sample_twitter_response
    ):
        """Test generating complete post kits."""
        with patch("yt_transcriber.post_kits_generator._generate_linkedin_post") as mock_linkedin:
            with patch("yt_transcriber.post_kits_generator._generate_twitter_thread") as mock_twitter:
                with patch("yt_transcriber.post_kits_generator._translate_linkedin_post") as mock_trans_li:
                    with patch("yt_transcriber.post_kits_generator._translate_twitter_thread") as mock_trans_tw:
                        with patch("yt_transcriber.post_kits_generator.settings") as mock_settings:
                            mock_settings.POST_KITS_LINKEDIN_MIN_INSIGHTS = 1
                            mock_settings.POST_KITS_LINKEDIN_MAX_INSIGHTS = 10
                            mock_settings.POST_KITS_LINKEDIN_MIN_CHARS = 0
                            mock_settings.POST_KITS_LINKEDIN_MAX_CHARS = 10000
                            mock_settings.POST_KITS_TWITTER_MIN_TWEETS = 1
                            mock_settings.POST_KITS_TWITTER_MAX_TWEETS = 20
                            mock_settings.POST_KITS_TWITTER_MAX_CHARS_PER_TWEET = 300
                            mock_settings.POST_KITS_TWITTER_MAX_HASHTAGS = 5

                            mock_linkedin.return_value = LinkedInPost(
                                hook="Test", intro="Test", insights=["A", "B", "C", "D"],
                                why_it_matters="Test", cta="Test?", tags="#Test"
                            )
                            mock_twitter.return_value = TwitterThread(
                                tweets=["T" + str(i) for i in range(10)],
                                hashtags=["Test"]
                            )
                            mock_trans_li.return_value = mock_linkedin.return_value
                            mock_trans_tw.return_value = mock_twitter.return_value

                            kits = generate_post_kits(
                                summary=sample_video_summary,
                                video_title="Test Video",
                                video_url="https://youtube.com/watch?v=test",
                            )

                            assert isinstance(kits, PostKits)
                            assert kits.video_title == "Test Video"

    def test_fallback_on_linkedin_translation_error(
        self, sample_video_summary, sample_linkedin_response, sample_twitter_response
    ):
        """Test fallback when LinkedIn translation fails."""
        with patch("yt_transcriber.post_kits_generator._generate_linkedin_post") as mock_linkedin:
            with patch("yt_transcriber.post_kits_generator._generate_twitter_thread") as mock_twitter:
                with patch("yt_transcriber.post_kits_generator._translate_linkedin_post") as mock_trans_li:
                    with patch("yt_transcriber.post_kits_generator._translate_twitter_thread") as mock_trans_tw:
                        with patch("yt_transcriber.post_kits_generator.settings") as mock_settings:
                            mock_settings.POST_KITS_LINKEDIN_MIN_INSIGHTS = 1
                            mock_settings.POST_KITS_LINKEDIN_MAX_INSIGHTS = 10
                            mock_settings.POST_KITS_LINKEDIN_MIN_CHARS = 0
                            mock_settings.POST_KITS_LINKEDIN_MAX_CHARS = 10000
                            mock_settings.POST_KITS_TWITTER_MIN_TWEETS = 1
                            mock_settings.POST_KITS_TWITTER_MAX_TWEETS = 20
                            mock_settings.POST_KITS_TWITTER_MAX_CHARS_PER_TWEET = 300
                            mock_settings.POST_KITS_TWITTER_MAX_HASHTAGS = 5

                            original_linkedin = LinkedInPost(
                                hook="Test EN", intro="Test", insights=["A", "B", "C", "D"],
                                why_it_matters="Test", cta="Test?", tags="#Test"
                            )
                            mock_linkedin.return_value = original_linkedin
                            mock_twitter.return_value = TwitterThread(
                                tweets=["T" + str(i) for i in range(10)],
                                hashtags=["Test"]
                            )
                            # Translation fails
                            mock_trans_li.side_effect = PostKitsError("Translation failed")
                            mock_trans_tw.return_value = mock_twitter.return_value

                            kits = generate_post_kits(
                                summary=sample_video_summary,
                                video_title="Test Video",
                                video_url="https://youtube.com/watch?v=test",
                            )

                            # Should use English version as fallback
                            assert kits.linkedin.hook == "Test EN"

    def test_validation_warnings_logged(
        self, sample_video_summary, sample_linkedin_response, sample_twitter_response
    ):
        """Test that validation warnings are logged but don't fail."""
        with patch("yt_transcriber.post_kits_generator._generate_linkedin_post") as mock_linkedin:
            with patch("yt_transcriber.post_kits_generator._generate_twitter_thread") as mock_twitter:
                with patch("yt_transcriber.post_kits_generator._translate_linkedin_post") as mock_trans_li:
                    with patch("yt_transcriber.post_kits_generator._translate_twitter_thread") as mock_trans_tw:
                        with patch("yt_transcriber.post_kits_generator.settings") as mock_settings:
                            # Set strict validation that will fail
                            mock_settings.POST_KITS_LINKEDIN_MIN_INSIGHTS = 10  # Too strict
                            mock_settings.POST_KITS_LINKEDIN_MAX_INSIGHTS = 15
                            mock_settings.POST_KITS_LINKEDIN_MIN_CHARS = 0
                            mock_settings.POST_KITS_LINKEDIN_MAX_CHARS = 10000
                            mock_settings.POST_KITS_TWITTER_MIN_TWEETS = 1
                            mock_settings.POST_KITS_TWITTER_MAX_TWEETS = 20
                            mock_settings.POST_KITS_TWITTER_MAX_CHARS_PER_TWEET = 300
                            mock_settings.POST_KITS_TWITTER_MAX_HASHTAGS = 5

                            mock_linkedin.return_value = LinkedInPost(
                                hook="Test", intro="Test", insights=["A", "B"],  # Too few
                                why_it_matters="Test", cta="Test?", tags="#Test"
                            )
                            mock_twitter.return_value = TwitterThread(
                                tweets=["T" + str(i) for i in range(10)],
                                hashtags=["Test"]
                            )
                            mock_trans_li.return_value = mock_linkedin.return_value
                            mock_trans_tw.return_value = mock_twitter.return_value

                            # Should not raise, just log warning
                            kits = generate_post_kits(
                                summary=sample_video_summary,
                                video_title="Test Video",
                                video_url="https://youtube.com/watch?v=test",
                            )

                            assert kits is not None
