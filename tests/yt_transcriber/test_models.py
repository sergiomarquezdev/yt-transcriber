"""Tests for yt_transcriber.models module."""

from datetime import datetime
from unittest.mock import patch

import pytest

from yt_transcriber.models import LinkedInPost, PostKits, TwitterThread


class TestLinkedInPost:
    """Tests for LinkedInPost dataclass."""

    @pytest.fixture
    def valid_post(self):
        """Create a valid LinkedInPost for testing."""
        return LinkedInPost(
            hook="AI is transforming how we work",
            intro="After exploring the latest tools, here's what stands out.",
            insights=[
                "🔹 Speed: Tasks that took hours now take minutes",
                "🔹 Quality: AI catches errors humans miss",
                "🔹 Scale: Handle 10x more without extra staff",
                "🔹 Cost: ROI visible within weeks",
            ],
            why_it_matters="This isn't just efficiency - it's competitive advantage.",
            cta="Which AI tool has made the biggest impact for you?",
            tags="#AI #Productivity #FutureOfWork",
        )

    # =========================================================================
    # CREATION TESTS
    # =========================================================================

    def test_basic_creation(self, valid_post):
        """Test basic LinkedInPost creation."""
        assert valid_post.hook == "AI is transforming how we work"
        assert len(valid_post.insights) == 4
        assert valid_post.tags is not None

    def test_creation_without_tags(self):
        """Test creation without tags."""
        post = LinkedInPost(
            hook="Test hook",
            intro="Test intro",
            insights=["Insight 1", "Insight 2", "Insight 3", "Insight 4"],
            why_it_matters="Test value",
            cta="Test CTA",
            tags=None,
        )
        assert post.tags is None

    # =========================================================================
    # TO_MARKDOWN TESTS
    # =========================================================================

    def test_to_markdown_includes_all_parts(self, valid_post):
        """Test that to_markdown includes all parts."""
        md = valid_post.to_markdown()

        assert "AI is transforming" in md
        assert "After exploring" in md
        assert "🔹 Speed" in md
        assert "competitive advantage" in md
        assert "Which AI tool" in md

    def test_to_markdown_with_tags(self, valid_post):
        """Test that tags are included in markdown."""
        md = valid_post.to_markdown()
        assert "#AI" in md

    def test_to_markdown_without_tags(self):
        """Test markdown without tags."""
        post = LinkedInPost(
            hook="Hook",
            intro="Intro",
            insights=["I1", "I2", "I3", "I4"],
            why_it_matters="Value",
            cta="CTA",
            tags=None,
        )
        md = post.to_markdown()
        # Should still work, just no tags section
        assert "Hook" in md

    def test_to_markdown_bullet_format(self, valid_post):
        """Test that insights are formatted as bullets."""
        md = valid_post.to_markdown()
        # Should have bullet points
        assert "•" in md

    # =========================================================================
    # GET_CHAR_COUNT TESTS
    # =========================================================================

    def test_get_char_count_returns_int(self, valid_post):
        """Test that get_char_count returns integer."""
        count = valid_post.get_char_count()
        assert isinstance(count, int)
        assert count > 0

    def test_get_char_count_matches_markdown_length(self, valid_post):
        """Test that char count matches markdown length."""
        md = valid_post.to_markdown()
        count = valid_post.get_char_count()
        assert count == len(md)

    # =========================================================================
    # VALIDATION TESTS
    # =========================================================================

    def test_validate_valid_post(self, valid_post):
        """Test validation of valid post."""
        with patch("yt_transcriber.models.settings") as mock_settings:
            mock_settings.POST_KITS_LINKEDIN_MIN_INSIGHTS = 4
            mock_settings.POST_KITS_LINKEDIN_MAX_INSIGHTS = 8
            mock_settings.POST_KITS_LINKEDIN_MIN_CHARS = 200
            mock_settings.POST_KITS_LINKEDIN_MAX_CHARS = 2000

            is_valid, error = valid_post.validate()
            assert is_valid is True
            assert error == ""

    def test_validate_too_few_insights(self):
        """Test validation fails with too few insights."""
        post = LinkedInPost(
            hook="Hook",
            intro="Intro",
            insights=["Only one"],
            why_it_matters="Value",
            cta="CTA",
        )

        with patch("yt_transcriber.models.settings") as mock_settings:
            mock_settings.POST_KITS_LINKEDIN_MIN_INSIGHTS = 4
            mock_settings.POST_KITS_LINKEDIN_MAX_INSIGHTS = 8
            mock_settings.POST_KITS_LINKEDIN_MIN_CHARS = 0
            mock_settings.POST_KITS_LINKEDIN_MAX_CHARS = 10000

            is_valid, error = post.validate()
            assert is_valid is False
            assert "too few insights" in error

    def test_validate_too_many_insights(self):
        """Test validation fails with too many insights."""
        post = LinkedInPost(
            hook="Hook",
            intro="Intro",
            insights=[f"Insight {i}" for i in range(15)],
            why_it_matters="Value",
            cta="CTA",
        )

        with patch("yt_transcriber.models.settings") as mock_settings:
            mock_settings.POST_KITS_LINKEDIN_MIN_INSIGHTS = 4
            mock_settings.POST_KITS_LINKEDIN_MAX_INSIGHTS = 8
            mock_settings.POST_KITS_LINKEDIN_MIN_CHARS = 0
            mock_settings.POST_KITS_LINKEDIN_MAX_CHARS = 10000

            is_valid, error = post.validate()
            assert is_valid is False
            assert "too many insights" in error

    def test_validate_too_short(self):
        """Test validation fails when post is too short."""
        post = LinkedInPost(
            hook="Hi",
            intro="X",
            insights=["A", "B", "C", "D"],
            why_it_matters="Y",
            cta="Z",
        )

        with patch("yt_transcriber.models.settings") as mock_settings:
            mock_settings.POST_KITS_LINKEDIN_MIN_INSIGHTS = 1
            mock_settings.POST_KITS_LINKEDIN_MAX_INSIGHTS = 10
            mock_settings.POST_KITS_LINKEDIN_MIN_CHARS = 500
            mock_settings.POST_KITS_LINKEDIN_MAX_CHARS = 2000

            is_valid, error = post.validate()
            assert is_valid is False
            assert "too short" in error


class TestTwitterThread:
    """Tests for TwitterThread dataclass."""

    @pytest.fixture
    def valid_thread(self):
        """Create a valid TwitterThread for testing."""
        return TwitterThread(
            tweets=[
                "Just discovered something amazing about Python testing 🚀",
                "First, pytest makes testing actually enjoyable",
                "The fixture system is incredibly powerful",
                "Parametrize lets you test multiple cases easily",
                "Mocking with pytest-mock is straightforward",
                "Coverage reports help find untested code",
                "Watch mode saves so much time during development",
                "Integration with CI/CD is seamless",
                "This has completely changed how I write code",
                "What's your favorite testing tip? 👇",
            ],
            hashtags=["Python", "Testing", "DevTips"],
        )

    # =========================================================================
    # CREATION TESTS
    # =========================================================================

    def test_basic_creation(self, valid_thread):
        """Test basic TwitterThread creation."""
        assert len(valid_thread.tweets) == 10
        assert len(valid_thread.hashtags) == 3

    def test_creation_without_hashtags(self):
        """Test creation without hashtags."""
        thread = TwitterThread(
            tweets=["Tweet " + str(i) for i in range(8)],
        )
        assert thread.hashtags == []

    # =========================================================================
    # TO_MARKDOWN TESTS
    # =========================================================================

    def test_to_markdown_numbered(self, valid_thread):
        """Test that tweets are numbered."""
        md = valid_thread.to_markdown()

        assert "1." in md
        assert "2." in md
        assert "10." in md

    def test_to_markdown_hashtags_in_last_tweet(self, valid_thread):
        """Test that hashtags appear in last tweet."""
        md = valid_thread.to_markdown()
        lines = md.split("\n\n")
        last_line = lines[-1]

        assert "#Python" in last_line
        assert "#Testing" in last_line

    def test_to_markdown_limits_hashtags_to_3(self):
        """Test that only 3 hashtags are included."""
        thread = TwitterThread(
            tweets=["Tweet " + str(i) for i in range(8)],
            hashtags=["One", "Two", "Three", "Four", "Five"],
        )

        md = thread.to_markdown()

        assert "#One" in md
        assert "#Two" in md
        assert "#Three" in md
        assert "#Four" not in md

    # =========================================================================
    # VALIDATION TESTS
    # =========================================================================

    def test_validate_valid_thread(self, valid_thread):
        """Test validation of valid thread."""
        with patch("yt_transcriber.models.settings") as mock_settings:
            mock_settings.POST_KITS_TWITTER_MIN_TWEETS = 8
            mock_settings.POST_KITS_TWITTER_MAX_TWEETS = 12
            mock_settings.POST_KITS_TWITTER_MAX_CHARS_PER_TWEET = 280
            mock_settings.POST_KITS_TWITTER_MAX_HASHTAGS = 3

            is_valid, error = valid_thread.validate()
            assert is_valid is True
            assert error == ""

    def test_validate_too_few_tweets(self):
        """Test validation fails with too few tweets."""
        thread = TwitterThread(
            tweets=["Tweet 1", "Tweet 2"],
            hashtags=[],
        )

        with patch("yt_transcriber.models.settings") as mock_settings:
            mock_settings.POST_KITS_TWITTER_MIN_TWEETS = 8
            mock_settings.POST_KITS_TWITTER_MAX_TWEETS = 12
            mock_settings.POST_KITS_TWITTER_MAX_CHARS_PER_TWEET = 280
            mock_settings.POST_KITS_TWITTER_MAX_HASHTAGS = 3

            is_valid, error = thread.validate()
            assert is_valid is False
            assert "too short" in error

    def test_validate_too_many_tweets(self):
        """Test validation fails with too many tweets."""
        thread = TwitterThread(
            tweets=["Tweet " + str(i) for i in range(20)],
            hashtags=[],
        )

        with patch("yt_transcriber.models.settings") as mock_settings:
            mock_settings.POST_KITS_TWITTER_MIN_TWEETS = 8
            mock_settings.POST_KITS_TWITTER_MAX_TWEETS = 12
            mock_settings.POST_KITS_TWITTER_MAX_CHARS_PER_TWEET = 280
            mock_settings.POST_KITS_TWITTER_MAX_HASHTAGS = 3

            is_valid, error = thread.validate()
            assert is_valid is False
            assert "too long" in error

    def test_validate_tweet_too_long(self):
        """Test validation fails when a tweet is too long."""
        thread = TwitterThread(
            tweets=[
                "Normal tweet",
                "A" * 300,  # Too long
                "Normal tweet",
                "Normal tweet",
                "Normal tweet",
                "Normal tweet",
                "Normal tweet",
                "Normal tweet",
            ],
            hashtags=[],
        )

        with patch("yt_transcriber.models.settings") as mock_settings:
            mock_settings.POST_KITS_TWITTER_MIN_TWEETS = 8
            mock_settings.POST_KITS_TWITTER_MAX_TWEETS = 12
            mock_settings.POST_KITS_TWITTER_MAX_CHARS_PER_TWEET = 280
            mock_settings.POST_KITS_TWITTER_MAX_HASHTAGS = 3

            is_valid, error = thread.validate()
            assert is_valid is False
            assert "too long" in error

    def test_validate_too_many_hashtags(self):
        """Test validation fails with too many hashtags."""
        thread = TwitterThread(
            tweets=["Tweet " + str(i) for i in range(8)],
            hashtags=["One", "Two", "Three", "Four", "Five"],
        )

        with patch("yt_transcriber.models.settings") as mock_settings:
            mock_settings.POST_KITS_TWITTER_MIN_TWEETS = 8
            mock_settings.POST_KITS_TWITTER_MAX_TWEETS = 12
            mock_settings.POST_KITS_TWITTER_MAX_CHARS_PER_TWEET = 280
            mock_settings.POST_KITS_TWITTER_MAX_HASHTAGS = 3

            is_valid, error = thread.validate()
            assert is_valid is False
            assert "Too many hashtags" in error


class TestPostKits:
    """Tests for PostKits dataclass."""

    @pytest.fixture
    def valid_linkedin(self):
        """Create valid LinkedIn post."""
        return LinkedInPost(
            hook="Test hook",
            intro="Test intro",
            insights=["I1", "I2", "I3", "I4"],
            why_it_matters="Value",
            cta="CTA?",
            tags="#Test",
        )

    @pytest.fixture
    def valid_twitter(self):
        """Create valid Twitter thread."""
        return TwitterThread(
            tweets=["Tweet " + str(i) for i in range(10)],
            hashtags=["Test"],
        )

    @pytest.fixture
    def valid_post_kits(self, valid_linkedin, valid_twitter):
        """Create valid PostKits."""
        return PostKits(
            linkedin=valid_linkedin,
            twitter=valid_twitter,
            video_title="Test Video",
            video_url="https://youtube.com/watch?v=test123",
        )

    # =========================================================================
    # CREATION TESTS
    # =========================================================================

    def test_basic_creation(self, valid_post_kits):
        """Test basic PostKits creation."""
        assert valid_post_kits.video_title == "Test Video"
        assert valid_post_kits.video_url == "https://youtube.com/watch?v=test123"
        assert isinstance(valid_post_kits.generated_at, datetime)

    def test_generated_at_auto_set(self, valid_linkedin, valid_twitter):
        """Test that generated_at is automatically set."""
        before = datetime.now()
        kits = PostKits(
            linkedin=valid_linkedin,
            twitter=valid_twitter,
            video_title="Test",
            video_url="https://example.com",
        )
        after = datetime.now()

        assert before <= kits.generated_at <= after

    # =========================================================================
    # TO_MARKDOWN TESTS
    # =========================================================================

    def test_to_markdown_includes_header(self, valid_post_kits):
        """Test that markdown includes header."""
        md = valid_post_kits.to_markdown()

        assert "# 📱 Post Kits:" in md
        assert "Test Video" in md

    def test_to_markdown_includes_video_url(self, valid_post_kits):
        """Test that markdown includes video URL."""
        md = valid_post_kits.to_markdown()
        assert "https://youtube.com/watch?v=test123" in md

    def test_to_markdown_includes_both_sections(self, valid_post_kits):
        """Test that markdown includes both LinkedIn and Twitter sections."""
        md = valid_post_kits.to_markdown()

        assert "## 💼 LinkedIn Post" in md
        assert "## 🐦 X/Twitter Thread" in md

    def test_to_markdown_includes_stats(self, valid_post_kits):
        """Test that markdown includes statistics."""
        md = valid_post_kits.to_markdown()

        assert "Character count" in md
        assert "Thread length" in md

    # =========================================================================
    # VALIDATION TESTS
    # =========================================================================

    def test_validate_both_valid(self, valid_post_kits):
        """Test validation when both are valid."""
        with patch("yt_transcriber.models.settings") as mock_settings:
            mock_settings.POST_KITS_LINKEDIN_MIN_INSIGHTS = 1
            mock_settings.POST_KITS_LINKEDIN_MAX_INSIGHTS = 10
            mock_settings.POST_KITS_LINKEDIN_MIN_CHARS = 0
            mock_settings.POST_KITS_LINKEDIN_MAX_CHARS = 10000
            mock_settings.POST_KITS_TWITTER_MIN_TWEETS = 8
            mock_settings.POST_KITS_TWITTER_MAX_TWEETS = 12
            mock_settings.POST_KITS_TWITTER_MAX_CHARS_PER_TWEET = 280
            mock_settings.POST_KITS_TWITTER_MAX_HASHTAGS = 3

            is_valid, errors = valid_post_kits.validate()
            assert is_valid is True
            assert errors == []

    def test_validate_returns_all_errors(self, valid_twitter):
        """Test that validation returns all errors."""
        # Create invalid LinkedIn
        bad_linkedin = LinkedInPost(
            hook="H",
            intro="I",
            insights=["X"],  # Too few
            why_it_matters="W",
            cta="C",
        )

        kits = PostKits(
            linkedin=bad_linkedin,
            twitter=valid_twitter,
            video_title="Test",
            video_url="https://example.com",
        )

        with patch("yt_transcriber.models.settings") as mock_settings:
            mock_settings.POST_KITS_LINKEDIN_MIN_INSIGHTS = 4
            mock_settings.POST_KITS_LINKEDIN_MAX_INSIGHTS = 8
            mock_settings.POST_KITS_LINKEDIN_MIN_CHARS = 0
            mock_settings.POST_KITS_LINKEDIN_MAX_CHARS = 10000
            mock_settings.POST_KITS_TWITTER_MIN_TWEETS = 8
            mock_settings.POST_KITS_TWITTER_MAX_TWEETS = 12
            mock_settings.POST_KITS_TWITTER_MAX_CHARS_PER_TWEET = 280
            mock_settings.POST_KITS_TWITTER_MAX_HASHTAGS = 3

            is_valid, errors = kits.validate()
            assert is_valid is False
            assert len(errors) >= 1
            assert any("LinkedIn" in err for err in errors)
