"""Data models for post kits generation.

This module provides data models for LinkedIn and Twitter post generation.
Validation rules are parameterized via AppSettings to keep behavior
configurable and aligned with documentation/tests.
"""

from dataclasses import dataclass, field
from datetime import datetime

from core.settings import settings


@dataclass
class LinkedInPost:
    """LinkedIn post structure (800-1200 chars)."""

    hook: str  # Catchy opening line
    intro: str  # One-liner context
    insights: list[str]  # 4-8 bullet points
    why_it_matters: str  # Value proposition
    cta: str  # Call to action
    tags: str | None = None  # Optional hashtags

    def to_markdown(self) -> str:
        """Convert to markdown format.

        Returns:
            Formatted markdown string
        """
        bullets = "\n".join(f"• {insight}" for insight in self.insights)

        parts = [
            f"**{self.hook}**",
            "",
            self.intro,
            "",
            bullets,
            "",
            f"**Why it matters**: {self.why_it_matters}",
            "",
            f"**{self.cta}**",
        ]

        if self.tags:
            parts.extend(["", f"_{self.tags}_"])

        return "\n".join(parts)

    def get_char_count(self) -> int:
        """Get total character count.

        Returns:
            Total character count of markdown representation
        """
        return len(self.to_markdown())

    def validate(self) -> tuple[bool, str]:
        """Validate char limits and structure.

        Returns:
            Tuple of (is_valid, error_message)
        """
        min_ins = settings.POST_KITS_LINKEDIN_MIN_INSIGHTS
        max_ins = settings.POST_KITS_LINKEDIN_MAX_INSIGHTS
        if len(self.insights) < min_ins:
            return False, f"too few insights: {len(self.insights)} (min {min_ins})"
        if len(self.insights) > max_ins:
            return False, f"too many insights: {len(self.insights)} (max {max_ins})"

        # Char count validation
        char_count = self.get_char_count()
        min_chars = settings.POST_KITS_LINKEDIN_MIN_CHARS
        max_chars = settings.POST_KITS_LINKEDIN_MAX_CHARS
        if char_count < min_chars:
            return False, f"LinkedIn post too short: {char_count} chars (min {min_chars})"
        if char_count > max_chars:
            return False, f"LinkedIn post too long: {char_count} chars (max {max_chars})"

        return True, ""


@dataclass
class TwitterThread:
    """Twitter thread structure (8-12 tweets)."""

    tweets: list[str]  # 8-12 tweets, 280 chars each
    hashtags: list[str] = field(default_factory=list)  # Max 3, only in last tweet

    def to_markdown(self) -> str:
        """Convert to markdown format.

        Returns:
            Formatted markdown string with numbered tweets
        """
        lines = []
        for i, tweet in enumerate(self.tweets, 1):
            if i == len(self.tweets) and self.hashtags:
                # Add hashtags to last tweet
                hashtags_str = " ".join(f"#{tag}" for tag in self.hashtags[:3])
                lines.append(f"{i}. {tweet} {hashtags_str}")
            else:
                lines.append(f"{i}. {tweet}")

        return "\n\n".join(lines)

    def validate(self) -> tuple[bool, str]:
        """Validate tweet count and char limits.

        Returns:
            Tuple of (is_valid, error_message)
        """
        min_tweets = settings.POST_KITS_TWITTER_MIN_TWEETS
        max_tweets = settings.POST_KITS_TWITTER_MAX_TWEETS
        if len(self.tweets) < min_tweets:
            return False, f"Thread too short: {len(self.tweets)} tweets (min {min_tweets})"
        if len(self.tweets) > max_tweets:
            return False, f"Thread too long: {len(self.tweets)} tweets (max {max_tweets})"

        # Check each tweet length (flexible - allow up to 300 chars since you'll edit)
        max_chars = settings.POST_KITS_TWITTER_MAX_CHARS_PER_TWEET
        for i, tweet in enumerate(self.tweets, 1):
            tweet_len = len(tweet)
            if i == len(self.tweets) and self.hashtags:
                # Account for hashtags in last tweet
                hashtags_len = sum(
                    len(tag) + 2 for tag in self.hashtags[: settings.POST_KITS_TWITTER_MAX_HASHTAGS]
                )  # +2 for "# "
                total_len = tweet_len + hashtags_len + 1  # +1 for space
                if total_len > max_chars:
                    return (
                        False,
                        f"Tweet {i} too long with hashtags: {total_len} chars (max {max_chars})",
                    )
            else:
                if tweet_len > max_chars:
                    return False, f"Tweet {i} too long: {tweet_len} chars (max {max_chars})"

        if len(self.hashtags) > settings.POST_KITS_TWITTER_MAX_HASHTAGS:
            return (
                False,
                f"Too many hashtags: {len(self.hashtags)} (max {settings.POST_KITS_TWITTER_MAX_HASHTAGS})",
            )

        return True, ""


@dataclass
class PostKits:
    """Complete post kits for LinkedIn + Twitter."""

    linkedin: LinkedInPost
    twitter: TwitterThread
    video_title: str
    video_url: str
    generated_at: datetime = field(default_factory=datetime.now)

    def to_markdown(self) -> str:
        """Generate complete markdown file.

        Returns:
            Complete markdown document with both LinkedIn and Twitter posts
        """
        return f"""# 📱 Post Kits: {self.video_title}

**🔗 Video**: {self.video_url}
**📅 Generated**: {self.generated_at.strftime("%Y-%m-%d %H:%M")}

---

## 💼 LinkedIn Post

{self.linkedin.to_markdown()}

**Character count**: {self.linkedin.get_char_count()} / 1200

---

## 🐦 X/Twitter Thread

{self.twitter.to_markdown()}

**Thread length**: {len(self.twitter.tweets)} tweets

---

**📊 Stats**: LinkedIn ({self.linkedin.get_char_count()} chars) | Twitter ({len(self.twitter.tweets)} tweets)
"""

    def validate(self) -> tuple[bool, list[str]]:
        """Validate both LinkedIn and Twitter posts.

        Returns:
            Tuple of (all_valid, list_of_error_messages)
        """
        errors = []

        linkedin_valid, linkedin_error = self.linkedin.validate()
        if not linkedin_valid:
            errors.append(f"LinkedIn: {linkedin_error}")

        twitter_valid, twitter_error = self.twitter.validate()
        if not twitter_valid:
            errors.append(f"Twitter: {twitter_error}")

        return len(errors) == 0, errors
