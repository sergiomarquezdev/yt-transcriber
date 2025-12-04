"""Tests for core.models module."""

from datetime import datetime

import pytest

from core.models import TimestampedSection, VideoSummary


class TestTimestampedSection:
    """Tests for TimestampedSection dataclass."""

    def test_basic_creation(self):
        """Test basic TimestampedSection creation."""
        ts = TimestampedSection(
            timestamp="05:30",
            description="Introduction section",
        )
        assert ts.timestamp == "05:30"
        assert ts.description == "Introduction section"
        assert ts.importance == 3  # Default value

    def test_with_importance(self):
        """Test TimestampedSection with custom importance."""
        ts = TimestampedSection(
            timestamp="10:00",
            description="Key moment",
            importance=5,
        )
        assert ts.importance == 5

    def test_str_representation(self):
        """Test __str__ method returns markdown format."""
        ts = TimestampedSection(
            timestamp="05:30",
            description="Test description",
        )
        result = str(ts)
        assert "**05:30**" in result
        assert "Test description" in result
        assert result.startswith("-")

    def test_hour_timestamp_format(self):
        """Test with hour:minute:second format."""
        ts = TimestampedSection(
            timestamp="01:30:45",
            description="Long video section",
        )
        assert ts.timestamp == "01:30:45"


class TestVideoSummary:
    """Tests for VideoSummary dataclass."""

    @pytest.fixture
    def basic_summary(self):
        """Create a basic VideoSummary for testing."""
        return VideoSummary(
            video_url="https://youtube.com/watch?v=abc123",
            video_title="Test Video",
            video_id="abc123",
            executive_summary="This is the executive summary.",
            key_points=["Point 1", "Point 2", "Point 3"],
            timestamps=[
                TimestampedSection("00:00", "Intro", 3),
                TimestampedSection("05:00", "Main", 5),
            ],
            conclusion="This is the conclusion.",
            action_items=["Action 1", "Action 2"],
            word_count=1000,
            estimated_duration_minutes=10.5,
            language="en",
            generated_at=datetime(2024, 1, 15, 10, 30, 0),
        )

    def test_basic_creation(self, basic_summary):
        """Test basic VideoSummary creation."""
        assert basic_summary.video_id == "abc123"
        assert basic_summary.language == "en"
        assert len(basic_summary.key_points) == 3

    def test_to_markdown_contains_all_sections(self, basic_summary):
        """Test that to_markdown includes all sections."""
        md = basic_summary.to_markdown()

        # Check header
        assert "# 📹 Resumen: Test Video" in md
        assert "https://youtube.com/watch?v=abc123" in md

        # Check sections
        assert "## 🎯 Resumen Ejecutivo" in md
        assert "## 🔑 Puntos Clave" in md
        assert "## ⏱️ Momentos Importantes" in md
        assert "## 💡 Conclusión" in md
        assert "## ✅ Action Items" in md

        # Check content
        assert "This is the executive summary." in md
        assert "Point 1" in md
        assert "This is the conclusion." in md
        assert "Action 1" in md

    def test_to_markdown_includes_statistics(self, basic_summary):
        """Test that statistics are included in markdown."""
        md = basic_summary.to_markdown()

        assert "1,000 palabras" in md or "1000 palabras" in md
        assert "10.5 minutos" in md

    def test_to_markdown_timestamps_formatted(self, basic_summary):
        """Test that timestamps are properly formatted."""
        md = basic_summary.to_markdown()

        assert "**00:00**" in md
        assert "**05:00**" in md
        assert "Intro" in md
        assert "Main" in md

    def test_to_dict_structure(self, basic_summary):
        """Test to_dict returns correct structure."""
        d = basic_summary.to_dict()

        assert d["video_url"] == "https://youtube.com/watch?v=abc123"
        assert d["video_id"] == "abc123"
        assert d["language"] == "en"
        assert d["word_count"] == 1000
        assert len(d["key_points"]) == 3
        assert len(d["timestamps"]) == 2

    def test_to_dict_timestamps_serialized(self, basic_summary):
        """Test that timestamps are properly serialized in dict."""
        d = basic_summary.to_dict()

        assert d["timestamps"][0]["timestamp"] == "00:00"
        assert d["timestamps"][0]["description"] == "Intro"
        assert d["timestamps"][0]["importance"] == 3

    def test_to_dict_datetime_serialized(self, basic_summary):
        """Test that datetime is properly serialized."""
        d = basic_summary.to_dict()

        assert d["generated_at"] == "2024-01-15T10:30:00"

    def test_empty_timestamps_handled(self):
        """Test that empty timestamps list is handled."""
        summary = VideoSummary(
            video_url="https://example.com",
            video_title="No Timestamps",
            video_id="test",
            executive_summary="Summary",
            key_points=["Point"],
            timestamps=[],  # Empty
            conclusion="Conclusion",
            action_items=[],
            word_count=100,
            estimated_duration_minutes=1.0,
            language="en",
            generated_at=datetime.now(),
        )

        md = summary.to_markdown()
        # Should not have timestamps section if empty
        assert "Momentos Importantes" not in md or "---" in md

    def test_empty_action_items_handled(self):
        """Test that empty action_items list is handled."""
        summary = VideoSummary(
            video_url="https://example.com",
            video_title="No Actions",
            video_id="test",
            executive_summary="Summary",
            key_points=["Point"],
            timestamps=[],
            conclusion="Conclusion",
            action_items=[],  # Empty
            word_count=100,
            estimated_duration_minutes=1.0,
            language="en",
            generated_at=datetime.now(),
        )

        md = summary.to_markdown()
        # Should still be valid markdown
        assert "Resumen:" in md or "Resumen Ejecutivo" in md

    def test_spanish_language_marker(self):
        """Test summary with Spanish language."""
        summary = VideoSummary(
            video_url="https://example.com",
            video_title="Video en Español",
            video_id="test_es",
            executive_summary="Este es el resumen.",
            key_points=["Punto 1"],
            timestamps=[],
            conclusion="Conclusión.",
            action_items=[],
            word_count=100,
            estimated_duration_minutes=1.0,
            language="es",
            generated_at=datetime.now(),
        )

        assert summary.language == "es"
        d = summary.to_dict()
        assert d["language"] == "es"
