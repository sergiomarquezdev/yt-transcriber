"""Tests for core.models module."""

from core.models import TranscriptSegment


class TestTranscriptSegment:
    """Tests for TranscriptSegment dataclass."""

    def test_fields(self):
        """Test TranscriptSegment field values."""
        segment = TranscriptSegment(start=1.25, end=2.5, text="Hello")

        assert segment.start == 1.25
        assert segment.end == 2.5
        assert segment.text == "Hello"

    def test_to_dict(self):
        """Test TranscriptSegment serialization."""
        segment = TranscriptSegment(start=0.0, end=3.0, text="Hola")

        assert segment.to_dict() == {
            "start": 0.0,
            "end": 3.0,
            "text": "Hola",
        }
