"""Shared data models for yt-transcriber."""

from dataclasses import dataclass


@dataclass
class TranscriptSegment:
    """Structured transcript segment with timing metadata."""

    start: float
    end: float
    text: str

    def to_dict(self) -> dict:
        """Serialize segment for JSON sidecar output."""
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }


__all__ = ["TranscriptSegment"]
