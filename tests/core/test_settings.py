"""Focused tests for core.settings defaults."""

from core.settings import AppSettings


class TestTranscriptSegmentSettings:
    """Tests for segment and visual evidence settings defaults."""

    def test_new_settings_default_values(self, monkeypatch):
        """Ensure V1 toggles are backward-compatible by default."""
        monkeypatch.delenv("TRANSCRIPT_SEGMENTS_ENABLED", raising=False)
        monkeypatch.delenv("VISUAL_EVIDENCE_ENABLED", raising=False)
        monkeypatch.delenv("VISUAL_EVIDENCE_MIN_SEGMENT_SECONDS", raising=False)

        app_settings = AppSettings()

        assert app_settings.TRANSCRIPT_SEGMENTS_ENABLED is False
        assert app_settings.VISUAL_EVIDENCE_ENABLED is False
        assert app_settings.VISUAL_EVIDENCE_MIN_SEGMENT_SECONDS == 1.0
