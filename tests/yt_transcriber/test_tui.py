"""Tests for yt_transcriber.tui module."""
from pathlib import Path

import pytest


class TestDetectInputType:
    def test_local_existing_file(self, tmp_path: Path):
        from yt_transcriber.tui import InputType, detect_input_type

        f = tmp_path / "video.mp4"
        f.write_text("fake")

        assert detect_input_type(str(f)) == InputType.LOCAL

    def test_local_nonexistent_path(self, tmp_path: Path):
        from yt_transcriber.tui import InputType, detect_input_type

        assert detect_input_type(str(tmp_path / "does_not_exist.mp4")) == InputType.UNKNOWN

    def test_youtube_video_long_url(self):
        from yt_transcriber.tui import InputType, detect_input_type

        assert detect_input_type("https://www.youtube.com/watch?v=abc123") == InputType.YOUTUBE_VIDEO

    def test_youtube_video_short_url(self):
        from yt_transcriber.tui import InputType, detect_input_type

        assert detect_input_type("https://youtu.be/abc123") == InputType.YOUTUBE_VIDEO

    def test_youtube_playlist_explicit(self):
        from yt_transcriber.tui import InputType, detect_input_type

        assert detect_input_type("https://www.youtube.com/playlist?list=PLxxx") == InputType.YOUTUBE_PLAYLIST

    def test_youtube_video_with_list_param_prefers_playlist(self):
        """An URL with both v= and list= should be treated as playlist."""
        from yt_transcriber.tui import InputType, detect_input_type

        url = "https://www.youtube.com/watch?v=abc&list=PLxxx"
        assert detect_input_type(url) == InputType.YOUTUBE_PLAYLIST

    def test_drive_file_url(self):
        from yt_transcriber.tui import InputType, detect_input_type

        assert detect_input_type("https://drive.google.com/file/d/xxx/view") == InputType.DRIVE

    def test_drive_docs_url(self):
        from yt_transcriber.tui import InputType, detect_input_type

        assert detect_input_type("https://docs.google.com/document/d/xxx") == InputType.DRIVE

    def test_unknown_url(self):
        from yt_transcriber.tui import InputType, detect_input_type

        assert detect_input_type("https://example.com") == InputType.UNKNOWN

    def test_empty_string(self):
        from yt_transcriber.tui import InputType, detect_input_type

        assert detect_input_type("") == InputType.UNKNOWN

    def test_whitespace_string(self):
        from yt_transcriber.tui import InputType, detect_input_type

        assert detect_input_type("   ") == InputType.UNKNOWN
