"""Tests for yt_transcriber.tui module."""
from pathlib import Path


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

    def test_false_positive_non_youtube_with_list_param(self):
        """URL on non-YouTube domain with list= must NOT be classified as playlist."""
        from yt_transcriber.tui import InputType, detect_input_type

        assert detect_input_type("https://example.com/?x=1&list=foo") == InputType.UNKNOWN

    def test_youtube_watch_with_list_first_v_after(self):
        """URL with list= before v= still detected as playlist."""
        from yt_transcriber.tui import InputType, detect_input_type

        url = "https://www.youtube.com/watch?list=PLxxx&v=abc"
        assert detect_input_type(url) == InputType.YOUTUBE_PLAYLIST

    def test_youtube_short_url_with_list(self):
        """youtu.be/abc?list=PLxxx — playlist (any list= on youtube domain)."""
        from yt_transcriber.tui import InputType, detect_input_type

        assert detect_input_type("https://youtu.be/abc?list=PLxxx") == InputType.YOUTUBE_PLAYLIST


class TestApplyValidationRules:
    def test_post_kits_forces_summarize(self):
        from yt_transcriber.tui import apply_validation_rules

        opts = {"summarize": False, "post_kits": True, "segments": False, "visual_evidence": False}
        result = apply_validation_rules(opts)
        assert result["summarize"] is True
        assert result["post_kits"] is True

    def test_visual_evidence_forces_segments(self):
        from yt_transcriber.tui import apply_validation_rules

        opts = {"summarize": False, "post_kits": False, "segments": False, "visual_evidence": True}
        result = apply_validation_rules(opts)
        assert result["segments"] is True
        assert result["visual_evidence"] is True

    def test_no_implications_keeps_values(self):
        from yt_transcriber.tui import apply_validation_rules

        opts = {"summarize": True, "post_kits": False, "segments": True, "visual_evidence": False}
        result = apply_validation_rules(opts)
        assert result == opts

    def test_both_implications_apply(self):
        from yt_transcriber.tui import apply_validation_rules

        opts = {"summarize": False, "post_kits": True, "segments": False, "visual_evidence": True}
        result = apply_validation_rules(opts)
        assert result["summarize"] is True
        assert result["segments"] is True

    def test_does_not_mutate_input(self):
        from yt_transcriber.tui import apply_validation_rules

        opts = {"summarize": False, "post_kits": True, "segments": False, "visual_evidence": False}
        opts_copy = dict(opts)
        apply_validation_rules(opts)
        assert opts == opts_copy  # original untouched

    def test_playlist_options_no_segments_keys(self):
        """Playlist options dict has no segments/visual_evidence keys; rules must not crash."""
        from yt_transcriber.tui import apply_validation_rules

        opts = {"summarize": False, "post_kits": True}
        result = apply_validation_rules(opts)
        assert result["summarize"] is True


class TestFormatCommandPreview:
    def test_transcribe_minimal(self):
        from yt_transcriber.tui import format_command_preview

        opts = {"language": None, "summarize": False, "post_kits": False, "segments": False, "visual_evidence": False}
        out = format_command_preview("transcribe", "https://youtu.be/abc", opts)

        assert "transcribe" in out
        assert "https://youtu.be/abc" in out
        assert "--summarize" not in out
        assert "--language" not in out

    def test_transcribe_with_language_and_summarize(self):
        from yt_transcriber.tui import format_command_preview

        opts = {"language": "es", "summarize": True, "post_kits": False, "segments": False, "visual_evidence": False}
        out = format_command_preview("transcribe", "https://youtu.be/abc", opts)

        assert "--language es" in out
        assert "--summarize" in out

    def test_transcribe_with_post_kits_and_visual(self):
        from yt_transcriber.tui import format_command_preview

        opts = {"language": "en", "summarize": True, "post_kits": True, "segments": True, "visual_evidence": True}
        out = format_command_preview("transcribe", "/local/video.mp4", opts)

        assert "--post-kits" in out
        assert "--visual-evidence" in out
        assert "--segments" in out
        assert "/local/video.mp4" in out

    def test_playlist_minimal(self):
        from yt_transcriber.tui import format_command_preview

        opts = {"limit": None, "language": "es", "summarize": False, "post_kits": False}
        out = format_command_preview("playlist", "https://www.youtube.com/playlist?list=PLxxx", opts)

        assert "playlist" in out
        assert "--limit" not in out

    def test_playlist_with_limit(self):
        from yt_transcriber.tui import format_command_preview

        opts = {"limit": 5, "language": "en", "summarize": True, "post_kits": False}
        out = format_command_preview("playlist", "https://www.youtube.com/playlist?list=PLxxx", opts)

        assert "--limit 5" in out
        assert "--language en" in out
        assert "--summarize" in out


class TestSmoke:
    def test_module_imports(self):
        import yt_transcriber.tui  # noqa: F401

    def test_prompt_functions_exist(self):
        from yt_transcriber.tui import (
            prompt_input_url,
            prompt_playlist_options,
            prompt_run_again,
            prompt_run_confirmation,
            prompt_transcribe_options,
        )
        # Just check they are callables; we don't invoke them (would block on stdin)
        assert callable(prompt_transcribe_options)
        assert callable(prompt_playlist_options)
        assert callable(prompt_input_url)
        assert callable(prompt_run_confirmation)
        assert callable(prompt_run_again)
