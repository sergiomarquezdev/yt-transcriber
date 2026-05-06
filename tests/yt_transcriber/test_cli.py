"""Tests for the playlist CLI command."""

from unittest.mock import MagicMock, patch

import pytest

from core.media_downloader import DownloadError, PlaylistEntry


class TestCommandPlaylist:
    """Tests for command_playlist function."""

    def _make_args(
        self,
        url="https://www.youtube.com/playlist?list=PLtest",
        limit=None,
        language="es",
    ):
        """Build a mock args namespace matching the playlist subcommand."""
        args = MagicMock()
        args.url = url
        args.limit = limit
        args.language = language
        return args

    @patch("core.media_downloader.download_auto_subtitles")
    @patch("core.media_downloader.extract_playlist_entries")
    def test_command_playlist_basic(self, mock_extract, mock_download, tmp_path):
        """Test basic playlist download without summaries."""
        from yt_transcriber.cli import command_playlist

        mock_extract.return_value = [
            PlaylistEntry(video_id="v1", title="Video 1", url="https://www.youtube.com/watch?v=v1"),
            PlaylistEntry(video_id="v2", title="Video 2", url="https://www.youtube.com/watch?v=v2"),
        ]

        # mock returns a file inside the per-video dir that the loop creates
        def fake_download(video_url, output_dir, lang):
            f = output_dir / "raw.txt"
            f.write_text("transcript", encoding="utf-8")
            return f

        mock_download.side_effect = fake_download

        args = self._make_args()

        with patch("yt_transcriber.cli.settings") as mock_settings:
            mock_settings.OUTPUT_BASE_DIR = tmp_path
            command_playlist(args)

        assert mock_download.call_count == 2

    @patch("core.media_downloader.download_auto_subtitles")
    @patch("core.media_downloader.extract_playlist_entries")
    def test_command_playlist_with_limit(self, mock_extract, mock_download, tmp_path):
        """Test --limit slices last N entries."""
        from yt_transcriber.cli import command_playlist

        mock_extract.return_value = [
            PlaylistEntry(video_id="v1", title="Video 1", url="https://www.youtube.com/watch?v=v1"),
            PlaylistEntry(video_id="v2", title="Video 2", url="https://www.youtube.com/watch?v=v2"),
            PlaylistEntry(video_id="v3", title="Video 3", url="https://www.youtube.com/watch?v=v3"),
        ]

        def fake_download(video_url, output_dir, lang):
            f = output_dir / "raw.txt"
            f.write_text("transcript 3", encoding="utf-8")
            return f

        mock_download.side_effect = fake_download

        args = self._make_args(limit=1)

        with patch("yt_transcriber.cli.settings") as mock_settings:
            mock_settings.OUTPUT_BASE_DIR = tmp_path
            command_playlist(args)

        # Only the last 1 video should be processed
        assert mock_download.call_count == 1

    @patch("core.media_downloader.download_auto_subtitles")
    @patch("core.media_downloader.extract_playlist_entries")
    def test_command_playlist_continues_on_error(self, mock_extract, mock_download, tmp_path):
        """Test that batch continues when one video fails."""
        from yt_transcriber.cli import command_playlist

        mock_extract.return_value = [
            PlaylistEntry(video_id="v1", title="Video 1", url="https://www.youtube.com/watch?v=v1"),
            PlaylistEntry(video_id="v2", title="Video 2", url="https://www.youtube.com/watch?v=v2"),
            PlaylistEntry(video_id="v3", title="Video 3", url="https://www.youtube.com/watch?v=v3"),
        ]

        call_count = [0]

        def fake_download(video_url, output_dir, lang):
            call_count[0] += 1
            n = call_count[0]
            if n == 1:
                raise Exception("network error")
            if n == 2:
                return None  # no subs
            # n == 3: success
            f = output_dir / "raw.txt"
            f.write_text("transcript 3", encoding="utf-8")
            return f

        mock_download.side_effect = fake_download

        args = self._make_args()

        with patch("yt_transcriber.cli.settings") as mock_settings:
            mock_settings.OUTPUT_BASE_DIR = tmp_path
            command_playlist(args)

        # All 3 should have been attempted
        assert mock_download.call_count == 3

    @patch("core.media_downloader.extract_playlist_entries")
    def test_command_playlist_empty_playlist(self, mock_extract):
        """Test empty playlist returns zero stats (no longer sys.exit(0))."""
        from yt_transcriber.cli import command_playlist

        mock_extract.return_value = []
        args = self._make_args()

        result = command_playlist(args)
        assert result == {"successful": 0, "failed": 0, "files": []}

    @patch("core.media_downloader.extract_playlist_entries")
    def test_command_playlist_extract_failure(self, mock_extract):
        """Test playlist extraction failure exits with error."""
        from yt_transcriber.cli import command_playlist

        mock_extract.side_effect = DownloadError("bad playlist")
        args = self._make_args()

        with pytest.raises(SystemExit) as exc_info:
            command_playlist(args)
        assert exc_info.value.code == 1


class TestTranscribeFlags:
    """Focused tests for transcribe CLI flags wiring."""

    def test_transcribe_segments_flag(self):
        """--segments is parsed and forwarded as override=True."""
        from yt_transcriber.cli import main

        argv = [
            "yt-transcriber",
            "transcribe",
            "--url",
            "https://www.youtube.com/watch?v=abcdefghijk",
            "--segments",
        ]

        with patch("sys.argv", argv):
            with patch("yt_transcriber.cli.command_transcribe") as mock_command:
                main()

        args = mock_command.call_args.args[0]
        assert args.segments_override is True

    def test_transcribe_visual_evidence_flag(self):
        """--visual-evidence is parsed and forwarded as override=True."""
        from yt_transcriber.cli import main

        argv = [
            "yt-transcriber",
            "transcribe",
            "--url",
            "https://www.youtube.com/watch?v=abcdefghijk",
            "--visual-evidence",
        ]

        with patch("sys.argv", argv):
            with patch("yt_transcriber.cli.command_transcribe") as mock_command:
                main()

        args = mock_command.call_args.args[0]
        assert args.visual_override is True

    def test_visual_implies_segments_override(self):
        """Visual override flows to service so implication is resolved centrally."""
        from yt_transcriber.cli import run_transcribe_command

        with patch("yt_transcriber.cli.get_youtube_title", return_value="Test"):
            with patch("yt_transcriber.cli.whisper_model_context") as mock_context:
                with patch("yt_transcriber.cli.process_transcription") as mock_process:
                    mock_context.return_value.__enter__.return_value = MagicMock()
                    mock_process.return_value = None

                    run_transcribe_command(
                        url="https://www.youtube.com/watch?v=abcdefghijk",
                        visual_override=True,
                    )

                    assert mock_process.call_args.kwargs["visual_override"] is True
                    assert mock_process.call_args.kwargs["segments_override"] is None


class TestRunPlaylistCommand:
    """Tests for run_playlist_command (programmatic wrapper, no sys.exit)."""

    @patch("yt_transcriber.cli.command_playlist")
    def test_returns_dict_with_stats(self, mock_command_playlist, tmp_path):
        from yt_transcriber.cli import run_playlist_command

        # command_playlist is invoked with a Namespace built internally; we don't care
        # about its side effects in this unit test, only that the wrapper returns
        # a dict with the expected keys without sys.exit.
        mock_command_playlist.return_value = None

        result = run_playlist_command(
            url="https://www.youtube.com/playlist?list=PLxxx",
            limit=3,
            language="es",
        )

        assert isinstance(result, dict)
        assert "successful" in result
        assert "failed" in result
        assert "files" in result
        assert isinstance(result["files"], list)

    @patch("yt_transcriber.cli.command_playlist")
    def test_does_not_call_sys_exit_on_systemexit(self, mock_command_playlist):
        """If command_playlist raises SystemExit, the wrapper catches it and reports failure."""
        from yt_transcriber.cli import run_playlist_command

        mock_command_playlist.side_effect = SystemExit(1)

        result = run_playlist_command(
            url="https://www.youtube.com/playlist?list=PLxxx",
            limit=None,
            language="es",
        )

        assert result["failed"] >= 1
        assert result["successful"] == 0

    @patch("yt_transcriber.cli.command_playlist")
    def test_does_not_call_sys_exit_on_exception(self, mock_command_playlist):
        """If command_playlist raises a generic Exception, wrapper catches it."""
        from yt_transcriber.cli import run_playlist_command

        mock_command_playlist.side_effect = RuntimeError("boom")

        result = run_playlist_command(
            url="https://www.youtube.com/playlist?list=PLxxx",
            limit=None,
            language="es",
        )

        assert result["failed"] >= 1

    @patch("yt_transcriber.cli.command_playlist")
    def test_returns_real_stats_when_command_playlist_returns_dict(self, mock_command_playlist):
        """When command_playlist returns a stats dict, wrapper passes it through."""
        from yt_transcriber.cli import run_playlist_command

        mock_command_playlist.return_value = {"successful": 7, "failed": 2, "files": ["a.txt", "b.txt"]}

        result = run_playlist_command(
            url="https://www.youtube.com/playlist?list=PLxxx",
            limit=10,
            language="es",
        )

        assert result == {"successful": 7, "failed": 2, "files": ["a.txt", "b.txt"]}

    @patch("yt_transcriber.cli.command_playlist")
    def test_empty_playlist_returns_zero_zero(self, mock_command_playlist):
        """Empty playlist (sys.exit(0)): successful=0, failed=0, not successful=1."""
        from yt_transcriber.cli import run_playlist_command

        mock_command_playlist.side_effect = SystemExit(0)

        result = run_playlist_command(
            url="https://www.youtube.com/playlist?list=PLempty",
            limit=None,
            language="es",
        )

        # sys.exit(0) means "no work to do" (empty playlist) — neither success nor failure
        assert result["successful"] == 0
        assert result["failed"] == 0


class _fake_ctx:
    def __enter__(self):
        return object()

    def __exit__(self, *exc):
        return False


def test_run_transcribe_command_returns_single_path(monkeypatch, tmp_path):
    """run_transcribe_command returns just the transcript path string (or None)."""
    from yt_transcriber import cli

    fake_path = tmp_path / "video" / "video.txt"
    fake_path.parent.mkdir(parents=True)
    fake_path.write_text("hi", encoding="utf-8")

    def fake_pt(**kwargs):
        return fake_path

    monkeypatch.setattr(cli, "process_transcription", fake_pt)
    monkeypatch.setattr(cli, "whisper_model_context", lambda: _fake_ctx())

    result = cli.run_transcribe_command(url="https://youtu.be/abc")
    assert result == str(fake_path)
