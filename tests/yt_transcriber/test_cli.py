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
        summarize=False,
        post_kits=False,
    ):
        """Build a mock args namespace matching the playlist subcommand."""
        args = MagicMock()
        args.url = url
        args.limit = limit
        args.language = language
        args.summarize = summarize
        args.post_kits = post_kits
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

        txt1 = tmp_path / "v1.txt"
        txt1.write_text("transcript 1", encoding="utf-8")
        txt2 = tmp_path / "v2.txt"
        txt2.write_text("transcript 2", encoding="utf-8")
        mock_download.side_effect = [txt1, txt2]

        args = self._make_args()

        with patch("core.settings.settings") as mock_settings:
            mock_settings.OUTPUT_TRANSCRIPTS_DIR = tmp_path
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

        txt = tmp_path / "v3.txt"
        txt.write_text("transcript 3", encoding="utf-8")
        mock_download.return_value = txt

        args = self._make_args(limit=1)

        with patch("core.settings.settings") as mock_settings:
            mock_settings.OUTPUT_TRANSCRIPTS_DIR = tmp_path
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

        txt = tmp_path / "v3.txt"
        txt.write_text("transcript 3", encoding="utf-8")
        # First fails with exception, second returns None (no subs), third succeeds
        mock_download.side_effect = [
            Exception("network error"),
            None,
            txt,
        ]

        args = self._make_args()

        with patch("core.settings.settings") as mock_settings:
            mock_settings.OUTPUT_TRANSCRIPTS_DIR = tmp_path
            command_playlist(args)

        # All 3 should have been attempted
        assert mock_download.call_count == 3

    @patch("core.media_downloader.extract_playlist_entries")
    def test_command_playlist_empty_playlist(self, mock_extract):
        """Test empty playlist exits cleanly."""
        from yt_transcriber.cli import command_playlist

        mock_extract.return_value = []
        args = self._make_args()

        with pytest.raises(SystemExit) as exc_info:
            command_playlist(args)
        assert exc_info.value.code == 0

    @patch("core.media_downloader.extract_playlist_entries")
    def test_command_playlist_extract_failure(self, mock_extract):
        """Test playlist extraction failure exits with error."""
        from yt_transcriber.cli import command_playlist

        mock_extract.side_effect = DownloadError("bad playlist")
        args = self._make_args()

        with pytest.raises(SystemExit) as exc_info:
            command_playlist(args)
        assert exc_info.value.code == 1
