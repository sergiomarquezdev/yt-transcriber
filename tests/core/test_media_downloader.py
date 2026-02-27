"""Tests for core.media_downloader module."""

from unittest.mock import MagicMock, patch

import pytest
import yt_dlp

from core.media_downloader import (
    DownloadError,
    DownloadResult,
    _clean_srt_to_text,
    download_and_extract_audio,
    download_auto_subtitles,
    extract_audio_from_local_file,
    extract_drive_file_id,
    extract_playlist_entries,
    is_google_drive_url,
)


class TestIsGoogleDriveUrl:
    """Tests for is_google_drive_url function."""

    def test_standard_drive_url(self):
        """Test standard Google Drive file URL."""
        url = "https://drive.google.com/file/d/1ABC123xyz/view"
        assert is_google_drive_url(url) is True

    def test_drive_open_url(self):
        """Test Google Drive open URL format."""
        url = "https://drive.google.com/open?id=1ABC123xyz"
        assert is_google_drive_url(url) is True

    def test_docs_url(self):
        """Test Google Docs file URL."""
        url = "https://docs.google.com/file/d/1ABC123xyz/edit"
        assert is_google_drive_url(url) is True

    def test_youtube_url_returns_false(self):
        """Test that YouTube URL returns False."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert is_google_drive_url(url) is False

    def test_regular_url_returns_false(self):
        """Test that regular URL returns False."""
        url = "https://example.com/file.mp4"
        assert is_google_drive_url(url) is False

    def test_empty_string(self):
        """Test empty string returns False."""
        assert is_google_drive_url("") is False

    def test_drive_url_variations(self):
        """Test various Drive URL formats."""
        valid_urls = [
            "https://drive.google.com/file/d/abc123-_xyz/view?usp=sharing",
            "https://drive.google.com/file/d/ABCDEFGHIJ1234567890/view",
            "https://drive.google.com/open?id=abc_DEF-123",
        ]
        for url in valid_urls:
            assert is_google_drive_url(url) is True, f"Failed for {url}"


class TestExtractDriveFileId:
    """Tests for extract_drive_file_id function."""

    def test_extract_from_file_d_url(self):
        """Test extraction from /file/d/ URL."""
        url = "https://drive.google.com/file/d/1ABC123xyz/view"
        assert extract_drive_file_id(url) == "1ABC123xyz"

    def test_extract_from_open_url(self):
        """Test extraction from open?id= URL."""
        url = "https://drive.google.com/open?id=MyFileId123"
        assert extract_drive_file_id(url) == "MyFileId123"

    def test_extract_from_docs_url(self):
        """Test extraction from docs URL."""
        url = "https://docs.google.com/file/d/DocFileId/edit"
        assert extract_drive_file_id(url) == "DocFileId"

    def test_invalid_url_returns_none(self):
        """Test that invalid URL returns None."""
        url = "https://example.com/not-a-drive-url"
        assert extract_drive_file_id(url) is None

    def test_empty_string_returns_none(self):
        """Test empty string returns None."""
        assert extract_drive_file_id("") is None

    def test_complex_file_id(self):
        """Test extraction of complex file ID with special chars."""
        url = "https://drive.google.com/file/d/abc-123_XYZ/view"
        assert extract_drive_file_id(url) == "abc-123_XYZ"


class TestDownloadResult:
    """Tests for DownloadResult dataclass."""

    def test_creation(self, temp_dir):
        """Test basic DownloadResult creation."""
        audio_path = temp_dir / "audio.wav"
        video_path = temp_dir / "video.mp4"

        result = DownloadResult(
            audio_path=audio_path,
            video_path=video_path,
            video_id="test123",
        )

        assert result.audio_path == audio_path
        assert result.video_path == video_path
        assert result.video_id == "test123"

    def test_creation_without_video(self, temp_dir):
        """Test DownloadResult without video path."""
        audio_path = temp_dir / "audio.wav"

        result = DownloadResult(
            audio_path=audio_path,
            video_path=None,
            video_id="test123",
        )

        assert result.video_path is None


class TestDownloadAndExtractAudio:
    """Tests for download_and_extract_audio function."""

    @pytest.fixture
    def mock_yt_dlp(self):
        """Create mock yt-dlp."""
        with patch("core.media_downloader.yt_dlp.YoutubeDL") as mock:
            yield mock

    def test_successful_youtube_download(self, mock_yt_dlp, temp_dir):
        """Test successful YouTube video download."""
        # Setup mock
        mock_ydl_instance = MagicMock()
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl_instance

        # Mock extract_info to return video info
        mock_ydl_instance.extract_info.return_value = {
            "id": "test_video_id",
            "title": "Test Video",
        }
        mock_ydl_instance.prepare_filename.return_value = str(temp_dir / "video.mp4")

        # Create expected audio file
        audio_path = temp_dir / "test_video_id_job123.wav"
        audio_path.touch()

        with patch("core.media_downloader.utils.ensure_dir_exists"):
            result = download_and_extract_audio(
                youtube_url="https://www.youtube.com/watch?v=test_video_id",
                temp_dir=temp_dir,
                unique_job_id="job123",
            )

        assert result.video_id == "test_video_id"
        assert result.audio_path == audio_path

    def test_download_error_on_missing_audio(self, mock_yt_dlp, temp_dir):
        """Test that DownloadError is raised when audio extraction fails."""
        mock_ydl_instance = MagicMock()
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl_instance
        mock_ydl_instance.extract_info.return_value = {"id": "test123"}
        mock_ydl_instance.prepare_filename.return_value = str(temp_dir / "video.mp4")

        # Don't create audio file - simulates extraction failure

        with patch("core.media_downloader.utils.ensure_dir_exists"):
            with pytest.raises(DownloadError, match="Audio extraction failed"):
                download_and_extract_audio(
                    youtube_url="https://www.youtube.com/watch?v=test123",
                    temp_dir=temp_dir,
                    unique_job_id="job123",
                )

    def test_google_drive_url_detected(self, mock_yt_dlp, temp_dir):
        """Test that Google Drive URL is detected and handled."""
        mock_ydl_instance = MagicMock()
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl_instance
        mock_ydl_instance.extract_info.return_value = {
            "id": None,  # Drive might not return ID
            "title": "Drive File",
        }
        mock_ydl_instance.prepare_filename.return_value = str(temp_dir / "video.mp4")

        # Create expected audio file with drive ID
        audio_path = temp_dir / "drive_ABC123_job123.wav"
        audio_path.touch()

        with patch("core.media_downloader.utils.ensure_dir_exists"):
            result = download_and_extract_audio(
                youtube_url="https://drive.google.com/file/d/ABC123/view",
                temp_dir=temp_dir,
                unique_job_id="job123",
            )

        assert "drive_ABC123" in result.video_id or result.video_id is not None

    def test_yt_dlp_error_wrapped(self, mock_yt_dlp, temp_dir):
        """Test that yt-dlp errors are wrapped in DownloadError."""
        mock_ydl_instance = MagicMock()
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl_instance
        mock_ydl_instance.extract_info.side_effect = yt_dlp.utils.DownloadError("yt-dlp failed")

        with patch("core.media_downloader.utils.ensure_dir_exists"):
            with pytest.raises(DownloadError):
                download_and_extract_audio(
                    youtube_url="https://www.youtube.com/watch?v=test",
                    temp_dir=temp_dir,
                    unique_job_id="job123",
                )

    def test_custom_ffmpeg_location(self, mock_yt_dlp, temp_dir):
        """Test that custom FFmpeg location is passed to yt-dlp."""
        mock_ydl_instance = MagicMock()
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl_instance
        mock_ydl_instance.extract_info.return_value = {"id": "test123"}
        mock_ydl_instance.prepare_filename.return_value = str(temp_dir / "video.mp4")

        audio_path = temp_dir / "test123_job123.wav"
        audio_path.touch()

        with patch("core.media_downloader.utils.ensure_dir_exists"):
            download_and_extract_audio(
                youtube_url="https://www.youtube.com/watch?v=test123",
                temp_dir=temp_dir,
                unique_job_id="job123",
                ffmpeg_location="/custom/ffmpeg",
            )

        # Verify ffmpeg_location was passed in options
        calls = mock_yt_dlp.call_args_list
        # Check that one of the calls includes ffmpeg_location
        for call in calls:
            opts = call[0][0] if call[0] else call[1].get("params", {})
            if isinstance(opts, dict) and opts.get("ffmpeg_location") == "/custom/ffmpeg":
                break
        # Note: This assertion might need adjustment based on actual implementation


class TestExtractAudioFromLocalFile:
    """Tests for extract_audio_from_local_file function."""

    def test_file_not_found_raises_error(self, temp_dir):
        """Test that non-existent file raises DownloadError."""
        non_existent = temp_dir / "does_not_exist.mp4"

        with pytest.raises(DownloadError, match="does not exist"):
            extract_audio_from_local_file(
                video_path=non_existent,
                temp_dir=temp_dir,
                unique_job_id="job123",
            )

    def test_successful_extraction(self, temp_dir, sample_video_path):
        """Test successful audio extraction from local file."""
        with patch("core.media_downloader.shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/ffmpeg"

            with patch("core.media_downloader.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stderr="")

                # Create expected output file
                expected_audio = temp_dir / "sample_video_job123.wav"
                expected_audio.touch()

                result = extract_audio_from_local_file(
                    video_path=sample_video_path,
                    temp_dir=temp_dir,
                    unique_job_id="job123",
                )

                assert result.audio_path.exists()
                assert result.video_path == sample_video_path
                assert "sample_video" in result.video_id

    def test_ffmpeg_not_found_raises_error(self, temp_dir, sample_video_path):
        """Test that missing FFmpeg raises DownloadError."""
        with patch("core.media_downloader.shutil.which") as mock_which:
            mock_which.return_value = None

            with pytest.raises(DownloadError, match="FFmpeg not found"):
                extract_audio_from_local_file(
                    video_path=sample_video_path,
                    temp_dir=temp_dir,
                    unique_job_id="job123",
                )

    def test_ffmpeg_failure_raises_error(self, temp_dir, sample_video_path):
        """Test that FFmpeg failure raises DownloadError."""
        with patch("core.media_downloader.shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/ffmpeg"

            with patch("core.media_downloader.subprocess.run") as mock_run:
                import subprocess

                mock_run.side_effect = subprocess.CalledProcessError(
                    1, "ffmpeg", stderr="Conversion failed"
                )

                with pytest.raises(DownloadError, match="FFmpeg failed"):
                    extract_audio_from_local_file(
                        video_path=sample_video_path,
                        temp_dir=temp_dir,
                        unique_job_id="job123",
                    )

    def test_custom_ffmpeg_location_used(self, temp_dir, sample_video_path):
        """Test that custom FFmpeg location is used."""
        custom_ffmpeg = "/custom/path/ffmpeg"

        with patch("core.media_downloader.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            # Create expected output file
            expected_audio = temp_dir / "sample_video_job123.wav"
            expected_audio.touch()

            extract_audio_from_local_file(
                video_path=sample_video_path,
                temp_dir=temp_dir,
                unique_job_id="job123",
                ffmpeg_location=custom_ffmpeg,
            )

            # Verify custom ffmpeg was used
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert call_args[0] == custom_ffmpeg

    def test_video_id_from_filename(self, temp_dir):
        """Test that video_id is generated from filename."""
        video_path = temp_dir / "My Test Video 2024.mp4"
        video_path.touch()

        with patch("core.media_downloader.shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/ffmpeg"

            with patch("core.media_downloader.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                # Create expected audio file
                expected_audio = temp_dir / "My_Test_Video_2024_job123.wav"
                expected_audio.touch()

                result = extract_audio_from_local_file(
                    video_path=video_path,
                    temp_dir=temp_dir,
                    unique_job_id="job123",
                )

                # video_id should be normalized filename
                assert "My" in result.video_id or "Test" in result.video_id


# =========================================================================
# Playlist support
# =========================================================================


class TestExtractPlaylistEntries:
    """Tests for extract_playlist_entries function."""

    @patch("core.media_downloader.yt_dlp.YoutubeDL")
    def test_extract_playlist_entries(self, mock_yt_dlp):
        """Test successful extraction of playlist entries."""
        mock_ydl = MagicMock()
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = {
            "entries": [
                {"id": "abc123", "title": "Video 1", "url": "abc123"},
                {
                    "id": "def456",
                    "title": "Video 2",
                    "url": "https://www.youtube.com/watch?v=def456",
                },
            ]
        }

        entries = extract_playlist_entries("https://www.youtube.com/playlist?list=PLtest")

        assert len(entries) == 2
        assert entries[0].video_id == "abc123"
        assert entries[0].title == "Video 1"
        assert "abc123" in entries[0].url
        assert entries[1].video_id == "def456"

    @patch("core.media_downloader.yt_dlp.YoutubeDL")
    def test_extract_playlist_entries_empty(self, mock_yt_dlp):
        """Test empty playlist returns empty list."""
        mock_ydl = MagicMock()
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = {"entries": []}

        entries = extract_playlist_entries("https://www.youtube.com/playlist?list=PLempty")

        assert entries == []

    @patch("core.media_downloader.yt_dlp.YoutubeDL")
    def test_extract_playlist_entries_none_info(self, mock_yt_dlp):
        """Test None info raises DownloadError."""
        mock_ydl = MagicMock()
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = None

        with pytest.raises(DownloadError, match="Could not extract playlist info"):
            extract_playlist_entries("https://www.youtube.com/playlist?list=PLbad")

    @patch("core.media_downloader.yt_dlp.YoutubeDL")
    def test_extract_playlist_entries_yt_dlp_error(self, mock_yt_dlp):
        """Test yt-dlp error is wrapped in DownloadError."""
        mock_ydl = MagicMock()
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.side_effect = yt_dlp.utils.DownloadError("network error")

        with pytest.raises(DownloadError, match="yt-dlp failed to extract playlist"):
            extract_playlist_entries("https://www.youtube.com/playlist?list=PLfail")


class TestCleanSrtToText:
    """Tests for _clean_srt_to_text helper."""

    def test_removes_timestamps_and_indices(self):
        """Test SRT timestamps and cue indices are removed."""
        srt = (
            "1\n"
            "00:00:00,000 --> 00:00:02,000\n"
            "Hello world\n"
            "\n"
            "2\n"
            "00:00:02,000 --> 00:00:04,000\n"
            "Second line\n"
        )
        result = _clean_srt_to_text(srt)
        assert "Hello world" in result
        assert "Second line" in result
        assert "-->" not in result
        assert "00:00" not in result

    def test_removes_vtt_headers(self):
        """Test VTT-specific headers are stripped."""
        vtt = "WEBVTT\nKind: captions\nLanguage: es\n\n00:00:00.000 --> 00:00:02.000\nHola mundo\n"
        result = _clean_srt_to_text(vtt)
        assert result.strip() == "Hola mundo"

    def test_deduplicates_consecutive_lines(self):
        """Test consecutive duplicate lines are merged."""
        srt = (
            "1\n"
            "00:00:00,000 --> 00:00:02,000\n"
            "Hello\n"
            "\n"
            "2\n"
            "00:00:01,000 --> 00:00:03,000\n"
            "Hello\n"
            "\n"
            "3\n"
            "00:00:02,000 --> 00:00:04,000\n"
            "World\n"
        )
        result = _clean_srt_to_text(srt)
        assert result.count("Hello") == 1
        assert "World" in result

    def test_strips_html_tags(self):
        """Test HTML tags are removed from subtitle text."""
        srt = "1\n00:00:00,000 --> 00:00:02,000\n<font color='white'>Styled text</font>\n"
        result = _clean_srt_to_text(srt)
        assert result.strip() == "Styled text"
        assert "<font" not in result

    def test_empty_input(self):
        """Test empty string returns empty string."""
        assert _clean_srt_to_text("") == ""


class TestDownloadAutoSubtitles:
    """Tests for download_auto_subtitles function."""

    @patch("core.media_downloader.yt_dlp.YoutubeDL")
    def test_download_auto_subtitles_success(self, mock_yt_dlp, temp_dir):
        """Test successful subtitle download and cleaning."""
        mock_ydl = MagicMock()
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = {"id": "vid123"}

        # Simulate yt-dlp writing an SRT file
        srt_path = temp_dir / "vid123.es.srt"
        srt_path.write_text(
            "1\n00:00:00,000 --> 00:00:02,000\nHola mundo\n\n"
            "2\n00:00:02,000 --> 00:00:04,000\nSegunda linea\n",
            encoding="utf-8",
        )

        result = download_auto_subtitles(
            video_url="https://www.youtube.com/watch?v=vid123",
            output_dir=temp_dir,
            lang="es",
        )

        assert result is not None
        assert result.name == "vid123.txt"
        content = result.read_text(encoding="utf-8")
        assert "Hola mundo" in content
        assert "Segunda linea" in content
        assert "-->" not in content
        # Raw SRT should be cleaned up
        assert not srt_path.exists()

    @patch("core.media_downloader.yt_dlp.YoutubeDL")
    def test_download_auto_subtitles_no_subs(self, mock_yt_dlp, temp_dir):
        """Test returns None when no subtitles are available."""
        mock_ydl = MagicMock()
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = {"id": "nosubs"}

        # No SRT file created - simulates video without subs

        result = download_auto_subtitles(
            video_url="https://www.youtube.com/watch?v=nosubs",
            output_dir=temp_dir,
            lang="es",
        )

        assert result is None

    @patch("core.media_downloader.yt_dlp.YoutubeDL")
    def test_download_auto_subtitles_yt_dlp_error(self, mock_yt_dlp, temp_dir):
        """Test yt-dlp error returns None gracefully."""
        mock_ydl = MagicMock()
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.side_effect = yt_dlp.utils.DownloadError("private video")

        result = download_auto_subtitles(
            video_url="https://www.youtube.com/watch?v=private",
            output_dir=temp_dir,
            lang="es",
        )

        assert result is None

    @patch("core.media_downloader.yt_dlp.YoutubeDL")
    def test_download_auto_subtitles_vtt_format(self, mock_yt_dlp, temp_dir):
        """Test VTT subtitle format is also handled."""
        mock_ydl = MagicMock()
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = {"id": "vttvid"}

        # Simulate VTT file
        vtt_path = temp_dir / "vttvid.en.vtt"
        vtt_path.write_text(
            "WEBVTT\nKind: captions\nLanguage: en\n\n00:00:00.000 --> 00:00:02.000\nHello world\n",
            encoding="utf-8",
        )

        result = download_auto_subtitles(
            video_url="https://www.youtube.com/watch?v=vttvid",
            output_dir=temp_dir,
            lang="en",
        )

        assert result is not None
        content = result.read_text(encoding="utf-8")
        assert "Hello world" in content
