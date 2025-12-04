"""Tests for yt_transcriber.service module."""

from unittest.mock import MagicMock, patch

import pytest

from yt_transcriber.service import process_transcription


class TestProcessTranscription:
    """Tests for process_transcription function."""

    @pytest.fixture
    def mock_dependencies(self, temp_dir):
        """Set up common mocks for process_transcription."""
        with patch("yt_transcriber.service.settings") as mock_settings:
            mock_settings.TEMP_DOWNLOAD_DIR = temp_dir / "temp"
            mock_settings.OUTPUT_TRANSCRIPTS_DIR = temp_dir / "transcripts"
            mock_settings.SUMMARY_OUTPUT_DIR = temp_dir / "summaries"
            mock_settings.TRANSCRIPT_CACHE_DIR = temp_dir / "cache"
            mock_settings.TRANSCRIPT_CACHE_ENABLED = False
            mock_settings.SUMMARIZER_MODEL = "gemini-2.5-flash"

            (temp_dir / "temp").mkdir(exist_ok=True)
            (temp_dir / "transcripts").mkdir(exist_ok=True)
            (temp_dir / "summaries").mkdir(exist_ok=True)

            yield mock_settings

    @pytest.fixture
    def mock_whisper_model(self):
        """Create mock Whisper model."""
        model = MagicMock()
        model.transcribe.return_value = {
            "text": "This is the transcribed text.",
            "language": "en",
        }
        return model

    # =========================================================================
    # BASIC FUNCTIONALITY TESTS
    # =========================================================================

    def test_successful_transcription_only(self, mock_dependencies, mock_whisper_model, temp_dir):
        """Test successful transcription without summary."""
        with patch("yt_transcriber.service.download_and_extract_audio") as mock_download:
            with patch("yt_transcriber.service.transcribe_audio_file") as mock_transcribe:
                with patch("yt_transcriber.service.utils") as mock_utils:
                    # Setup mocks
                    audio_path = temp_dir / "temp" / "audio.wav"
                    audio_path.parent.mkdir(exist_ok=True)
                    audio_path.touch()

                    mock_download.return_value = MagicMock(
                        audio_path=audio_path,
                        video_path=None,
                        video_id="test123",
                    )
                    mock_transcribe.return_value = MagicMock(
                        text="Transcribed text here",
                        language="en",
                    )

                    transcript_path = temp_dir / "transcripts" / "test.txt"
                    transcript_path.parent.mkdir(exist_ok=True)
                    transcript_path.write_text("Transcribed text")
                    mock_utils.save_transcription_to_file.return_value = transcript_path
                    mock_utils.normalize_title_for_filename.return_value = "test_video"
                    mock_utils.cleanup_temp_dir = MagicMock()

                    result = process_transcription(
                        youtube_url="https://www.youtube.com/watch?v=test123",
                        title="Test Video",
                        model=mock_whisper_model,
                        only_transcript=True,
                    )

                    transcript, summary_en, summary_es, post_kits = result

                    assert transcript is not None
                    assert summary_en is None
                    assert summary_es is None

    def test_returns_none_on_download_error(self, mock_dependencies, mock_whisper_model):
        """Test that None is returned on download error."""
        with patch("yt_transcriber.service.download_and_extract_audio") as mock_download:
            with patch("yt_transcriber.service.utils") as mock_utils:
                from core.media_downloader import DownloadError

                mock_download.side_effect = DownloadError("Download failed")
                mock_utils.cleanup_temp_dir = MagicMock()

                result = process_transcription(
                    youtube_url="https://www.youtube.com/watch?v=test",
                    title="Test",
                    model=mock_whisper_model,
                )

                assert result == (None, None, None, None)

    def test_returns_none_on_transcription_error(
        self, mock_dependencies, mock_whisper_model, temp_dir
    ):
        """Test that None is returned on transcription error."""
        with patch("yt_transcriber.service.download_and_extract_audio") as mock_download:
            with patch("yt_transcriber.service.transcribe_audio_file") as mock_transcribe:
                with patch("yt_transcriber.service.utils") as mock_utils:
                    from core.media_transcriber import TranscriptionError

                    audio_path = temp_dir / "temp" / "audio.wav"
                    audio_path.parent.mkdir(exist_ok=True)
                    audio_path.touch()

                    mock_download.return_value = MagicMock(
                        audio_path=audio_path,
                        video_path=None,
                        video_id="test123",
                    )
                    mock_transcribe.side_effect = TranscriptionError("Transcription failed")
                    mock_utils.cleanup_temp_dir = MagicMock()

                    result = process_transcription(
                        youtube_url="https://www.youtube.com/watch?v=test",
                        title="Test",
                        model=mock_whisper_model,
                    )

                    assert result == (None, None, None, None)

    # =========================================================================
    # INPUT TYPE DETECTION TESTS
    # =========================================================================

    def test_detects_local_file(self, mock_dependencies, mock_whisper_model, temp_dir):
        """Test that local file is detected."""
        local_file = temp_dir / "video.mp4"
        local_file.touch()

        with patch("yt_transcriber.service.extract_audio_from_local_file") as mock_extract:
            with patch("yt_transcriber.service.transcribe_audio_file") as mock_transcribe:
                with patch("yt_transcriber.service.utils") as mock_utils:
                    audio_path = temp_dir / "temp" / "audio.wav"
                    audio_path.parent.mkdir(exist_ok=True)
                    audio_path.touch()

                    mock_extract.return_value = MagicMock(
                        audio_path=audio_path,
                        video_path=local_file,
                        video_id="video",
                    )
                    mock_transcribe.return_value = MagicMock(
                        text="Text",
                        language="en",
                    )

                    transcript_path = temp_dir / "transcripts" / "test.txt"
                    transcript_path.parent.mkdir(exist_ok=True)
                    transcript_path.write_text("Text")
                    mock_utils.save_transcription_to_file.return_value = transcript_path
                    mock_utils.normalize_title_for_filename.return_value = "video"
                    mock_utils.cleanup_temp_dir = MagicMock()

                    process_transcription(
                        youtube_url=str(local_file),
                        title="",
                        model=mock_whisper_model,
                        only_transcript=True,
                    )

                    # Should call extract_audio_from_local_file
                    mock_extract.assert_called_once()

    def test_detects_google_drive_url(self, mock_dependencies, mock_whisper_model, temp_dir):
        """Test that Google Drive URL is detected and processed."""
        with patch("yt_transcriber.service.download_and_extract_audio") as mock_download:
            with patch("yt_transcriber.service.transcribe_audio_file") as mock_transcribe:
                with patch("yt_transcriber.service.utils") as mock_utils:
                    audio_path = temp_dir / "temp" / "audio.wav"
                    audio_path.parent.mkdir(exist_ok=True)
                    audio_path.touch()

                    mock_download.return_value = MagicMock(
                        audio_path=audio_path,
                        video_path=None,
                        video_id="drive_abc123",
                    )
                    mock_transcribe.return_value = MagicMock(
                        text="Text",
                        language="en",
                    )

                    transcript_path = temp_dir / "transcripts" / "test.txt"
                    transcript_path.parent.mkdir(exist_ok=True)
                    transcript_path.write_text("Text")
                    mock_utils.save_transcription_to_file.return_value = transcript_path
                    mock_utils.normalize_title_for_filename.return_value = "test"
                    mock_utils.cleanup_temp_dir = MagicMock()

                    # Google Drive URL - detection happens automatically
                    process_transcription(
                        youtube_url="https://drive.google.com/file/d/abc123/view",
                        title="Test",
                        model=mock_whisper_model,
                        only_transcript=True,
                    )

                    mock_download.assert_called_once()

    # =========================================================================
    # SKIP FLAGS TESTS
    # =========================================================================

    def test_skip_summary_flag(self, mock_dependencies, mock_whisper_model, temp_dir):
        """Test skip_summary flag prevents summary generation."""
        with patch("yt_transcriber.service.download_and_extract_audio") as mock_download:
            with patch("yt_transcriber.service.transcribe_audio_file") as mock_transcribe:
                with patch("yt_transcriber.service.utils") as mock_utils:
                    with patch("yt_transcriber.service.generate_summary") as mock_summary:
                        audio_path = temp_dir / "temp" / "audio.wav"
                        audio_path.parent.mkdir(exist_ok=True)
                        audio_path.touch()

                        mock_download.return_value = MagicMock(
                            audio_path=audio_path,
                            video_path=None,
                            video_id="test123",
                        )
                        mock_transcribe.return_value = MagicMock(
                            text="Text",
                            language="en",
                        )

                        transcript_path = temp_dir / "transcripts" / "test.txt"
                        transcript_path.parent.mkdir(exist_ok=True)
                        transcript_path.write_text("Text")
                        mock_utils.save_transcription_to_file.return_value = transcript_path
                        mock_utils.normalize_title_for_filename.return_value = "test"
                        mock_utils.cleanup_temp_dir = MagicMock()

                        process_transcription(
                            youtube_url="https://www.youtube.com/watch?v=test",
                            title="Test",
                            model=mock_whisper_model,
                            skip_summary=True,
                        )

                        # Should not call generate_summary
                        mock_summary.assert_not_called()

    def test_only_transcript_flag(self, mock_dependencies, mock_whisper_model, temp_dir):
        """Test only_transcript flag prevents summary generation."""
        with patch("yt_transcriber.service.download_and_extract_audio") as mock_download:
            with patch("yt_transcriber.service.transcribe_audio_file") as mock_transcribe:
                with patch("yt_transcriber.service.utils") as mock_utils:
                    with patch("yt_transcriber.service.generate_summary") as mock_summary:
                        audio_path = temp_dir / "temp" / "audio.wav"
                        audio_path.parent.mkdir(exist_ok=True)
                        audio_path.touch()

                        mock_download.return_value = MagicMock(
                            audio_path=audio_path,
                            video_path=None,
                            video_id="test123",
                        )
                        mock_transcribe.return_value = MagicMock(
                            text="Text",
                            language="en",
                        )

                        transcript_path = temp_dir / "transcripts" / "test.txt"
                        transcript_path.parent.mkdir(exist_ok=True)
                        transcript_path.write_text("Text")
                        mock_utils.save_transcription_to_file.return_value = transcript_path
                        mock_utils.normalize_title_for_filename.return_value = "test"
                        mock_utils.cleanup_temp_dir = MagicMock()

                        result = process_transcription(
                            youtube_url="https://www.youtube.com/watch?v=test",
                            title="Test",
                            model=mock_whisper_model,
                            only_transcript=True,
                        )

                        mock_summary.assert_not_called()
                        transcript, summary_en, summary_es, post_kits = result
                        assert transcript is not None
                        assert summary_en is None

    # =========================================================================
    # CLEANUP TESTS
    # =========================================================================

    def test_cleanup_called_on_success(self, mock_dependencies, mock_whisper_model, temp_dir):
        """Test that cleanup is called on success."""
        with patch("yt_transcriber.service.download_and_extract_audio") as mock_download:
            with patch("yt_transcriber.service.transcribe_audio_file") as mock_transcribe:
                with patch("yt_transcriber.service.utils") as mock_utils:
                    audio_path = temp_dir / "temp" / "audio.wav"
                    audio_path.parent.mkdir(exist_ok=True)
                    audio_path.touch()

                    mock_download.return_value = MagicMock(
                        audio_path=audio_path,
                        video_path=None,
                        video_id="test123",
                    )
                    mock_transcribe.return_value = MagicMock(
                        text="Text",
                        language="en",
                    )

                    transcript_path = temp_dir / "transcripts" / "test.txt"
                    transcript_path.parent.mkdir(exist_ok=True)
                    transcript_path.write_text("Text")
                    mock_utils.save_transcription_to_file.return_value = transcript_path
                    mock_utils.normalize_title_for_filename.return_value = "test"
                    mock_utils.cleanup_temp_dir = MagicMock()

                    process_transcription(
                        youtube_url="https://www.youtube.com/watch?v=test",
                        title="Test",
                        model=mock_whisper_model,
                        only_transcript=True,
                    )

                    mock_utils.cleanup_temp_dir.assert_called_once()

    def test_cleanup_called_on_error(self, mock_dependencies, mock_whisper_model):
        """Test that cleanup is called even on error."""
        with patch("yt_transcriber.service.download_and_extract_audio") as mock_download:
            with patch("yt_transcriber.service.utils") as mock_utils:
                from core.media_downloader import DownloadError

                mock_download.side_effect = DownloadError("Failed")
                mock_utils.cleanup_temp_dir = MagicMock()

                process_transcription(
                    youtube_url="https://www.youtube.com/watch?v=test",
                    title="Test",
                    model=mock_whisper_model,
                )

                mock_utils.cleanup_temp_dir.assert_called_once()
