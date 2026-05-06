"""Tests for yt_transcriber.service module."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from core.models import TranscriptSegment
from yt_transcriber.service import process_transcription


class TestProcessTranscription:
    """Tests for process_transcription function."""

    @pytest.fixture
    def mock_dependencies(self, temp_dir):
        """Set up common mocks for process_transcription."""
        with patch("yt_transcriber.service.settings") as mock_settings:
            mock_settings.TEMP_DOWNLOAD_DIR = temp_dir / "temp"
            mock_settings.OUTPUT_BASE_DIR = temp_dir / "output"
            mock_settings.TRANSCRIPT_SEGMENTS_ENABLED = False
            mock_settings.VISUAL_EVIDENCE_ENABLED = False
            mock_settings.VISUAL_EVIDENCE_MIN_SEGMENT_SECONDS = 1.0

            (temp_dir / "temp").mkdir(exist_ok=True)
            (temp_dir / "output").mkdir(exist_ok=True)

            yield mock_settings

    @pytest.fixture
    def mock_whisper_model(self):
        """Create mock faster-whisper model."""
        model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "This is the transcribed text."
        mock_info = MagicMock()
        mock_info.language = "en"
        model.transcribe.return_value = ([mock_segment], mock_info)
        return model

    # =========================================================================
    # BASIC FUNCTIONALITY TESTS
    # =========================================================================

    def test_successful_transcription_only(self, mock_dependencies, mock_whisper_model, temp_dir):
        """Test successful transcription without summary (default behavior)."""
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

                    transcript_path = temp_dir / "output" / "test_video_vid_test123_job_1" / "test_video_vid_test123_job_1.txt"
                    transcript_path.parent.mkdir(parents=True, exist_ok=True)
                    transcript_path.write_text("Transcribed text")
                    mock_utils.save_transcription_to_file.return_value = transcript_path
                    mock_utils.normalize_title_for_filename.return_value = "test_video"

                    result = process_transcription(
                        youtube_url="https://www.youtube.com/watch?v=test123",
                        title="Test Video",
                        model=mock_whisper_model,
                    )

                    assert result is not None
                    assert result == transcript_path

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

                assert result is None

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

                    assert result is None

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

                    transcript_path = temp_dir / "output" / "video_vid_video_job_1" / "video_vid_video_job_1.txt"
                    transcript_path.parent.mkdir(parents=True, exist_ok=True)
                    transcript_path.write_text("Text")
                    mock_utils.save_transcription_to_file.return_value = transcript_path
                    mock_utils.normalize_title_for_filename.return_value = "video"

                    process_transcription(
                        youtube_url=str(local_file),
                        title="",
                        model=mock_whisper_model,
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

                    transcript_path = temp_dir / "output" / "test_vid_drive_abc123_job_1" / "test_vid_drive_abc123_job_1.txt"
                    transcript_path.parent.mkdir(parents=True, exist_ok=True)
                    transcript_path.write_text("Text")
                    mock_utils.save_transcription_to_file.return_value = transcript_path
                    mock_utils.normalize_title_for_filename.return_value = "test"

                    process_transcription(
                        youtube_url="https://drive.google.com/file/d/abc123/view",
                        title="Test",
                        model=mock_whisper_model,
                    )

                    mock_download.assert_called_once()

    # =========================================================================
    # SEGMENT / VISUAL TOGGLE TESTS
    # =========================================================================

    def test_resolve_segments_and_visual_uses_env_defaults(self, mock_dependencies):
        """When CLI overrides are omitted, env defaults are used."""
        from yt_transcriber import service as svc

        mock_dependencies.TRANSCRIPT_SEGMENTS_ENABLED = True
        mock_dependencies.VISUAL_EVIDENCE_ENABLED = False

        segments_enabled, visual_enabled = svc._resolve_segments_and_visual(
            segments_override=None,
            visual_override=None,
        )

        assert segments_enabled is True
        assert visual_enabled is False

    def test_resolve_visual_implies_segments_when_no_segments_override(self, mock_dependencies):
        """Visual CLI enablement implies segments when segments flag is omitted."""
        from yt_transcriber import service as svc

        mock_dependencies.TRANSCRIPT_SEGMENTS_ENABLED = False
        mock_dependencies.VISUAL_EVIDENCE_ENABLED = False

        segments_enabled, visual_enabled = svc._resolve_segments_and_visual(
            segments_override=None,
            visual_override=True,
        )

        assert segments_enabled is True
        assert visual_enabled is True

    def test_resolve_explicit_no_segments_beats_visual_implication(self, mock_dependencies):
        """Explicit --no-segments keeps segments disabled even when visual is enabled."""
        from yt_transcriber import service as svc

        mock_dependencies.TRANSCRIPT_SEGMENTS_ENABLED = True
        mock_dependencies.VISUAL_EVIDENCE_ENABLED = False

        segments_enabled, visual_enabled = svc._resolve_segments_and_visual(
            segments_override=False,
            visual_override=True,
        )

        assert segments_enabled is False
        assert visual_enabled is True

    def test_segments_json_written_when_enabled(
        self, mock_dependencies, mock_whisper_model, temp_dir
    ):
        """Segments sidecar is written when effective setting is enabled."""
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
                    segments = [TranscriptSegment(start=0.0, end=2.0, text="hello")]
                    mock_transcribe.return_value = MagicMock(
                        text="Text",
                        language="en",
                        segments=segments,
                    )

                    transcript_path = temp_dir / "output" / "test_vid_test123_job_1" / "test_vid_test123_job_1.txt"
                    transcript_path.parent.mkdir(parents=True, exist_ok=True)
                    transcript_path.write_text("Text")
                    segments_path = transcript_path.parent / "test_segments.json"

                    mock_utils.save_transcription_to_file.return_value = transcript_path
                    mock_utils.normalize_title_for_filename.return_value = "test"
                    mock_utils.derive_sibling_path.return_value = segments_path

                    process_transcription(
                        youtube_url="https://www.youtube.com/watch?v=test",
                        title="Test",
                        model=mock_whisper_model,
                        segments_override=True,
                    )

                    mock_utils.save_segments_json.assert_called_once_with(
                        segments=segments,
                        language="en",
                        output_path=segments_path,
                    )

    def test_segments_json_skipped_when_disabled(
        self, mock_dependencies, mock_whisper_model, temp_dir
    ):
        """Segments sidecar is skipped when setting remains disabled."""
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
                        segments=[TranscriptSegment(start=0.0, end=2.0, text="hello")],
                    )

                    transcript_path = temp_dir / "output" / "test_vid_test123_job_1" / "test_vid_test123_job_1.txt"
                    transcript_path.parent.mkdir(parents=True, exist_ok=True)
                    transcript_path.write_text("Text")
                    mock_utils.save_transcription_to_file.return_value = transcript_path
                    mock_utils.normalize_title_for_filename.return_value = "test"

                    process_transcription(
                        youtube_url="https://www.youtube.com/watch?v=test",
                        title="Test",
                        model=mock_whisper_model,
                    )

                    mock_utils.save_segments_json.assert_not_called()

    def test_segments_json_written_when_enabled_by_env(
        self, mock_dependencies, mock_whisper_model, temp_dir
    ):
        """Env toggle enables segments sidecar when CLI override is omitted."""
        mock_dependencies.TRANSCRIPT_SEGMENTS_ENABLED = True

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
                    segments = [TranscriptSegment(start=0.0, end=2.0, text="hello")]
                    mock_transcribe.return_value = MagicMock(
                        text="Text",
                        language="en",
                        segments=segments,
                    )

                    transcript_path = temp_dir / "output" / "test_vid_test123_job_1" / "test_vid_test123_job_1.txt"
                    transcript_path.parent.mkdir(parents=True, exist_ok=True)
                    transcript_path.write_text("Text")
                    segments_path = transcript_path.parent / "test_segments.json"

                    mock_utils.save_transcription_to_file.return_value = transcript_path
                    mock_utils.normalize_title_for_filename.return_value = "test"
                    mock_utils.derive_sibling_path.return_value = segments_path

                    process_transcription(
                        youtube_url="https://www.youtube.com/watch?v=test",
                        title="Test",
                        model=mock_whisper_model,
                    )

                    mock_utils.save_segments_json.assert_called_once_with(
                        segments=segments,
                        language="en",
                        output_path=segments_path,
                    )

    def test_visual_evidence_skipped_for_url(self, mock_dependencies, mock_whisper_model, temp_dir):
        """URL inputs skip frames with warning but still emit segments via visual implication."""
        with patch("yt_transcriber.service.download_and_extract_audio") as mock_download:
            with patch("yt_transcriber.service.transcribe_audio_file") as mock_transcribe:
                with patch("yt_transcriber.service.utils") as mock_utils:
                    with patch("yt_transcriber.service._extract_visual_evidence") as mock_extract:
                        with patch("yt_transcriber.service.logger") as mock_logger:
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
                                segments=[TranscriptSegment(start=0.0, end=2.0, text="hello")],
                            )

                            transcript_path = temp_dir / "output" / "test_vid_test123_job_1" / "test_vid_test123_job_1.txt"
                            transcript_path.parent.mkdir(parents=True, exist_ok=True)
                            transcript_path.write_text("Text")
                            segments_path = transcript_path.parent / "test_segments.json"
                            mock_utils.save_transcription_to_file.return_value = transcript_path
                            mock_utils.normalize_title_for_filename.return_value = "test"
                            mock_utils.derive_sibling_path.return_value = segments_path

                            process_transcription(
                                youtube_url="https://www.youtube.com/watch?v=test",
                                title="Test",
                                model=mock_whisper_model,
                                visual_override=True,
                            )

                            mock_extract.assert_not_called()
                            mock_utils.save_segments_json.assert_called_once_with(
                                segments=[TranscriptSegment(start=0.0, end=2.0, text="hello")],
                                language="en",
                                output_path=segments_path,
                            )
                            mock_logger.warning.assert_called_once()

    def test_visual_evidence_local_file_calls_extractor(
        self, mock_dependencies, mock_whisper_model, temp_dir
    ):
        """Local-file flow runs visual extraction when visual evidence is enabled."""
        local_file = temp_dir / "video.mp4"
        local_file.touch()

        with patch("yt_transcriber.service.extract_audio_from_local_file") as mock_extract_audio:
            with patch("yt_transcriber.service.transcribe_audio_file") as mock_transcribe:
                with patch("yt_transcriber.service.utils") as mock_utils:
                    with patch("yt_transcriber.service._extract_visual_evidence") as mock_extract:
                        audio_path = temp_dir / "temp" / "audio.wav"
                        audio_path.parent.mkdir(exist_ok=True)
                        audio_path.touch()

                        mock_extract_audio.return_value = MagicMock(
                            audio_path=audio_path,
                            video_path=local_file,
                            video_id="video",
                        )
                        segments = [TranscriptSegment(start=10.0, end=15.0, text="segment")]
                        mock_transcribe.return_value = MagicMock(
                            text="Text",
                            language="en",
                            segments=segments,
                        )

                        output_stem = "video_vid_video_job_1"
                        transcript_path = temp_dir / "output" / output_stem / f"{output_stem}.txt"
                        transcript_path.parent.mkdir(parents=True, exist_ok=True)
                        transcript_path.write_text("Text")
                        mock_utils.save_transcription_to_file.return_value = transcript_path
                        mock_utils.normalize_title_for_filename.return_value = "video"
                        mock_utils.derive_sibling_path.return_value = (
                            transcript_path.parent / f"{output_stem}_segments.json"
                        )

                        process_transcription(
                            youtube_url=str(local_file),
                            title="",
                            model=mock_whisper_model,
                            visual_override=True,
                        )

                        mock_extract.assert_called_once()
                        kwargs = mock_extract.call_args.kwargs
                        assert kwargs["video_path"] == local_file
                        assert kwargs["segments"] == segments
                        # output_dir is the per-video subdir under OUTPUT_BASE_DIR
                        assert kwargs["output_dir"].parent == mock_dependencies.OUTPUT_BASE_DIR

    def test_visual_evidence_ffmpeg_failure_non_fatal_in_process_flow(
        self, mock_dependencies, mock_whisper_model, temp_dir
    ):
        """Local process flow remains successful when ffmpeg fails for frame extraction."""
        local_file = temp_dir / "video.mp4"
        local_file.touch()

        with patch("yt_transcriber.service.extract_audio_from_local_file") as mock_extract_audio:
            with patch("yt_transcriber.service.transcribe_audio_file") as mock_transcribe:
                with patch("yt_transcriber.service.utils") as mock_utils:
                    with patch("yt_transcriber.service.subprocess.run") as mock_run:
                        audio_path = temp_dir / "temp" / "audio.wav"
                        audio_path.parent.mkdir(exist_ok=True)
                        audio_path.touch()

                        mock_extract_audio.return_value = MagicMock(
                            audio_path=audio_path,
                            video_path=local_file,
                            video_id="video",
                        )
                        segments = [TranscriptSegment(start=10.0, end=15.0, text="segment")]
                        mock_transcribe.return_value = MagicMock(
                            text="Text",
                            language="en",
                            segments=segments,
                        )

                        output_stem = "video_vid_video_job_1"
                        transcript_path = temp_dir / "output" / output_stem / f"{output_stem}.txt"
                        transcript_path.parent.mkdir(parents=True, exist_ok=True)
                        transcript_path.write_text("Text")
                        segments_path = transcript_path.parent / f"{output_stem}_segments.json"

                        mock_utils.save_transcription_to_file.return_value = transcript_path
                        mock_utils.normalize_title_for_filename.return_value = "video"
                        mock_utils.derive_sibling_path.return_value = segments_path
                        mock_run.side_effect = subprocess.CalledProcessError(1, ["ffmpeg"])

                        result = process_transcription(
                            youtube_url=str(local_file),
                            title="",
                            model=mock_whisper_model,
                            visual_override=True,
                        )

                        assert result == transcript_path
                        mock_utils.save_segments_json.assert_called_once_with(
                            segments=segments,
                            language="en",
                            output_path=segments_path,
                        )
                        mock_run.assert_called_once()

    def test_visual_evidence_extracts_midpoint_single_frame(self, mock_dependencies, temp_dir):
        """Extractor uses midpoint timestamp and emits one frame per eligible segment."""
        from yt_transcriber import service as svc

        segment = TranscriptSegment(start=10.0, end=15.0, text="midpoint")
        with patch("yt_transcriber.service.subprocess.run") as mock_run:
            result = svc._extract_visual_evidence(
                video_path=temp_dir / "video.mp4",
                segments=[segment],
                output_filename_base="test_video",
                output_dir=temp_dir,
                ffmpeg_location=None,
            )

            expected_frame = temp_dir / "test_video_frame_0.jpg"
            assert result == [expected_frame]
            mock_run.assert_called_once()

            cmd = mock_run.call_args.args[0]
            assert "-ss" in cmd
            assert cmd[cmd.index("-ss") + 1] == "12.500"
            assert cmd[-1] == str(expected_frame)

    def test_visual_evidence_warns_for_short_segment(self, mock_dependencies, temp_dir):
        """Very short segments are skipped and ffmpeg is not called."""
        from yt_transcriber import service as svc

        short_segment = TranscriptSegment(start=0.0, end=0.4, text="short")
        with patch("yt_transcriber.service.subprocess.run") as mock_run:
            with patch("yt_transcriber.service.logger") as mock_logger:
                svc._extract_visual_evidence(
                    video_path=temp_dir / "video.mp4",
                    segments=[short_segment],
                    output_filename_base="test_video",
                    output_dir=temp_dir,
                    ffmpeg_location=None,
                )

                mock_run.assert_not_called()
                mock_logger.debug.assert_called()

    def test_visual_evidence_failure_non_fatal(self, mock_dependencies, temp_dir):
        """ffmpeg failures are logged and do not propagate from extractor."""
        from yt_transcriber import service as svc

        segment = TranscriptSegment(start=10.0, end=12.0, text="long enough")
        with patch("yt_transcriber.service.subprocess.run") as mock_run:
            with patch("yt_transcriber.service.logger") as mock_logger:
                mock_run.side_effect = subprocess.CalledProcessError(1, ["ffmpeg"])

                result = svc._extract_visual_evidence(
                    video_path=temp_dir / "video.mp4",
                    segments=[segment],
                    output_filename_base="test_video",
                    output_dir=temp_dir,
                    ffmpeg_location=None,
                )

                assert result == []
                mock_logger.warning.assert_called()

    def test_transcript_lives_in_per_video_subdir(
        self, mock_dependencies, mock_whisper_model, temp_dir
    ):
        """The transcript .txt is written under output/<stem>/<stem>.txt — never directly under output/."""
        with patch("yt_transcriber.service.download_and_extract_audio") as mock_dl, patch(
            "yt_transcriber.service.transcribe_audio_file"
        ) as mock_tr:
            mock_dl.return_value = MagicMock(
                audio_path=temp_dir / "fake.wav",
                video_path=None,
                video_id="ABC123",
            )
            mock_tr.return_value = MagicMock(
                text="hello world",
                language="en",
                segments=[],
            )

            result = process_transcription(
                youtube_url="https://www.youtube.com/watch?v=ABC123",
                title="Example Video",
                model=mock_whisper_model,
            )

            assert result is not None, "expected a Path, got None"
            assert result.suffix == ".txt"
            # The transcript must sit one level deep: output/<stem>/<stem>.txt
            assert result.parent.parent == mock_dependencies.OUTPUT_BASE_DIR
            assert result.parent.name == result.stem
            assert "ABC123" in result.stem

    @pytest.mark.skip(
        reason="Cleanup now handled by TemporaryDirectory context manager (Issue #12)"
    )
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

                    transcript_path = temp_dir / "output" / "test_vid_test123_job_1" / "test_vid_test123_job_1.txt"
                    transcript_path.parent.mkdir(parents=True, exist_ok=True)
                    transcript_path.write_text("Text")
                    mock_utils.save_transcription_to_file.return_value = transcript_path
                    mock_utils.normalize_title_for_filename.return_value = "test"
                    mock_utils.cleanup_temp_dir = MagicMock()

                    process_transcription(
                        youtube_url="https://www.youtube.com/watch?v=test",
                        title="Test",
                        model=mock_whisper_model,
                    )

                    mock_utils.cleanup_temp_dir.assert_called_once()

    @pytest.mark.skip(
        reason="Cleanup now handled by TemporaryDirectory context manager (Issue #12)"
    )
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
