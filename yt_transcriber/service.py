"""Transcriber Service - shared orchestration for the transcription pipeline.

This module centralizes the end-to-end flow used by both CLI and TUI:
Download -> Transcribe -> Save (transcript + optional segments + optional frames)

CLI keeps a thin wrapper that forwards to this service for testability and
backward compatibility (tests patch functions in yt_transcriber.cli).
"""

import logging
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from core.media_downloader import (
    DownloadError,
    download_and_extract_audio,
    extract_audio_from_local_file,
)
from core.media_transcriber import TranscriptionError, transcribe_audio_file
from core.settings import settings
from yt_transcriber import utils

logger = logging.getLogger(__name__)


def _resolve_segments_and_visual(
    segments_override: bool | None,
    visual_override: bool | None,
) -> tuple[bool, bool]:
    """Resolve effective segment/visual toggles with CLI/env precedence."""
    visual_enabled = (
        settings.VISUAL_EVIDENCE_ENABLED if visual_override is None else visual_override
    )

    if segments_override is not None:
        segments_enabled = segments_override
    elif visual_override is True:
        segments_enabled = True
    else:
        segments_enabled = settings.TRANSCRIPT_SEGMENTS_ENABLED

    return segments_enabled, visual_enabled


def _extract_visual_evidence(
    video_path: Path,
    segments: list[Any] | None,
    output_filename_base: str,
    output_dir: Path,
    ffmpeg_location: str | None,
) -> list[Path]:
    """Extract one frame (midpoint) per eligible transcript segment."""
    if not segments:
        logger.debug("No segments available for visual evidence extraction")
        return []

    min_duration = settings.VISUAL_EVIDENCE_MIN_SEGMENT_SECONDS
    ffmpeg_bin = ffmpeg_location or "ffmpeg"
    extracted_paths: list[Path] = []

    for idx, segment in enumerate(segments):
        duration = float(segment.end) - float(segment.start)
        if duration < min_duration:
            logger.debug(
                "Skipping visual evidence for segment %s: duration %.3fs < %.3fs",
                idx,
                duration,
                min_duration,
            )
            continue

        midpoint = float(segment.start) + (duration / 2.0)
        frame_path = output_dir / f"{output_filename_base}_frame_{idx}.jpg"
        cmd = [
            ffmpeg_bin,
            "-y",
            "-ss",
            f"{midpoint:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(frame_path),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            extracted_paths.append(frame_path)
        except Exception as e:
            logger.warning(
                "Could not extract visual evidence frame for segment %s (continuing): %s",
                idx,
                e,
            )

    return extracted_paths


def process_transcription(
    youtube_url: str,
    title: str,
    model: Any,
    language: str | None = None,
    ffmpeg_location: str | None = None,
    segments_override: bool | None = None,
    visual_override: bool | None = None,
) -> Path | None:
    """Download/extract -> transcribe -> save into a per-video output folder.

    Args:
        youtube_url: YouTube URL, Google Drive URL, or local file path.
        title: Video title (inferred from filename for local files when empty).
        model: Preloaded Whisper model.
        language: Optional language code; None = auto-detect.
        ffmpeg_location: Optional FFmpeg path.
        segments_override: Optional CLI override for segments JSON sidecar.
        visual_override: Optional CLI override for visual evidence extraction.

    Returns:
        Path to the saved transcript .txt, or None on failure.
    """
    segments_enabled, visual_enabled = _resolve_segments_and_visual(
        segments_override=segments_override,
        visual_override=visual_override,
    )

    # Detect input type
    is_local_file = False
    local_file_path: Path | None = None
    is_drive_url = False

    if not (youtube_url.startswith("http://") or youtube_url.startswith("https://")):
        potential_path = Path(youtube_url)
        if potential_path.exists() and potential_path.is_file():
            is_local_file = True
            local_file_path = potential_path
            logger.info(f"Detected local file: {local_file_path}")
            if not title or not title.strip():
                title = local_file_path.stem
    else:
        from core.media_downloader import is_google_drive_url

        is_drive_url = is_google_drive_url(youtube_url)
        if is_drive_url:
            logger.info(f"Detected Google Drive URL: {youtube_url}")

    if is_local_file:
        logger.info(f"Starting transcription for local file: {local_file_path}")
    elif is_drive_url:
        logger.info(f"Starting transcription for Google Drive file: {youtube_url}")
    else:
        logger.info(f"Starting transcription for URL: {youtube_url}")

    unique_job_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    settings.TEMP_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory(
            dir=settings.TEMP_DOWNLOAD_DIR, prefix=f"{unique_job_id}_"
        ) as temp_dir_str:
            job_temp_dir = Path(temp_dir_str)
            logger.info(f"Using temp dir: {job_temp_dir}")

            # 1) Download / extract audio
            if is_local_file and local_file_path:
                logger.info("Step 1: Extracting audio from local file...")
                download_result = extract_audio_from_local_file(
                    video_path=local_file_path,
                    temp_dir=job_temp_dir,
                    unique_job_id=unique_job_id,
                    ffmpeg_location=ffmpeg_location,
                )
            else:
                logger.info("Step 1: Downloading and extracting audio...")
                download_result = download_and_extract_audio(
                    youtube_url=youtube_url,
                    temp_dir=job_temp_dir,
                    unique_job_id=unique_job_id,
                    ffmpeg_location=ffmpeg_location,
                )
            logger.info(f"Audio extracted to: {download_result.audio_path}")

            # 2) Transcribe
            logger.info("Step 2: Transcribing audio...")
            transcription_result = transcribe_audio_file(
                audio_path=download_result.audio_path, model=model, language=language
            )
            logger.info(
                f"Transcription complete. Detected language: {transcription_result.language}"
            )

            # 3) Build per-video output dir
            normalized_title = utils.normalize_title_for_filename(title)
            video_id = download_result.video_id or "local"
            output_filename_base = f"{normalized_title}_vid_{video_id}_job_{unique_job_id}"
            output_dir = settings.OUTPUT_BASE_DIR / output_filename_base
            output_dir.mkdir(parents=True, exist_ok=True)

            # 4) Save transcript
            logger.info("Step 3: Saving transcript...")
            transcript_path = utils.save_transcription_to_file(
                transcription_text=transcription_result.text,
                output_filename_no_ext=output_filename_base,
                output_dir=output_dir,
                original_title=title,
            )

            if not transcript_path:
                raise OSError("Could not save transcript file.")

            logger.info(f"Transcript saved successfully to: {transcript_path}")
            print(f"\nTranscript saved to: {transcript_path}")

            # 5) Optional segments JSON
            if segments_enabled:
                segments_path = utils.derive_sibling_path(transcript_path, "_segments.json")
                utils.save_segments_json(
                    segments=transcription_result.segments,
                    language=transcription_result.language,
                    output_path=segments_path,
                )

            # 6) Optional visual evidence (local files only)
            if visual_enabled:
                if not is_local_file or not local_file_path:
                    logger.warning(
                        "Visual evidence is only supported for local files in V1; skipping extraction."
                    )
                else:
                    _extract_visual_evidence(
                        video_path=local_file_path,
                        segments=transcription_result.segments,
                        output_filename_base=output_filename_base,
                        output_dir=output_dir,
                        ffmpeg_location=ffmpeg_location,
                    )

            return transcript_path

    except (OSError, DownloadError, TranscriptionError) as e:
        logger.error(f"An error occurred during processing: {e}", exc_info=True)
        print(f"\nError: {e}", file=sys.stderr)
        return None
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        return None
