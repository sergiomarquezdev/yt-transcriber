"""Shared YouTube downloader and audio extractor.

This module centralizes the download + audio extraction logic used by both the
transcription and script generation pipelines.
"""

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yt_dlp

from core import utils

logger = logging.getLogger(__name__)


def is_google_drive_url(url: str) -> bool:
    """Check if a URL is a Google Drive link.

    Args:
        url: URL to check

    Returns:
        True if the URL is a Google Drive link, False otherwise
    """
    drive_patterns = [
        r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
        r"drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
        r"docs\.google\.com/file/d/([a-zA-Z0-9_-]+)",
    ]
    return any(re.search(pattern, url) for pattern in drive_patterns)


def extract_drive_file_id(url: str) -> str | None:
    """Extract Google Drive file ID from URL.

    Args:
        url: Google Drive URL

    Returns:
        File ID if found, None otherwise
    """
    patterns = [
        r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
        r"drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
        r"docs\.google\.com/file/d/([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


class DownloadError(Exception):
    """Custom exception for download errors."""

    pass


@dataclass
class DownloadResult:
    """Holds the results of a successful download."""

    audio_path: Path
    video_path: Path | None
    video_id: str


def download_and_extract_audio(
    youtube_url: str,
    temp_dir: Path,
    unique_job_id: str,
    ffmpeg_location: str | None = None,
) -> DownloadResult:
    """Download a YouTube video or Google Drive file and extract its audio to WAV.

    Args:
        youtube_url: Full YouTube URL or Google Drive URL
        temp_dir: Directory to store temporary files
        unique_job_id: Unique identifier for this job
        ffmpeg_location: Optional custom path to FFmpeg

    Returns:
        DownloadResult with audio path, optional video path, and video ID

    Raises:
        DownloadError: On any download or extraction failure
    """
    logger.info(f"Starting download for URL: {youtube_url}, job_id: {unique_job_id}")
    utils.ensure_dir_exists(temp_dir)

    # Check if it's a Google Drive URL
    is_drive = is_google_drive_url(youtube_url)
    drive_file_id: str | None = None

    if is_drive:
        logger.info("Detected Google Drive URL, attempting download with yt-dlp...")
        # Try to get file ID for naming
        drive_file_id = extract_drive_file_id(youtube_url)
        if drive_file_id:
            logger.info(f"Extracted Google Drive file ID: {drive_file_id}")

    try:
        # 1) Extract video info and ID
        info_opts = {"quiet": True, "noplaylist": True, "logger": logger}

        with yt_dlp.YoutubeDL(info_opts) as ydl:
            try:
                info_dict = ydl.extract_info(youtube_url, download=False)
                video_id = info_dict.get("id")

                # For Google Drive, if yt-dlp doesn't provide an ID, use the Drive file ID
                if not video_id and is_drive and drive_file_id:
                    video_id = f"drive_{drive_file_id}"
                elif not video_id:
                    # Fallback: try to extract from title or use a hash
                    title = info_dict.get("title", "")
                    if title:
                        video_id = utils.normalize_title_for_filename(title)[:50]
                    elif drive_file_id:
                        video_id = f"drive_{drive_file_id}"
                    else:
                        video_id = f"unknown_{unique_job_id}"

                if not video_id:
                    raise DownloadError(f"Could not extract video ID from URL: {youtube_url}")
            except yt_dlp.utils.DownloadError as e:
                if is_drive:
                    # For Drive, provide more helpful error message
                    error_msg = (
                        f"Could not access Google Drive file: {e}\n\n"
                        "Suggestions:\n"
                        "1. Make sure the file is shared with download permissions\n"
                        "2. Verify the link is correct\n"
                        "3. If the file has restrictions, download it manually and use the local path"
                    )
                    raise DownloadError(error_msg) from e
                raise
        logger.info(f"Extracted video ID: {video_id}")

        # 2) Configure predictable filenames
        base_filename = f"{video_id}_{unique_job_id}"
        output_template = temp_dir / f"{base_filename}.%(ext)s"
        expected_audio_path = temp_dir / f"{base_filename}.wav"

        ydl_opts: dict = {
            "format": "bestaudio/best",
            "quiet": False,
            "noplaylist": True,
            "keepvideo": True,
            "outtmpl": str(output_template),
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "nopostoverwrites": False,
                }
            ],
            "postprocessor_args": {"FFmpegExtractAudio": ["-ar", "16000", "-ac", "1"]},
            "logger": logger,
        }

        if ffmpeg_location:
            ydl_opts["ffmpeg_location"] = ffmpeg_location
            logger.info(f"Using FFmpeg at: {ffmpeg_location}")

        # 3) Download + post-process
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            download_info = ydl.extract_info(youtube_url, download=True)
            video_path_str = ydl.prepare_filename(download_info)
            video_path = (
                Path(video_path_str) if video_path_str and Path(video_path_str).exists() else None
            )

        # 4) Verify audio exists
        if not expected_audio_path.exists():
            logger.error(
                f"Critical failure: WAV audio not found at expected path: {expected_audio_path}"
            )
            if video_path:
                try:
                    video_path.unlink()
                except OSError as e_clean:
                    logger.error(
                        f"Error cleaning video '{video_path}' after audio failure: {e_clean}"
                    )
            raise DownloadError(f"Audio extraction failed for video ID {video_id}.")

        logger.info(f"Audio extracted to: {expected_audio_path}")
        if video_path:
            logger.info(f"Video downloaded to: {video_path}")

        return DownloadResult(
            audio_path=expected_audio_path,
            video_path=video_path,
            video_id=video_id,
        )

    except DownloadError:
        # Re-raise our own DownloadError without wrapping
        raise
    except yt_dlp.utils.DownloadError as e_yt:
        logger.error(f"yt-dlp error for '{youtube_url}': {e_yt}")
        raise DownloadError(f"yt-dlp failed: {e_yt}") from e_yt
    except Exception as e_gen:
        logger.error(f"Unexpected error downloading '{youtube_url}': {e_gen}", exc_info=True)
        raise DownloadError(f"General download error: {e_gen}") from e_gen


def extract_audio_from_local_file(
    video_path: Path,
    temp_dir: Path,
    unique_job_id: str,
    ffmpeg_location: str | None = None,
) -> DownloadResult:
    """Extract audio from a local video file using FFmpeg.

    Args:
        video_path: Path to local video file
        temp_dir: Directory to store temporary files
        unique_job_id: Unique identifier for this job
        ffmpeg_location: Optional custom path to FFmpeg

    Returns:
        DownloadResult with audio path, video path, and a generated video_id

    Raises:
        DownloadError: On any extraction failure
    """
    logger.info(f"Starting audio extraction for local file: {video_path}, job_id: {unique_job_id}")

    if not video_path.exists():
        raise DownloadError(f"Video file does not exist: {video_path}")

    utils.ensure_dir_exists(temp_dir)

    # Generate a video_id from the file name (sanitized)
    video_id = utils.normalize_title_for_filename(video_path.stem)[:50]  # Limit length
    if not video_id:
        video_id = f"local_{unique_job_id}"

    # Output audio path
    base_filename = f"{video_id}_{unique_job_id}"
    expected_audio_path = temp_dir / f"{base_filename}.wav"

    try:
        # Find FFmpeg executable
        ffmpeg_cmd = ffmpeg_location or shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
        if not ffmpeg_cmd:
            raise DownloadError(
                "FFmpeg not found. Please install FFmpeg or provide the path with --ffmpeg-location"
            )

        logger.info(f"Using FFmpeg at: {ffmpeg_cmd}")

        # Extract audio using FFmpeg
        # -i: input file
        # -ar 16000: sample rate 16kHz (Whisper standard)
        # -ac 1: mono channel
        # -y: overwrite output file
        cmd = [
            ffmpeg_cmd,
            "-i",
            str(video_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-y",  # Overwrite if exists
            str(expected_audio_path),
        ]

        logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300,  # 5 minutes timeout to prevent hangs
        )

        # Verify audio file was created
        if not expected_audio_path.exists():
            logger.error(f"FFmpeg did not generate expected audio file: {expected_audio_path}")
            if result.stderr:
                logger.error(f"FFmpeg stderr: {result.stderr}")
            raise DownloadError(f"Audio extraction failed for file: {video_path}")

        logger.info(f"Audio extracted successfully to: {expected_audio_path}")

        return DownloadResult(
            audio_path=expected_audio_path,
            video_path=video_path,
            video_id=video_id,
        )

    except subprocess.TimeoutExpired as e:
        logger.error(f"FFmpeg timed out after {e.timeout} seconds.")
        raise DownloadError(f"FFmpeg timed out (timeout={e.timeout}s).") from e
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running FFmpeg: {e}")
        if e.stderr:
            logger.error(f"FFmpeg stderr: {e.stderr}")
        raise DownloadError(
            f"FFmpeg failed to extract audio: {e.stderr if e.stderr else str(e)}"
        ) from e
    except Exception as e_gen:
        logger.error(
            f"Unexpected error extracting audio from '{video_path}': {e_gen}", exc_info=True
        )
        raise DownloadError(f"General audio extraction error: {e_gen}") from e_gen
