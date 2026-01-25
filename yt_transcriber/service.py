"""Transcriber Service - shared orchestration for download/transcribe/summarize.

This module centralizes the end-to-end flow used by both CLI and Web UI:
Download -> Transcribe -> (Summarize EN) -> (Translate ES) -> (Post Kits)

CLI keeps a thin wrapper that forwards to this service for testability and
backward compatibility (tests patch functions in yt_transcriber.cli).
"""

import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

# Prefer core media for new code paths
from core.llm import is_model_configured
from core.media_downloader import (
    DownloadError,
    download_and_extract_audio,
    extract_audio_from_local_file,
)
from core.media_transcriber import TranscriptionError, transcribe_audio_file
from core.settings import settings
from core.translator import ScriptTranslator
from yt_transcriber import utils
from yt_transcriber.summarizer import generate_summary as create_summary

if TYPE_CHECKING:
    import whisper

    from core.models import VideoSummary

logger = logging.getLogger(__name__)


def _generate_summary_outputs(
    transcript_text: str,
    video_title: str,
    video_url: str,
    video_id: str,
    output_filename_base: str,
    generate_post_kits: bool,
    is_local_file: bool,
    source_label: str = "",
) -> tuple[Path | None, Path | None, Path | None]:
    """Generate EN summary, ES translation, and optionally post kits.

    Args:
        transcript_text: Full transcript text
        video_title: Video title
        video_url: Original URL (empty for local files)
        video_id: Video ID
        output_filename_base: Base filename for outputs
        generate_post_kits: Whether to generate LinkedIn/Twitter posts
        is_local_file: Whether source is a local file
        source_label: Label for log messages (e.g., "cache" or empty)

    Returns:
        Tuple of (summary_path_en, summary_path_es, post_kits_path)
    """
    label = f" ({source_label})" if source_label else ""

    # Check if summarizer model is configured
    ok, reason = is_model_configured(settings.SUMMARIZER_MODEL)
    if not ok:
        logger.warning(f"{reason}. Skipping summary and Post Kits.")
        print(f"\nWarning: {reason}. Skipping summary and Post Kits.")
        return None, None, None

    # Generate EN summary
    video_url_for_summary = video_url if not is_local_file else ""
    summary_en: VideoSummary = create_summary(
        transcript=transcript_text,
        video_title=video_title,
        video_url=video_url_for_summary,
        video_id=video_id,
    )

    settings.SUMMARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_filename_en = f"{output_filename_base}_summary_EN.md"
    summary_path_en = settings.SUMMARY_OUTPUT_DIR / summary_filename_en
    summary_path_en.write_text(summary_en.to_markdown(), encoding="utf-8")

    logger.info(f"Summary EN{label} saved successfully to: {summary_path_en}")
    print(f"Summary (EN) saved to: {summary_path_en}")

    # Translate to ES
    summary_path_es: Path | None = None
    try:
        translator = ScriptTranslator(use_translation_model=True)
        summary_es = translator.translate_summary(summary_en)

        summary_filename_es = f"{output_filename_base}_summary_ES.md"
        summary_path_es = settings.SUMMARY_OUTPUT_DIR / summary_filename_es
        summary_path_es.write_text(summary_es.to_markdown(), encoding="utf-8")

        logger.info(f"Summary ES{label} saved successfully to: {summary_path_es}")
        print(f"Summary (ES) saved to: {summary_path_es}")
    except Exception as e:
        logger.warning(f"Could not translate summary{label} (continuing): {e}")
        print(f"\nWarning: Could not translate summary: {e}", file=sys.stderr)
        return summary_path_en, None, None

    # Generate Post Kits if requested
    post_kits_path: Path | None = None
    if generate_post_kits:
        ok_pk, reason_pk = is_model_configured(settings.SUMMARIZER_MODEL)
        if not ok_pk:
            logger.warning(f"No LLM configured for Post Kits: {reason_pk}.")
        else:
            try:
                from yt_transcriber.post_kits_generator import (
                    generate_post_kits as gen_kits,
                )

                video_url_for_kits = video_url if not is_local_file else ""
                post_kits = gen_kits(
                    summary=summary_en,
                    video_title=video_title,
                    video_url=video_url_for_kits,
                )
                post_kits_filename = f"{output_filename_base}_post_kits.md"
                post_kits_path = settings.SUMMARY_OUTPUT_DIR / post_kits_filename
                post_kits_path.write_text(post_kits.to_markdown(), encoding="utf-8")

                logger.info(f"Post Kits{label} saved successfully to: {post_kits_path}")
                print(f"Post Kits saved to: {post_kits_path}")
            except Exception as e:
                logger.warning(f"Could not generate Post Kits{label} (continuing): {e}")
                print(
                    f"\nWarning: Could not generate Post Kits: {e}",
                    file=sys.stderr,
                )

    return summary_path_en, summary_path_es, post_kits_path


def process_transcription(
    youtube_url: str,
    title: str,
    model: "whisper.Whisper",
    language: str | None = None,
    ffmpeg_location: str | None = None,
    generate_post_kits: bool = False,
    generate_summary: bool = False,
    reuse_transcripts: bool = False,
) -> tuple[Path | None, Path | None, Path | None, Path | None]:
    """Main logic to download, transcribe, and optionally summarize + post kits.

    Args:
        youtube_url: YouTube URL or path to local video file
        title: Video title (if empty, will be inferred from file name for local files)
        model: Preloaded Whisper model
        language: Optional language code
        ffmpeg_location: Optional FFmpeg path
        generate_post_kits: Generate LinkedIn + Twitter posts (implies generate_summary)
        generate_summary: Generate EN + ES summaries (default: False, only transcript)
        reuse_transcripts: Reuse cached transcripts if available

    Returns:
        Tuple of (transcript_path, summary_path_en, summary_path_es, post_kits_path)
        or (None, None, None, None) on failure.
    """
    # Post kits requires summary, so enable it implicitly
    if generate_post_kits:
        generate_summary = True
    # Detect if input is a local file, Google Drive URL, or YouTube URL
    is_local_file = False
    local_file_path: Path | None = None
    is_drive_url = False

    # Check if it's a local file path
    if not (youtube_url.startswith("http://") or youtube_url.startswith("https://")):
        potential_path = Path(youtube_url)
        if potential_path.exists() and potential_path.is_file():
            is_local_file = True
            local_file_path = potential_path
            logger.info(f"Detected local file: {local_file_path}")
            # Use file name as title if title is empty
            if not title or not title.strip():
                title = local_file_path.stem
    else:
        # Check if it's a Google Drive URL
        from core.media_downloader import is_google_drive_url

        is_drive_url = is_google_drive_url(youtube_url)
        if is_drive_url:
            logger.info(f"Detected Google Drive URL: {youtube_url}")

    # --- OPTIMIZATION START (Issue #8) ---
    # Extract ID once at start to avoid duplicate network calls.
    # Try regex first (fast), fall back to yt-dlp (slow).
    video_id: str | None = None
    if not is_local_file and not is_drive_url:
        import re

        # Try Regex 1 (standard)
        m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", youtube_url)
        if m:
            video_id = m.group(1)

        # Fallback to yt-dlp only if regex failed (and we need it for cache)
        if not video_id and (reuse_transcripts or settings.TRANSCRIPT_CACHE_ENABLED):
            try:
                import yt_dlp

                with yt_dlp.YoutubeDL({"quiet": True, "noplaylist": True}) as ydl:
                    info = ydl.extract_info(youtube_url, download=False)
                    if info:
                        video_id = info.get("id")
            except Exception:
                video_id = None
    # --- OPTIMIZATION END ---

    if is_local_file:
        logger.info(f"Starting transcription for local file: {local_file_path}")
    elif is_drive_url:
        logger.info(f"Starting transcription for Google Drive file: {youtube_url}")
    else:
        logger.info(f"Starting transcription for URL: {youtube_url} (Video ID: {video_id})")

    unique_job_id = datetime.now().strftime("%Y%m%d%H%M%S%f")

    # Ensure output directories
    settings.OUTPUT_TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    settings.SUMMARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    settings.TEMP_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)  # Ensure base temp dir exists

    # Fast path: reuse cached transcript if available (only for YouTube URLs, not Drive or local files)
    if (reuse_transcripts or settings.TRANSCRIPT_CACHE_ENABLED) and video_id:
        video_id_for_cache = video_id
        # Removed redundant yt-dlp call here

        if video_id_for_cache:
            cache_file = settings.TRANSCRIPT_CACHE_DIR / f"{video_id_for_cache}.txt"
            if cache_file.exists():
                try:
                    transcript_text = cache_file.read_text(encoding="utf-8").strip()
                    if transcript_text:
                        normalized_title = utils.normalize_title_for_filename(title)
                        output_filename_base = (
                            f"{normalized_title}_vid_{video_id_for_cache}_job_{unique_job_id}"
                        )
                        transcript_path = utils.save_transcription_to_file(
                            transcription_text=transcript_text,
                            output_filename_no_ext=output_filename_base,
                            output_dir=settings.OUTPUT_TRANSCRIPTS_DIR,
                            original_title=title,
                        )
                        logger.info(f"Reused cached transcript, saved to: {transcript_path}")
                        print(f"\nTranscript (cached) saved to: {transcript_path}")

                        # Generate summaries if requested
                        if generate_summary:
                            summary_path_en, summary_path_es, post_kits_path = (
                                _generate_summary_outputs(
                                    transcript_text=transcript_text,
                                    video_title=title,
                                    video_url=youtube_url,
                                    video_id=video_id_for_cache,
                                    output_filename_base=output_filename_base,
                                    generate_post_kits=generate_post_kits,
                                    is_local_file=is_local_file,
                                    source_label="cache",
                                )
                            )
                            return transcript_path, summary_path_en, summary_path_es, post_kits_path

                        return transcript_path, None, None, None
                except Exception as e:
                    logger.warning(f"Error reading transcript cache: {e}")

    # Use TemporaryDirectory for robust cleanup (Issue #12)
    # This automatically cleans up files even if exceptions occur
    try:
        with tempfile.TemporaryDirectory(
            dir=settings.TEMP_DOWNLOAD_DIR, prefix=f"{unique_job_id}_"
        ) as temp_dir_str:
            job_temp_dir = Path(temp_dir_str)
            logger.info(f"Using temp dir: {job_temp_dir}")

            # 1) Download/extract audio
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

            # 3) Save transcript
            logger.info("Step 3: Saving transcript...")
            normalized_title = utils.normalize_title_for_filename(title)
            output_filename_base = (
                f"{normalized_title}_vid_{download_result.video_id}_job_{unique_job_id}"
            )
            transcript_path = utils.save_transcription_to_file(
                transcription_text=transcription_result.text,
                output_filename_no_ext=output_filename_base,
                output_dir=settings.OUTPUT_TRANSCRIPTS_DIR,
                original_title=title,
            )

            if not transcript_path:
                raise OSError("Could not save transcript file.")

            logger.info(f"Transcript saved successfully to: {transcript_path}")
            print(f"\nTranscript saved to: {transcript_path}")

            # Cache transcript for future runs if enabled (only for YouTube URLs, not Drive or local files)
            try:
                if (
                    (reuse_transcripts or settings.TRANSCRIPT_CACHE_ENABLED)
                    and not is_local_file
                    and not is_drive_url
                ):
                    video_id_for_cache = download_result.video_id

                    if video_id_for_cache:
                        cache_path = settings.TRANSCRIPT_CACHE_DIR / f"{video_id_for_cache}.txt"
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        cache_path.write_text(
                            transcript_path.read_text(encoding="utf-8"), encoding="utf-8"
                        )
            except Exception as e:
                logger.warning(f"Could not write transcript to cache: {e}")

            # 4) Generate summaries (only if requested)
            if not generate_summary:
                logger.info("Transcript only requested (no --summarize)")
                return transcript_path, None, None, None

            logger.info("Step 4: Generating AI summary...")
            try:
                summary_path_en, summary_path_es, post_kits_path = _generate_summary_outputs(
                    transcript_text=transcription_result.text,
                    video_title=title,
                    video_url=youtube_url,
                    video_id=download_result.video_id,
                    output_filename_base=output_filename_base,
                    generate_post_kits=generate_post_kits,
                    is_local_file=is_local_file,
                )
                return transcript_path, summary_path_en, summary_path_es, post_kits_path
            except Exception as e:
                logger.warning(f"Could not generate summary (continuing): {e}")
                print(f"\nWarning: Could not generate summary: {e}", file=sys.stderr)
                return transcript_path, None, None, None

    except (OSError, DownloadError, TranscriptionError) as e:
        logger.error(f"An error occurred during processing: {e}", exc_info=True)
        print(f"\nError: {e}", file=sys.stderr)
        return None, None, None, None
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        return None, None, None, None
