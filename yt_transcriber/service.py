"""Transcriber Service - shared orchestration for download/transcribe/summarize.

This module centralizes the end-to-end flow used by both CLI and Web UI:
Download → Transcribe → (Summarize EN) → (Translate ES) → (Post Kits)

CLI keeps a thin wrapper that forwards to this service for testability and
backward compatibility (tests patch functions in yt_transcriber.cli).
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

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

logger = logging.getLogger(__name__)


def process_transcription(
    youtube_url: str,
    title: str,
    model,
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
            logger.info(f"Detectado archivo local: {local_file_path}")
            # Use file name as title if title is empty
            if not title or not title.strip():
                title = local_file_path.stem
    else:
        # Check if it's a Google Drive URL
        from core.media_downloader import is_google_drive_url
        is_drive_url = is_google_drive_url(youtube_url)
        if is_drive_url:
            logger.info(f"Detectada URL de Google Drive: {youtube_url}")

    if is_local_file:
        logger.info(f"Iniciando transcripción para archivo local: {local_file_path}")
    elif is_drive_url:
        logger.info(f"Iniciando transcripción para archivo de Google Drive: {youtube_url}")
    else:
        logger.info(f"Iniciando transcripción para URL: {youtube_url}")
    unique_job_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    job_temp_dir = settings.TEMP_DOWNLOAD_DIR / unique_job_id

    # Ensure output directories
    settings.OUTPUT_TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    settings.SUMMARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Fast path: reuse cached transcript if available (only for YouTube URLs, not Drive or local files)
    if (reuse_transcripts or settings.TRANSCRIPT_CACHE_ENABLED) and not is_local_file and not is_drive_url:
        video_id_for_cache: str | None = None
        try:
            import yt_dlp

            with yt_dlp.YoutubeDL({"quiet": True, "noplaylist": True}) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                if info:
                    video_id_for_cache = info.get("id")
        except Exception:
            video_id_for_cache = None

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
                        logger.info(
                            f"✔ Reutilizada transcripción cacheada y guardada en: {transcript_path}"
                        )
                        print(f"\n✔ Transcripción (cache) guardada en: {transcript_path}")

                        # Summaries if requested and model is configured
                        ok, reason = is_model_configured(settings.SUMMARIZER_MODEL)
                        if generate_summary and ok:
                            summary_en = create_summary(
                                transcript=transcript_text,
                                video_title=title,
                                video_url=youtube_url,
                                video_id=video_id_for_cache,
                            )

                            summary_filename_en = f"{output_filename_base}_summary_EN.md"
                            summary_path_en = settings.SUMMARY_OUTPUT_DIR / summary_filename_en
                            summary_path_en.write_text(summary_en.to_markdown(), encoding="utf-8")
                            logger.info(
                                f"✔ Resumen EN (cache) guardado exitosamente en: {summary_path_en}"
                            )
                            print(f"✔ Resumen (EN) guardado en: {summary_path_en}")

                            # Translate to ES
                            try:
                                translator = ScriptTranslator(use_translation_model=True)
                                summary_es = translator.translate_summary(summary_en)
                                summary_filename_es = f"{output_filename_base}_summary_ES.md"
                                summary_path_es = settings.SUMMARY_OUTPUT_DIR / summary_filename_es
                                summary_path_es.write_text(
                                    summary_es.to_markdown(), encoding="utf-8"
                                )
                                logger.info(
                                    f"✔ Resumen ES (cache) guardado exitosamente en: {summary_path_es}"
                                )
                                print(f"✔ Resumen (ES) guardado en: {summary_path_es}")

                                # Optional Post Kits
                                post_kits_path = None
                                if generate_post_kits:
                                    try:
                                        from yt_transcriber.post_kits_generator import (
                                            generate_post_kits as gen_kits,
                                        )

                                        post_kits = gen_kits(
                                            summary=summary_en,
                                            video_title=title,
                                            video_url=youtube_url,
                                        )
                                        post_kits_filename = f"{output_filename_base}_post_kits.md"
                                        post_kits_path = (
                                            settings.SUMMARY_OUTPUT_DIR / post_kits_filename
                                        )
                                        post_kits_path.write_text(
                                            post_kits.to_markdown(), encoding="utf-8"
                                        )
                                        logger.info(
                                            f"✔ Post Kits (cache) guardado exitosamente en: {post_kits_path}"
                                        )
                                        print(f"✔ Post Kits guardado en: {post_kits_path}")
                                    except Exception as e:
                                        logger.warning(
                                            f"No se pudieron generar los Post Kits (cache): {e}"
                                        )
                                return (
                                    transcript_path,
                                    summary_path_en,
                                    summary_path_es,
                                    post_kits_path,
                                )
                            except Exception as e:
                                logger.warning(
                                    f"No se pudo traducir el resumen (cache, continuando): {e}"
                                )
                                return transcript_path, summary_path_en, None, None

                        if not ok:
                            logger.warning(
                                f"Modelo de resumen no configurado: {reason}. Se omite el resumen y Post Kits."
                            )
                            print(
                                f"\n⚠️  Advertencia: {reason}. Se omite el resumen y Post Kits.",
                                file=sys.stderr,
                            )
                        return transcript_path, None, None, None
                except Exception as e:
                    logger.warning(f"Error leyendo caché de transcripción: {e}")

    try:
        # 1) Download/extract audio
        if is_local_file and local_file_path:
            logger.info("Paso 1: Extrayendo audio de archivo local...")
            download_result = extract_audio_from_local_file(
                video_path=local_file_path,
                temp_dir=job_temp_dir,
                unique_job_id=unique_job_id,
                ffmpeg_location=ffmpeg_location,
            )
        else:
            logger.info("Paso 1: Descargando y extrayendo audio...")
            download_result = download_and_extract_audio(
                youtube_url=youtube_url,
                temp_dir=job_temp_dir,
                unique_job_id=unique_job_id,
                ffmpeg_location=ffmpeg_location,
            )
        logger.info(f"Audio extraído a: {download_result.audio_path}")

        # 2) Transcribe
        logger.info("Paso 2: Transcribiendo audio...")
        transcription_result = transcribe_audio_file(
            audio_path=download_result.audio_path, model=model, language=language
        )
        logger.info(f"Transcripción completada. Idioma detectado: {transcription_result.language}")

        # 3) Save transcript
        logger.info("Paso 3: Guardando transcripción...")
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
            raise OSError("No se pudo guardar el archivo de transcripción.")

        logger.info(f"✔ Transcripción guardada exitosamente en: {transcript_path}")
        print(f"\n✔ Transcripción guardada en: {transcript_path}")

        # Cache transcript for future runs if enabled (only for YouTube URLs, not Drive or local files)
        try:
            if (reuse_transcripts or settings.TRANSCRIPT_CACHE_ENABLED) and not is_local_file and not is_drive_url:
                video_id_for_cache: str | None = None
                try:
                    import yt_dlp

                    with yt_dlp.YoutubeDL({"quiet": True, "noplaylist": True}) as ydl:
                        info = ydl.extract_info(youtube_url, download=False)
                        if info:
                            video_id_for_cache = info.get("id")
                except Exception:
                    video_id_for_cache = None

                if not video_id_for_cache:
                    import re

                    m = re.search(r"v=([A-Za-z0-9_-]{11})", youtube_url)
                    if not m:
                        m = re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", youtube_url)
                    video_id_for_cache = m.group(1) if m else None

                if video_id_for_cache:
                    cache_path = settings.TRANSCRIPT_CACHE_DIR / f"{video_id_for_cache}.txt"
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(
                        transcript_path.read_text(encoding="utf-8"), encoding="utf-8"
                    )
        except Exception as e:
            logger.warning(f"No se pudo escribir transcripción al caché: {e}")

        # 4) Generate EN summary (only if requested)
        if not generate_summary:
            logger.info("Solo transcripción solicitada (sin --summarize)")
            return transcript_path, None, None, None

        logger.info("Paso 4: Generando resumen con IA (inglés)...")
        try:

            ok, reason = is_model_configured(settings.SUMMARIZER_MODEL)
            if not ok:
                logger.warning(f"{reason}. Saltando resumen y Post Kits.")
                print(f"\n⚠️  Advertencia: {reason}. Se omite el resumen y Post Kits.")
                return transcript_path, None, None, None

            # Use empty URL for local files in summary generation
            video_url_for_summary = youtube_url if not is_local_file else ""
            summary_en = create_summary(
                transcript=transcription_result.text,
                video_title=title,
                video_url=video_url_for_summary,
                video_id=download_result.video_id,
            )

            settings.SUMMARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            summary_filename_en = f"{output_filename_base}_summary_EN.md"
            summary_path_en = settings.SUMMARY_OUTPUT_DIR / summary_filename_en
            summary_path_en.write_text(summary_en.to_markdown(), encoding="utf-8")

            logger.info(f"✔ Resumen EN guardado exitosamente en: {summary_path_en}")
            print(f"✔ Resumen (EN) guardado en: {summary_path_en}")

            # 5) Translate summary to ES
            logger.info("Paso 5: Traduciendo resumen a español...")
            try:
                translator = ScriptTranslator(use_translation_model=True)
                summary_es = translator.translate_summary(summary_en)

                summary_filename_es = f"{output_filename_base}_summary_ES.md"
                summary_path_es = settings.SUMMARY_OUTPUT_DIR / summary_filename_es
                summary_path_es.write_text(summary_es.to_markdown(), encoding="utf-8")

                logger.info(f"✔ Resumen ES guardado exitosamente en: {summary_path_es}")
                print(f"✔ Resumen (ES) guardado en: {summary_path_es}")

                # 6) Post Kits optionally
                post_kits_path = None
                if generate_post_kits:
                    ok_pk, reason_pk = is_model_configured(settings.SUMMARIZER_MODEL)
                    if not ok_pk:
                        logger.warning(f"Sin LLM configurado para Post Kits: {reason_pk}.")
                    else:
                        try:
                            from yt_transcriber.post_kits_generator import (
                                generate_post_kits as gen_kits,
                            )

                            # Use empty URL for local files in post kits
                            video_url_for_kits = youtube_url if not is_local_file else ""
                            post_kits = gen_kits(
                                summary=summary_en,
                                video_title=title,
                                video_url=video_url_for_kits,
                            )
                            post_kits_filename = f"{output_filename_base}_post_kits.md"
                            post_kits_path = settings.SUMMARY_OUTPUT_DIR / post_kits_filename
                            post_kits_path.write_text(post_kits.to_markdown(), encoding="utf-8")

                            logger.info(f"✔ Post Kits guardado exitosamente en: {post_kits_path}")
                            print(f"✔ Post Kits guardado en: {post_kits_path}")

                        except Exception as e:
                            logger.warning(
                                f"No se pudieron generar los Post Kits (continuando): {e}"
                            )
                            print(
                                f"\n⚠️  Advertencia: No se pudieron generar los Post Kits: {e}",
                                file=sys.stderr,
                            )

                return transcript_path, summary_path_en, summary_path_es, post_kits_path

            except Exception as e:
                logger.warning(f"No se pudo traducir el resumen (continuando): {e}")
                print(f"\n⚠️  Advertencia: No se pudo traducir el resumen: {e}", file=sys.stderr)
                return transcript_path, summary_path_en, None, None

        except Exception as e:
            logger.warning(f"No se pudo generar el resumen (continuando): {e}")
            print(f"\n⚠️  Advertencia: No se pudo generar el resumen: {e}", file=sys.stderr)
            return transcript_path, None, None, None

    except (OSError, DownloadError, TranscriptionError) as e:
        logger.error(f"Ha ocurrido un error en el proceso: {e}", exc_info=True)
        print(f"\nError: {e}", file=sys.stderr)
        return None, None, None, None
    except Exception as e:
        logger.critical(f"Ocurrió un error inesperado: {e}", exc_info=True)
        print(f"\nError inesperado: {e}", file=sys.stderr)
        return None, None, None, None
    finally:
        logger.info(f"Limpiando directorio temporal: {job_temp_dir}")
        utils.cleanup_temp_dir(job_temp_dir)
