"""CLI for YouTube video transcription."""

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

from core.settings import settings

logger = logging.getLogger(__name__)


def setup_logging():
    """Configura el logging basico para la aplicacion."""
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


from yt_transcriber.whisper_context import whisper_model_context


class _Console:
    def print(self, *args, **kwargs):
        msg = " ".join(str(a) for a in args)
        logger.info(msg.strip())


console = _Console()


def get_youtube_title(youtube_url: str) -> str:
    """Extrae el titulo de un video de YouTube usando yt-dlp."""
    try:
        import yt_dlp

        with yt_dlp.YoutubeDL({"quiet": True, "noplaylist": True}) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            if info is None:
                return "untitled"
            title = info.get("title") or "untitled"
            return title
    except Exception as e:
        logger.error(f"No se pudo extraer el titulo automaticamente: {e}")
        return "untitled"


def process_transcription(
    youtube_url: str,
    title: str,
    model: Any,
    language: str | None = None,
    ffmpeg_location: str | None = None,
    segments_override: bool | None = None,
    visual_override: bool | None = None,
) -> Path | None:
    """Delegate transcription pipeline to the service implementation."""
    from yt_transcriber import service as _service

    return _service.process_transcription(
        youtube_url=youtube_url,
        title=title,
        model=model,
        language=language,
        ffmpeg_location=ffmpeg_location,
        segments_override=segments_override,
        visual_override=visual_override,
    )


def _ffmpeg_available(ffmpeg_location: str | None) -> bool:
    if ffmpeg_location:
        return Path(ffmpeg_location).exists()
    return shutil.which("ffmpeg") is not None or shutil.which("ffmpeg.exe") is not None


def command_transcribe(args):
    """Command handler for transcribing a single video / Drive / local file."""
    setup_logging()

    is_local_file = False
    local_file_path: Path | None = None
    is_drive_url = False

    if not (args.url.startswith("http://") or args.url.startswith("https://")):
        potential_path = Path(args.url)
        if potential_path.exists() and potential_path.is_file():
            is_local_file = True
            local_file_path = potential_path
            logger.info(f"Archivo local detectado: {local_file_path}")
        else:
            logger.error(f"La ruta no es una URL valida ni un archivo existente: {args.url}")
            print(
                "Error: Debe ser una URL valida (YouTube o Google Drive) o una ruta a un archivo de video local.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        from core.media_downloader import is_google_drive_url

        is_drive_url = is_google_drive_url(args.url)

        if is_drive_url:
            logger.info(f"URL de Google Drive detectada: {args.url}")
        else:
            if not (
                args.url.startswith("https://www.youtube.com/")
                or args.url.startswith("https://youtu.be/")
            ):
                logger.error(f"URL no valida: {args.url}")
                print(
                    "Error: La URL debe ser de YouTube o Google Drive, o una ruta a un archivo local.",
                    file=sys.stderr,
                )
                sys.exit(1)

    if is_local_file:
        if not _ffmpeg_available(args.ffmpeg_location):
            logger.error("FFmpeg es requerido para procesar archivos locales.")
            print("Error: FFmpeg es requerido para procesar archivos locales.", file=sys.stderr)
            sys.exit(1)
    else:
        if not _ffmpeg_available(args.ffmpeg_location):
            logger.warning("FFmpeg no encontrado. yt-dlp podria fallar al extraer audio.")

    if is_local_file and local_file_path:
        title = local_file_path.stem
        logger.info(f"Usando nombre de archivo como titulo: {title}")
    else:
        logger.info("Extrayendo titulo del video...")
        title = get_youtube_title(args.url)
        logger.info(f"Titulo extraido: {title}")

    with whisper_model_context() as model:
        transcript_path = process_transcription(
            youtube_url=args.url,
            title=title,
            model=model,
            language=args.language,
            ffmpeg_location=args.ffmpeg_location,
            segments_override=args.segments_override,
            visual_override=args.visual_override,
        )

    if transcript_path:
        logger.info("Proceso completado exitosamente.")
        logger.info(f"Transcripcion: {transcript_path}")
        sys.exit(0)
    else:
        logger.error("El proceso de transcripcion fallo.")
        sys.exit(1)


def run_transcribe_command(
    url: str,
    language: str | None = None,
    ffmpeg_location: str | None = None,
    segments_override: bool | None = None,
    visual_override: bool | None = None,
) -> str | None:
    """Programmatic transcription wrapper (used by the TUI)."""
    setup_logging()

    is_local_file = False
    local_file_path: Path | None = None

    if not (url.startswith("http://") or url.startswith("https://")):
        potential_path = Path(url)
        if potential_path.exists() and potential_path.is_file():
            is_local_file = True
            local_file_path = potential_path
            logger.info(f"Local file detected: {local_file_path}")
        else:
            logger.error(f"Invalid input: {url} is not a valid URL or existing file")
            return None
    else:
        from core.media_downloader import is_google_drive_url

        is_drive_url = is_google_drive_url(url)
        if not is_drive_url and not (
            url.startswith("https://www.youtube.com/") or url.startswith("https://youtu.be/")
        ):
            logger.error(f"Invalid URL: {url} (must be YouTube or Google Drive)")
            return None

    title = local_file_path.stem if is_local_file and local_file_path else get_youtube_title(url)
    lang = None if language == "Auto-detectar" else language

    try:
        with whisper_model_context() as model:
            transcript_path = process_transcription(
                youtube_url=url,
                title=title,
                model=model,
                language=lang,
                ffmpeg_location=ffmpeg_location,
                segments_override=segments_override,
                visual_override=visual_override,
            )
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return None

    return str(transcript_path) if transcript_path else None


def command_playlist(args):
    """Download auto-subs from a YouTube playlist into one folder per video."""
    setup_logging()

    import core.media_downloader as _dl
    from datetime import datetime

    from yt_transcriber.utils import normalize_title_for_filename

    logger.info(f"Extracting playlist entries from: {args.url}")
    try:
        entries = _dl.extract_playlist_entries(args.url)
    except _dl.DownloadError as e:
        logger.error(f"Failed to extract playlist: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not entries:
        logger.warning("Playlist is empty or could not be read.")
        print("No videos found in the playlist.")
        return {"successful": 0, "failed": 0, "files": []}

    if args.limit and args.limit > 0:
        entries = entries[-args.limit :]

    total = len(entries)
    logger.info(f"Processing {total} video(s) from playlist")

    completed = 0
    failed = 0
    files: list[str] = []

    for i, entry in enumerate(entries, 1):
        logger.info(f"[{i}/{total}] {entry.title}")
        print(f"\n[{i}/{total}] {entry.title}")

        normalized = normalize_title_for_filename(entry.title)
        unique_job_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        stem = f"{normalized}_vid_{entry.video_id}_job_{unique_job_id}"
        entry_dir = settings.OUTPUT_BASE_DIR / stem
        entry_dir.mkdir(parents=True, exist_ok=True)

        try:
            raw_path = _dl.download_auto_subtitles(
                video_url=entry.url,
                output_dir=entry_dir,
                lang=args.language,
            )

            if raw_path is None:
                logger.warning(f"No subtitles found for: {entry.title}")
                print(f"  -> No auto-subs ({args.language}) available, skipping.")
                # Cleanup the per-video folder so the user doesn't see ghosts
                # (use rmtree because yt-dlp may have written partial subtitle files)
                shutil.rmtree(entry_dir, ignore_errors=True)
                failed += 1
                continue

            # Rename <video_id>.txt to <stem>.txt for consistency with transcribe
            final_path = entry_dir / f"{stem}.txt"
            if raw_path != final_path:
                raw_path.rename(final_path)
            print(f"  -> Transcript saved: {final_path}")
            files.append(str(final_path))
            completed += 1

        except Exception as e:
            logger.error(f"Error processing {entry.title}: {e}")
            print(f"  -> Error: {e}", file=sys.stderr)
            failed += 1
            continue

    print(f"\nCompleted: {completed}/{total}, Failed: {failed}")
    logger.info(f"Playlist batch done. Completed: {completed}/{total}, Failed: {failed}")
    return {"successful": completed, "failed": failed, "files": files}


def main():
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="YouTube Video Transcriber",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Comandos disponibles")

    # transcribe
    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="Transcribe un video de YouTube, Google Drive o archivo local",
    )
    transcribe_parser.add_argument(
        "-u", "--url", required=True, type=str,
        help="URL de YouTube/Google Drive o ruta a archivo local.",
    )
    transcribe_parser.add_argument(
        "-l", "--language", type=str, default=None,
        help="Codigo de idioma (ej. 'en', 'es') para forzar la transcripcion.",
    )
    transcribe_parser.add_argument(
        "--ffmpeg-location", type=str, default=None,
        help="Ruta personalizada a FFmpeg.",
    )

    segments_group = transcribe_parser.add_mutually_exclusive_group()
    segments_group.add_argument(
        "--segments", dest="segments_override", action="store_true",
        help="Habilitar sidecar JSON de segmentos timestamped.",
    )
    segments_group.add_argument(
        "--no-segments", dest="segments_override", action="store_false",
        help="Deshabilitar sidecar JSON de segmentos timestamped.",
    )

    visual_group = transcribe_parser.add_mutually_exclusive_group()
    visual_group.add_argument(
        "--visual-evidence", dest="visual_override", action="store_true",
        help="Habilitar extracción de frames por segmento (solo archivos locales en V1).",
    )
    visual_group.add_argument(
        "--no-visual-evidence", dest="visual_override", action="store_false",
        help="Deshabilitar extracción de evidencia visual por segmento.",
    )

    transcribe_parser.set_defaults(segments_override=None, visual_override=None)

    # playlist
    playlist_parser = subparsers.add_parser(
        "playlist",
        help="Descarga auto-subs de una playlist de YouTube en batch",
    )
    playlist_parser.add_argument(
        "-u", "--url", required=True, type=str,
        help="URL de la playlist de YouTube.",
    )
    playlist_parser.add_argument(
        "-n", "--limit", type=int, default=None,
        help="Ultimos N videos de la playlist (default: todos).",
    )
    playlist_parser.add_argument(
        "-l", "--language", type=str, default="es",
        help="Idioma de los subtitulos automaticos (default: es).",
    )

    args = parser.parse_args()

    if args.command == "transcribe":
        command_transcribe(args)
    elif args.command == "playlist":
        command_playlist(args)
    else:
        parser.print_help()
        sys.exit(1)


def run_playlist_command(
    url: str,
    limit: int | None = None,
    language: str = "es",
) -> dict:
    """Programmatic API mirror of command_playlist."""
    args = argparse.Namespace(url=url, limit=limit, language=language)

    try:
        result = command_playlist(args)
        if isinstance(result, dict):
            return result
        return {"successful": 0, "failed": 0, "files": []}
    except SystemExit as e:
        if e.code == 0:
            return {"successful": 0, "failed": 0, "files": []}
        return {"successful": 0, "failed": 1, "files": []}
    except Exception as e:
        print(f"[run_playlist_command] error: {e}", file=sys.stderr)
        return {"successful": 0, "failed": 1, "files": [], "error": str(e)}


if __name__ == "__main__":
    main()
