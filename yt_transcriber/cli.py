"""CLI for YouTube video transcription and summarization."""

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from core.settings import settings

if TYPE_CHECKING:
    import whisper

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
            title: str = info.get("title", "untitled")
            return title
    except Exception as e:
        logger.error(f"No se pudo extraer el titulo automaticamente: {e}")
        return "untitled"


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
    """Delegate transcription pipeline to the service implementation."""
    from yt_transcriber import service as _service

    return _service.process_transcription(
        youtube_url=youtube_url,
        title=title,
        model=model,
        language=language,
        ffmpeg_location=ffmpeg_location,
        generate_post_kits=generate_post_kits,
        generate_summary=generate_summary,
        reuse_transcripts=reuse_transcripts,
    )


def _ffmpeg_available(ffmpeg_location: str | None) -> bool:
    """Chequea si FFmpeg esta disponible (ruta explicita o en PATH)."""
    if ffmpeg_location:
        return Path(ffmpeg_location).exists()
    return shutil.which("ffmpeg") is not None or shutil.which("ffmpeg.exe") is not None


def command_transcribe(args):
    """Command handler for transcribing a single YouTube video or local file."""
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

    # --post-kits implies --summarize (post kits requires summary)
    generate_summary = getattr(args, "summarize", False) or args.post_kits

    # Use context manager for auto-cleanup of model memory
    with whisper_model_context() as model:
        transcript_path, summary_path_en, summary_path_es, post_kits_path = process_transcription(
            youtube_url=args.url,
            title=title,
            model=model,
            language=args.language,
            ffmpeg_location=args.ffmpeg_location,
            generate_post_kits=args.post_kits,
            generate_summary=generate_summary,
        )

    if transcript_path:
        logger.info("Proceso completado exitosamente.")
        logger.info("Archivos generados:")
        logger.info(f"Transcripcion: {transcript_path}")
        if summary_path_en:
            logger.info(f"Resumen (EN): {summary_path_en}")
        if summary_path_es:
            logger.info(f"Resumen (ES): {summary_path_es}")
        if post_kits_path:
            logger.info(f"Post Kits (LinkedIn + Twitter): {post_kits_path}")
        sys.exit(0)
    else:
        logger.error("El proceso de transcripcion fallo.")
        sys.exit(1)


def run_transcribe_command(
    url: str,
    language: str | None = None,
    ffmpeg_location: str | None = None,
    generate_post_kits: bool = False,
    generate_summary: bool = False,
    reuse_transcripts: bool = False,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Wrapper function for transcription to be called programmatically.

    Args:
        url: YouTube URL, Google Drive URL, or local file path
        language: Language code for transcription (None for auto-detect)
        ffmpeg_location: Custom FFmpeg path
        generate_post_kits: Generate LinkedIn + Twitter posts (implies generate_summary)
        generate_summary: Generate EN + ES summaries
        reuse_transcripts: Reuse cached transcripts

    Returns:
        Tuple of (transcript_path, summary_en_path, summary_es_path, post_kits_path)
    """
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
            return None, None, None, None
    else:
        from core.media_downloader import is_google_drive_url

        is_drive_url = is_google_drive_url(url)

        if not is_drive_url and not (
            url.startswith("https://www.youtube.com/") or url.startswith("https://youtu.be/")
        ):
            logger.error(f"Invalid URL: {url} (must be YouTube or Google Drive)")
            return None, None, None, None

    title = local_file_path.stem if is_local_file and local_file_path else get_youtube_title(url)

    lang = None if language == "Auto-detectar" else language

    try:
        with whisper_model_context() as model:
            transcript_path, summary_path_en, summary_path_es, post_kits_path = (
                process_transcription(
                    youtube_url=url,
                    title=title,
                    model=model,
                    language=lang,
                    ffmpeg_location=ffmpeg_location,
                    generate_post_kits=generate_post_kits,
                    generate_summary=generate_summary,
                    reuse_transcripts=reuse_transcripts,
                )
            )
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return None, None, None, None

    if transcript_path:
        return (
            str(transcript_path),
            str(summary_path_en) if summary_path_en else None,
            str(summary_path_es) if summary_path_es else None,
            str(post_kits_path) if post_kits_path else None,
        )
    else:
        return None, None, None, None


def main():
    """Punto de entrada principal para el CLI."""
    parser = argparse.ArgumentParser(
        description="YouTube Video Transcriber & Summarizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Comandos disponibles")

    # Subcommand: transcribe
    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="Transcribe un video de YouTube, Google Drive o archivo local",
    )
    transcribe_parser.add_argument(
        "-u",
        "--url",
        required=True,
        type=str,
        help="URL de YouTube/Google Drive o ruta a archivo local.",
    )
    transcribe_parser.add_argument(
        "-l",
        "--language",
        type=str,
        default=None,
        help="Codigo de idioma (ej. 'en', 'es') para forzar la transcripcion.",
    )
    transcribe_parser.add_argument(
        "--ffmpeg-location",
        type=str,
        default=None,
        help="Ruta personalizada a FFmpeg.",
    )
    transcribe_parser.add_argument(
        "--summarize",
        action="store_true",
        help="Generar resumenes EN + ES ademas de la transcripcion.",
    )
    transcribe_parser.add_argument(
        "--post-kits",
        action="store_true",
        help="Generar LinkedIn post y Twitter thread (implica --summarize).",
    )

    args = parser.parse_args()

    if args.command == "transcribe":
        command_transcribe(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
