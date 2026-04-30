"""Interactive TUI for yt-transcriber.

Wraps the programmatic API in `cli.py` (`run_transcribe_command`, `run_playlist_command`)
with questionary prompts for an interactive workflow. Launch via `python -m yt_transcriber.tui`,
the `ytt` bash function in ~/ai_configs/shell/ai-tools.sh, or `launch_tui.bat`.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import questionary

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

console = Console()


class InputType(str, Enum):
    YOUTUBE_VIDEO = "youtube_video"
    YOUTUBE_PLAYLIST = "youtube_playlist"
    DRIVE = "drive"
    LOCAL = "local"
    UNKNOWN = "unknown"


_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"}
_DRIVE_HOSTS = {"drive.google.com", "docs.google.com"}


def detect_input_type(value: str) -> InputType:
    """Classify a user-supplied input into one of the supported types.

    Order:
    1. Existing local path beats URL detection.
    2. YouTube domain check (youtube.com / youtu.be variants).
       - If query has `list=` → playlist (regardless of v= presence/order).
       - Else youtube watch path or youtu.be path → video.
    3. Google Drive / Docs domain.
    4. Anything else → UNKNOWN.
    """
    stripped = value.strip()
    if not stripped:
        return InputType.UNKNOWN

    # 1. Local path
    try:
        if Path(stripped).expanduser().exists():
            return InputType.LOCAL
    except (OSError, ValueError):
        pass

    # Parse URL
    try:
        parsed = urlparse(stripped)
    except ValueError:
        return InputType.UNKNOWN

    host = (parsed.netloc or "").lower()

    # 2. YouTube
    if host in _YOUTUBE_HOSTS:
        query = parse_qs(parsed.query)
        if "list" in query:
            return InputType.YOUTUBE_PLAYLIST
        # video paths: /watch (long), /<id> (short youtu.be)
        if parsed.path.startswith("/watch") or host == "youtu.be":
            return InputType.YOUTUBE_VIDEO
        return InputType.UNKNOWN

    # 3. Drive
    if host in _DRIVE_HOSTS:
        return InputType.DRIVE

    return InputType.UNKNOWN


def apply_validation_rules(options: dict) -> dict:
    """Enforce CLI cross-flag implications on a copy of `options`.

    Rules (mirror `cli.py` and `service.py`):
    - `post_kits=True` forces `summarize=True`.
    - `visual_evidence=True` forces `segments=True`.

    Missing keys are left untouched (e.g. playlist options dict has no
    `segments` / `visual_evidence`).

    Returns a new dict; does not mutate the input.
    """
    result = dict(options)
    if result.get("post_kits"):
        result["summarize"] = True
    if result.get("visual_evidence"):
        result["segments"] = True
    return result


def format_command_preview(subcommand: str, url: str, options: dict) -> str:
    """Build a human-readable equivalent CLI command for the user to confirm.

    The returned string is informational (used in a 'about to run' prompt). It is
    NOT executed; the TUI calls the programmatic wrappers directly.

    Args:
        subcommand: "transcribe" or "playlist".
        url: the input URL or path.
        options: dict of resolved options (post-validation).

    Returns:
        A string like: `yt-transcriber transcribe -u "<url>" --language es --summarize`.
    """
    parts = ["yt-transcriber", subcommand, "-u", f'"{url}"']

    lang = options.get("language")
    if lang:
        parts.extend(["--language", lang])

    if subcommand == "playlist":
        limit = options.get("limit")
        if limit is not None:
            parts.extend(["--limit", str(limit)])

    if options.get("summarize"):
        parts.append("--summarize")
    if options.get("post_kits"):
        parts.append("--post-kits")

    if subcommand == "transcribe":
        if options.get("segments"):
            parts.append("--segments")
        if options.get("visual_evidence"):
            parts.append("--visual-evidence")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Tooltip strings (shown above each prompt via _hint)
# ---------------------------------------------------------------------------
_T_URL = "YouTube video URL, YouTube playlist URL, Google Drive URL, o ruta a archivo local"
_T_LANGUAGE_TRANSCRIBE = "Auto-detect funciona pero es algo más lento. Códigos ISO 639-1: es, en, pt, fr, de, ..."
_T_LANGUAGE_PLAYLIST = "Idioma de los subtítulos automáticos a descargar de YouTube"
_T_SUMMARIZE = "Genera resúmenes EN + ES con Claude (incrementa tiempo y consume cuota Claude)"
_T_POST_KITS = "LinkedIn post + Twitter thread. Activa --summarize automáticamente."
_T_SEGMENTS = "Genera _segments.json con timestamps por segmento (útil para procesar después)"
_T_VISUAL_EVIDENCE = "Extrae frames clave del video. Solo funciona con archivos locales. Activa --segments."
_T_LIMIT = "Vacío = playlist completa. Número entero = procesar últimos N videos."


# ---------------------------------------------------------------------------
# Presentation helpers (rich)
# ---------------------------------------------------------------------------

def _banner() -> None:
    """Print the welcome banner once at TUI start."""
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]yt-transcriber[/bold cyan] — TUI",
            border_style="cyan",
            padding=(0, 2),
        )
    )


def _section(title: str) -> None:
    """Horizontal rule with a title — visual separator between phases."""
    console.print()
    console.print(Rule(f"[bold]{title}[/bold]", style="cyan"))
    console.print()


def _hint(text: str) -> None:
    """Tooltip rendered above the next prompt, dim style with arrow prefix."""
    console.print(f"  [dim]→ {text}[/dim]")


def _success(text: str) -> None:
    """Positive line (file generated, type detected, etc.)."""
    console.print(f"  [green]✓[/green] {text}")


def _skip(text: str) -> None:
    """Item that does not apply / was skipped, neutral but visible."""
    console.print(f"  [yellow]✗[/yellow] {text}")


def _warn(text: str) -> None:
    """Warning or recoverable issue."""
    console.print(f"  [yellow][!] {text}[/yellow]")


def _info_line(label: str, value: str) -> None:
    """Bullet line for the pre-run summary."""
    console.print(f"  • [bold]{label:<14}[/bold] {value}")


def prompt_input_url() -> str:
    """Ask for the input URL/path. Returns stripped non-empty string or raises KeyboardInterrupt."""
    while True:
        _hint(_T_URL)
        value = questionary.text("URL o ruta:").unsafe_ask()
        if value and value.strip():
            console.print()
            return value.strip()


def prompt_transcribe_options(input_type: InputType) -> dict:
    """Ask all transcribe-flow questions. Returns dict with options (pre-validation)."""
    total = 5

    # [1/5] Language
    _hint(_T_LANGUAGE_TRANSCRIBE)
    lang_choice = questionary.select(
        f"[1/{total}] Idioma del audio:",
        choices=["auto-detect", "es", "en", "otro (escribir código ISO)"],
        default="auto-detect",
    ).unsafe_ask()
    console.print()

    if lang_choice == "auto-detect":
        language = None
    elif lang_choice.startswith("otro"):
        _hint("ISO 639-1, ej. pt, fr, de. Vacío = auto-detect.")
        language = questionary.text("Código de idioma:").unsafe_ask()
        language = (language or "").strip() or None
        console.print()
    else:
        language = lang_choice

    # [2/5] Summarize
    _hint(_T_SUMMARIZE)
    summarize = questionary.confirm(
        f"[2/{total}] ¿Generar resúmenes (EN + ES)?",
        default=False,
    ).unsafe_ask()
    console.print()

    # [3/5] Post kits
    _hint(_T_POST_KITS)
    post_kits = questionary.confirm(
        f"[3/{total}] ¿Generar post kits (LinkedIn + Twitter)?",
        default=False,
    ).unsafe_ask()
    console.print()

    # [4/5] Segments
    _hint(_T_SEGMENTS)
    segments = questionary.confirm(
        f"[4/{total}] ¿Sidecar de segmentos JSON?",
        default=False,
    ).unsafe_ask()
    console.print()

    # [5/5] Visual evidence (only if local)
    if input_type == InputType.LOCAL:
        _hint(_T_VISUAL_EVIDENCE)
        visual_evidence = questionary.confirm(
            f"[5/{total}] ¿Extraer frames clave (visual evidence)?",
            default=False,
        ).unsafe_ask()
        console.print()
    else:
        _skip(f"[5/{total}] Visual evidence: omitido (no aplica para URLs)")
        console.print()
        visual_evidence = False

    return {
        "language": language,
        "summarize": summarize,
        "post_kits": post_kits,
        "segments": segments,
        "visual_evidence": visual_evidence,
    }


def prompt_playlist_options() -> dict:
    """Ask all playlist-flow questions. Returns dict with options (pre-validation)."""
    total = 4

    # [1/4] Limit
    while True:
        _hint(_T_LIMIT)
        limit_raw = questionary.text(
            f"[1/{total}] Cuántos videos (vacío = todos):",
        ).unsafe_ask()
        limit_raw = (limit_raw or "").strip()
        if not limit_raw:
            limit: int | None = None
            console.print()
            break
        try:
            n = int(limit_raw)
            if n <= 0:
                _warn("El número debe ser positivo.")
                continue
            limit = n
            console.print()
            break
        except ValueError:
            _warn(f"'{limit_raw}' no es un número entero.")
            continue

    # [2/4] Language
    _hint(_T_LANGUAGE_PLAYLIST)
    lang_choice = questionary.select(
        f"[2/{total}] Idioma de auto-subs:",
        choices=["es", "en", "otro (escribir código ISO)"],
        default="es",
    ).unsafe_ask()
    console.print()

    if lang_choice.startswith("otro"):
        _hint("ISO 639-1, ej. pt, fr, de.")
        language = questionary.text("Código de idioma:").unsafe_ask()
        language = (language or "").strip() or "es"
        console.print()
    else:
        language = lang_choice

    # [3/4] Summarize
    _hint(_T_SUMMARIZE)
    summarize = questionary.confirm(
        f"[3/{total}] ¿Generar resúmenes (EN + ES) por video?",
        default=False,
    ).unsafe_ask()
    console.print()

    # [4/4] Post kits
    _hint(_T_POST_KITS)
    post_kits = questionary.confirm(
        f"[4/{total}] ¿Generar post kits por video?",
        default=False,
    ).unsafe_ask()
    console.print()

    return {
        "limit": limit,
        "language": language,
        "summarize": summarize,
        "post_kits": post_kits,
    }


def prompt_run_confirmation() -> bool:
    """Ask for go/no-go. The pre-run summary is printed by the caller."""
    return questionary.confirm("¿Ejecutar?", default=True).unsafe_ask()


def prompt_run_again() -> bool:
    """Ask if the user wants another run. Default Yes for chained workflows."""
    return questionary.confirm("¿Otra transcripción?", default=True).unsafe_ask()


def _print_transcribe_results(result: tuple) -> None:
    """Pretty-print the tuple returned by run_transcribe_command."""
    transcript_path, summary_en, summary_es, post_kits = result
    if not any(result):
        print("\n[!] No se generó ningún archivo. Revisa los logs.")
        return
    print("\nArchivos generados:")
    if transcript_path:
        print(f"  - transcript: {transcript_path}")
    if summary_en:
        print(f"  - summary EN: {summary_en}")
    if summary_es:
        print(f"  - summary ES: {summary_es}")
    if post_kits:
        print(f"  - post kits:  {post_kits}")


def _print_playlist_results(stats: dict) -> None:
    """Pretty-print the dict returned by run_playlist_command.

    Real per-video counters (since the playlist refactor): `successful`, `failed`,
    `files`, optional `error` for unexpected exceptions.
    """
    successful = stats.get("successful", 0)
    failed = stats.get("failed", 0)
    files = stats.get("files", [])
    error = stats.get("error")

    total = successful + failed
    if total == 0 and not error:
        print("\n[i] Playlist vacía o sin trabajo nuevo.")
        return

    print(f"\nPlaylist procesada — {successful}/{total} videos con éxito.")
    if failed:
        print(f"  [!] {failed} fallos. Revisa los logs arriba.")
    if error:
        print(f"  [!] Error fatal: {error}")
    if files:
        print(f"\nArchivos generados ({len(files)}):")
        for f in files[:10]:
            print(f"  - {f}")
        if len(files) > 10:
            print(f"  ... y {len(files) - 10} más")


def _run_transcribe(url: str, input_type: InputType) -> None:
    from yt_transcriber.cli import run_transcribe_command

    options = prompt_transcribe_options(input_type)
    options = apply_validation_rules(options)
    preview = format_command_preview("transcribe", url, options)

    if not prompt_run_confirmation():
        print("Cancelado.")
        return

    print("\nEjecutando...\n")
    result = run_transcribe_command(
        url=url,
        language=options["language"],
        ffmpeg_location=None,
        generate_post_kits=options["post_kits"],
        generate_summary=options["summarize"],
        reuse_transcripts=False,
        segments_override=options["segments"],
        visual_override=options["visual_evidence"],
    )
    _print_transcribe_results(result)


def _run_playlist(url: str) -> None:
    from yt_transcriber.cli import run_playlist_command

    options = prompt_playlist_options()
    options = apply_validation_rules(options)
    preview = format_command_preview("playlist", url, options)

    if not prompt_run_confirmation():
        print("Cancelado.")
        return

    print("\nEjecutando...\n")
    stats = run_playlist_command(
        url=url,
        limit=options["limit"],
        language=options["language"],
        generate_summary=options["summarize"],
        generate_post_kits=options["post_kits"],
    )
    _print_playlist_results(stats)


def main() -> int:
    """TUI entry point. Returns exit code."""
    print("=" * 60)
    print("  yt-transcriber — TUI")
    print("=" * 60)

    while True:
        try:
            url = prompt_input_url()
            input_type = detect_input_type(url)

            if input_type == InputType.UNKNOWN:
                print(f"\n[!] No reconozco '{url}' como YouTube/Drive/archivo local. Reintenta.\n")
                continue

            print(f"\nDetectado: {input_type.value}\n")

            if input_type == InputType.YOUTUBE_PLAYLIST:
                _run_playlist(url)
            else:
                _run_transcribe(url, input_type)

            if not prompt_run_again():
                print("Hasta luego.")
                return 0

        except KeyboardInterrupt:
            print("\nCancelado por el usuario.")
            return 130
        except Exception as e:
            print(f"\n[!] Error inesperado: {type(e).__name__}: {e}")
            print("    Continuando con el loop. Ctrl+C para salir.")
            continue


if __name__ == "__main__":
    raise SystemExit(main())
