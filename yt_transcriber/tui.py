"""Interactive TUI for yt-transcriber.

Wraps the programmatic API in `cli.py` (`run_transcribe_command`, `run_playlist_command`)
with questionary prompts for an interactive workflow. Launch via `python -m yt_transcriber.tui`,
the `ytt` bash function in ~/ai_configs/shell/ai-tools.sh, or `launch_tui.bat`.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path

import questionary


class InputType(str, Enum):
    YOUTUBE_VIDEO = "youtube_video"
    YOUTUBE_PLAYLIST = "youtube_playlist"
    DRIVE = "drive"
    LOCAL = "local"
    UNKNOWN = "unknown"


def detect_input_type(value: str) -> InputType:
    """Classify a user-supplied input into one of the supported types.

    Order matters:
    1. Existing local path beats everything (so a path that happens to contain
       'youtube' is still treated as local).
    2. Explicit playlist marker (`playlist?list=` or `&list=`) trumps generic
       video URL detection — an URL with both v= and list= is treated as a
       playlist (most common user intent).
    3. Generic YouTube watch URL.
    4. Google Drive / Docs.
    5. Anything else → UNKNOWN.
    """
    stripped = value.strip()
    if not stripped:
        return InputType.UNKNOWN

    # 1. Local path
    try:
        if Path(stripped).expanduser().exists():
            return InputType.LOCAL
    except (OSError, ValueError):
        # Some strings (e.g. URLs with `?`) raise on Windows. Fall through.
        pass

    # 2. Playlist (explicit)
    if "playlist?list=" in stripped or "&list=" in stripped:
        return InputType.YOUTUBE_PLAYLIST

    # 3. YouTube video
    if "youtube.com/watch" in stripped or "youtu.be/" in stripped:
        return InputType.YOUTUBE_VIDEO

    # 4. Google Drive / Docs
    if "drive.google.com" in stripped or "docs.google.com" in stripped:
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
# Tooltip strings (shown as `instruction=` below each prompt)
# ---------------------------------------------------------------------------
_T_URL = "YouTube video URL, YouTube playlist URL, Google Drive URL, o ruta a archivo local"
_T_LANGUAGE_TRANSCRIBE = "Auto-detect funciona pero es algo más lento. Códigos ISO 639-1: es, en, pt, fr, de, ..."
_T_LANGUAGE_PLAYLIST = "Idioma de los subtítulos automáticos a descargar de YouTube"
_T_SUMMARIZE = "Genera resúmenes EN + ES con Claude (incrementa tiempo y consume cuota Claude)"
_T_POST_KITS = "LinkedIn post + Twitter thread. Activa --summarize automáticamente."
_T_SEGMENTS = "Genera _segments.json con timestamps por segmento (útil para procesar después)"
_T_VISUAL_EVIDENCE = "Extrae frames clave del video. Solo funciona con archivos locales. Activa --segments."
_T_LIMIT = "Vacío = playlist completa. Número entero = procesar últimos N videos."


def prompt_input_url() -> str:
    """Ask for the input URL/path. Returns stripped non-empty string or raises KeyboardInterrupt."""
    while True:
        value = questionary.text(
            "URL de YouTube/Drive o ruta local:",
            instruction=f"({_T_URL})",
        ).unsafe_ask()
        if value and value.strip():
            return value.strip()
        # else: empty -> re-prompt


def prompt_transcribe_options(input_type: InputType) -> dict:
    """Ask all transcribe-flow questions. Returns dict with options (pre-validation)."""
    # Language
    lang_choice = questionary.select(
        "Idioma del audio:",
        choices=["auto-detect", "es", "en", "otro (escribir código ISO)"],
        default="auto-detect",
        instruction=f"({_T_LANGUAGE_TRANSCRIBE})",
    ).unsafe_ask()

    if lang_choice == "auto-detect":
        language = None
    elif lang_choice.startswith("otro"):
        language = questionary.text(
            "Código de idioma (ej. pt, fr, de):",
            instruction="ISO 639-1; vacío = auto-detect",
        ).unsafe_ask()
        language = language.strip() or None
    else:
        language = lang_choice

    summarize = questionary.confirm(
        "¿Generar resúmenes (EN + ES)?",
        default=False,
        instruction=f"({_T_SUMMARIZE})",
    ).unsafe_ask()

    post_kits = questionary.confirm(
        "¿Generar post kits (LinkedIn + Twitter)?",
        default=False,
        instruction=f"({_T_POST_KITS})",
    ).unsafe_ask()

    segments = questionary.confirm(
        "¿Sidecar de segmentos JSON?",
        default=False,
        instruction=f"({_T_SEGMENTS})",
    ).unsafe_ask()

    if input_type == InputType.LOCAL:
        visual_evidence = questionary.confirm(
            "¿Extraer frames clave (visual evidence)?",
            default=False,
            instruction=f"({_T_VISUAL_EVIDENCE})",
        ).unsafe_ask()
    else:
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
    while True:
        limit_raw = questionary.text(
            "Cuántos videos (últimos N, vacío = todos):",
            instruction=f"({_T_LIMIT})",
        ).unsafe_ask()
        limit_raw = (limit_raw or "").strip()
        if not limit_raw:
            limit: int | None = None
            break
        try:
            n = int(limit_raw)
            if n <= 0:
                print("  El número debe ser positivo.")
                continue
            limit = n
            break
        except ValueError:
            print(f"  '{limit_raw}' no es un número entero.")
            continue

    lang_choice = questionary.select(
        "Idioma de auto-subs:",
        choices=["es", "en", "otro (escribir código ISO)"],
        default="es",
        instruction=f"({_T_LANGUAGE_PLAYLIST})",
    ).unsafe_ask()

    if lang_choice.startswith("otro"):
        language = questionary.text(
            "Código de idioma (ej. pt, fr, de):",
            instruction="ISO 639-1",
        ).unsafe_ask()
        language = language.strip() or "es"
    else:
        language = lang_choice

    summarize = questionary.confirm(
        "¿Generar resúmenes (EN + ES) por video?",
        default=False,
        instruction=f"({_T_SUMMARIZE})",
    ).unsafe_ask()

    post_kits = questionary.confirm(
        "¿Generar post kits por video?",
        default=False,
        instruction=f"({_T_POST_KITS})",
    ).unsafe_ask()

    return {
        "limit": limit,
        "language": language,
        "summarize": summarize,
        "post_kits": post_kits,
    }


def prompt_run_confirmation(preview: str) -> bool:
    """Show preview and ask for confirmation."""
    print()
    print("Comando equivalente:")
    print(f"  {preview}")
    print()
    return questionary.confirm("¿Ejecutar?", default=True).unsafe_ask()


def prompt_run_again() -> bool:
    """Ask if the user wants another run. Default Yes for chained workflows."""
    return questionary.confirm("¿Otra transcripción?", default=True).unsafe_ask()
