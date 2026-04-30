"""Interactive TUI for yt-transcriber.

Wraps the programmatic API in `cli.py` (`run_transcribe_command`, `run_playlist_command`)
with questionary prompts for an interactive workflow. Launch via `python -m yt_transcriber.tui`,
the `ytt` bash function in ~/ai_configs/shell/ai-tools.sh, or `launch_tui.bat`.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path


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
