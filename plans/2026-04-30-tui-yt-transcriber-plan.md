# TUI yt-transcriber Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an interactive TUI on top of the existing `yt-transcriber` CLI so the user can paste a URL/path, answer a few prompts with tooltips, and run transcription/playlist commands without typing flags. Launch via `ytt` alias (Git Bash/WezTerm) or `launch_tui.bat` (double-click on Windows).

**Architecture:** Single new file `yt_transcriber/tui.py` containing prompt logic and a main loop. It calls existing programmatic wrappers in `cli.py` (`run_transcribe_command`, plus a new `run_playlist_command` mirror). One new dependency (`questionary`). Launchers are thin wrappers (`.bat` + bash function in `~/ai_configs`). The `.env` is tuned once for the user's RTX 3060 Laptop hardware.

**Tech Stack:** Python 3.12, `questionary>=2.0.0` (TUI prompts), existing `pydantic`, `faster-whisper`, `yt-dlp`. Tests via `pytest`. Bash launcher for shells, `.bat` for Windows.

**Spec:** `docs/superpowers/specs/2026-04-30-tui-yt-transcriber-design.md`

---

## File Structure

| File | Action | Purpose |
|---|---|---|
| `pyproject.toml` | modify | Add `questionary>=2.0.0` to `[project] dependencies` |
| `.env` | modify | Replace with tuned config (medium Whisper, cuda, int8_float16, cache enabled, LLM models explicit) |
| `yt_transcriber/cli.py` | modify | Add `run_playlist_command(url, limit, language, generate_summary, generate_post_kits) -> dict` wrapper |
| `yt_transcriber/tui.py` | create | Module with `detect_input_type`, `prompt_transcribe_options`, `prompt_playlist_options`, `format_command_preview`, `apply_validation_rules`, `main` |
| `tests/yt_transcriber/test_tui.py` | create | Unit tests for `detect_input_type`, `format_command_preview`, `apply_validation_rules` |
| `tests/yt_transcriber/test_cli.py` | modify | Add `TestRunPlaylistCommand` class |
| `launch_tui.bat` | create | Windows double-click launcher with `.venv` validation |
| `~/ai_configs/shell/ai-tools.sh` | modify | Add `ytt()` bash function in `# AI Tools` section |

---

## Task 1: Bootstrap venv and add questionary dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add questionary to dependencies**

Edit `pyproject.toml`. Locate `[project] dependencies = [` block and add `questionary>=2.0.0`:

```toml
dependencies = [
    "yt-dlp>=2024.0.0",
    "faster-whisper>=1.1.0",
    "python-dotenv>=1.0.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "numpy>=1.24.0",
    "soundfile>=0.12.0",
    "tqdm>=4.65.0",
    "rich>=13.7.0",
    "questionary>=2.0.0",
]
```

- [ ] **Step 2: Create venv if not present**

Run from project root:

```bash
if [ ! -x .venv/Scripts/python.exe ]; then python -m venv .venv; fi
```

Expected: `.venv/Scripts/python.exe` exists after this step.

- [ ] **Step 3: Install editable + dev extras**

```bash
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -e ".[dev]"
```

Expected: install completes without errors. `questionary` and all existing deps land in `.venv/Lib/site-packages/`.

- [ ] **Step 4: Verify questionary import + existing tests still pass**

```bash
.venv/Scripts/python.exe -c "import questionary; print(questionary.__version__)"
.venv/Scripts/python.exe -m pytest -q
```

Expected: questionary version printed (e.g. `2.0.1`); all existing tests pass (no regression from adding the dep).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add questionary dep for TUI"
```

Note: `.venv/` is presumably already gitignored. If `git status` shows it tracked, add `.venv/` to `.gitignore` BEFORE committing.

---

## Task 2: Add `run_playlist_command` wrapper to cli.py (TDD)

**Files:**
- Modify: `yt_transcriber/cli.py` (add new function, do NOT modify `command_playlist`)
- Modify: `tests/yt_transcriber/test_cli.py` (add `TestRunPlaylistCommand` class)

- [ ] **Step 1: Write the failing tests**

Append to `tests/yt_transcriber/test_cli.py`:

```python
class TestRunPlaylistCommand:
    """Tests for run_playlist_command (programmatic wrapper, no sys.exit)."""

    @patch("yt_transcriber.cli.command_playlist")
    def test_returns_dict_with_stats(self, mock_command_playlist, tmp_path):
        from yt_transcriber.cli import run_playlist_command

        # command_playlist is invoked with a Namespace built internally; we don't care
        # about its side effects in this unit test, only that the wrapper returns
        # a dict with the expected keys without sys.exit.
        mock_command_playlist.return_value = None

        result = run_playlist_command(
            url="https://www.youtube.com/playlist?list=PLxxx",
            limit=3,
            language="es",
            generate_summary=False,
            generate_post_kits=False,
        )

        assert isinstance(result, dict)
        assert "successful" in result
        assert "failed" in result
        assert "files" in result
        assert isinstance(result["files"], list)

    @patch("yt_transcriber.cli.command_playlist")
    def test_does_not_call_sys_exit_on_systemexit(self, mock_command_playlist):
        """If command_playlist raises SystemExit, the wrapper catches it and reports failure."""
        from yt_transcriber.cli import run_playlist_command

        mock_command_playlist.side_effect = SystemExit(1)

        result = run_playlist_command(
            url="https://www.youtube.com/playlist?list=PLxxx",
            limit=None,
            language="es",
            generate_summary=False,
            generate_post_kits=False,
        )

        assert result["failed"] >= 1
        assert result["successful"] == 0

    @patch("yt_transcriber.cli.command_playlist")
    def test_does_not_call_sys_exit_on_exception(self, mock_command_playlist):
        """If command_playlist raises a generic Exception, wrapper catches it."""
        from yt_transcriber.cli import run_playlist_command

        mock_command_playlist.side_effect = RuntimeError("boom")

        result = run_playlist_command(
            url="https://www.youtube.com/playlist?list=PLxxx",
            limit=None,
            language="es",
            generate_summary=False,
            generate_post_kits=False,
        )

        assert result["failed"] >= 1
```

- [ ] **Step 2: Run tests, expect failure**

```bash
.venv/Scripts/python.exe -m pytest tests/yt_transcriber/test_cli.py::TestRunPlaylistCommand -v
```

Expected: FAIL with `ImportError: cannot import name 'run_playlist_command' from 'yt_transcriber.cli'`.

- [ ] **Step 3: Verify imports at top of cli.py**

Open `yt_transcriber/cli.py` and confirm both `import argparse` and `import sys` are present in the top imports block (they almost certainly are — `cli.py` defines the ArgumentParser). If either is missing, add it to the top imports block before continuing.

- [ ] **Step 4: Implement `run_playlist_command`**

Append to `yt_transcriber/cli.py` (at end of file, after `command_playlist`):

```python
def run_playlist_command(
    url: str,
    limit: int | None = None,
    language: str = "es",
    generate_summary: bool = False,
    generate_post_kits: bool = False,
) -> dict:
    """Programmatic API mirror of command_playlist.

    Builds a Namespace and invokes command_playlist while catching SystemExit and
    Exception so callers (e.g. the TUI) can handle errors without dying.

    Returns:
        dict with keys: successful (int), failed (int), files (list[str]).
    """
    args = argparse.Namespace(
        url=url,
        limit=limit,
        language=language,
        summarize=generate_summary,
        post_kits=generate_post_kits,
    )

    stats: dict = {"successful": 0, "failed": 0, "files": []}

    try:
        command_playlist(args)
        # command_playlist prints its own per-video progress; we record overall success.
        # If it returns normally, we trust it ran without fatal errors. Per-video failures
        # are visible in stdout but not retrievable here without refactoring command_playlist.
        # For V1 the dict is mostly a "did it finish" signal.
        stats["successful"] = 1
    except SystemExit as e:
        if e.code == 0:
            stats["successful"] = 1
        else:
            stats["failed"] = 1
    except Exception as e:
        stats["failed"] = 1
        print(f"[run_playlist_command] error: {e}", file=sys.stderr)

    return stats
```

- [ ] **Step 5: Run tests, expect pass**

```bash
.venv/Scripts/python.exe -m pytest tests/yt_transcriber/test_cli.py::TestRunPlaylistCommand -v
```

Expected: 3 tests PASS.

- [ ] **Step 6: Run full test suite (no regressions)**

```bash
.venv/Scripts/python.exe -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add yt_transcriber/cli.py tests/yt_transcriber/test_cli.py
git commit -m "feat: add run_playlist_command programmatic wrapper"
```

---

## Task 3: tui.py — detect_input_type (TDD)

**Files:**
- Create: `yt_transcriber/tui.py`
- Create: `tests/yt_transcriber/test_tui.py`

- [ ] **Step 1: Write failing tests**

Create `tests/yt_transcriber/test_tui.py`:

```python
"""Tests for yt_transcriber.tui module."""
from pathlib import Path

import pytest


class TestDetectInputType:
    def test_local_existing_file(self, tmp_path: Path):
        from yt_transcriber.tui import InputType, detect_input_type

        f = tmp_path / "video.mp4"
        f.write_text("fake")

        assert detect_input_type(str(f)) == InputType.LOCAL

    def test_local_nonexistent_path(self, tmp_path: Path):
        from yt_transcriber.tui import InputType, detect_input_type

        assert detect_input_type(str(tmp_path / "does_not_exist.mp4")) == InputType.UNKNOWN

    def test_youtube_video_long_url(self):
        from yt_transcriber.tui import InputType, detect_input_type

        assert detect_input_type("https://www.youtube.com/watch?v=abc123") == InputType.YOUTUBE_VIDEO

    def test_youtube_video_short_url(self):
        from yt_transcriber.tui import InputType, detect_input_type

        assert detect_input_type("https://youtu.be/abc123") == InputType.YOUTUBE_VIDEO

    def test_youtube_playlist_explicit(self):
        from yt_transcriber.tui import InputType, detect_input_type

        assert detect_input_type("https://www.youtube.com/playlist?list=PLxxx") == InputType.YOUTUBE_PLAYLIST

    def test_youtube_video_with_list_param_prefers_playlist(self):
        """An URL with both v= and list= should be treated as playlist."""
        from yt_transcriber.tui import InputType, detect_input_type

        url = "https://www.youtube.com/watch?v=abc&list=PLxxx"
        assert detect_input_type(url) == InputType.YOUTUBE_PLAYLIST

    def test_drive_file_url(self):
        from yt_transcriber.tui import InputType, detect_input_type

        assert detect_input_type("https://drive.google.com/file/d/xxx/view") == InputType.DRIVE

    def test_drive_docs_url(self):
        from yt_transcriber.tui import InputType, detect_input_type

        assert detect_input_type("https://docs.google.com/document/d/xxx") == InputType.DRIVE

    def test_unknown_url(self):
        from yt_transcriber.tui import InputType, detect_input_type

        assert detect_input_type("https://example.com") == InputType.UNKNOWN

    def test_empty_string(self):
        from yt_transcriber.tui import InputType, detect_input_type

        assert detect_input_type("") == InputType.UNKNOWN

    def test_whitespace_string(self):
        from yt_transcriber.tui import InputType, detect_input_type

        assert detect_input_type("   ") == InputType.UNKNOWN
```

- [ ] **Step 2: Run tests, expect failure**

```bash
.venv/Scripts/python.exe -m pytest tests/yt_transcriber/test_tui.py::TestDetectInputType -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'yt_transcriber.tui'`.

- [ ] **Step 3: Create `yt_transcriber/tui.py` with detect_input_type**

```python
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
```

- [ ] **Step 4: Run tests, expect pass**

```bash
.venv/Scripts/python.exe -m pytest tests/yt_transcriber/test_tui.py::TestDetectInputType -v
```

Expected: 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add yt_transcriber/tui.py tests/yt_transcriber/test_tui.py
git commit -m "feat: add tui.detect_input_type with tests"
```

---

## Task 4: tui.py — apply_validation_rules (TDD)

**Files:**
- Modify: `yt_transcriber/tui.py`
- Modify: `tests/yt_transcriber/test_tui.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/yt_transcriber/test_tui.py`:

```python
class TestApplyValidationRules:
    def test_post_kits_forces_summarize(self):
        from yt_transcriber.tui import apply_validation_rules

        opts = {"summarize": False, "post_kits": True, "segments": False, "visual_evidence": False}
        result = apply_validation_rules(opts)
        assert result["summarize"] is True
        assert result["post_kits"] is True

    def test_visual_evidence_forces_segments(self):
        from yt_transcriber.tui import apply_validation_rules

        opts = {"summarize": False, "post_kits": False, "segments": False, "visual_evidence": True}
        result = apply_validation_rules(opts)
        assert result["segments"] is True
        assert result["visual_evidence"] is True

    def test_no_implications_keeps_values(self):
        from yt_transcriber.tui import apply_validation_rules

        opts = {"summarize": True, "post_kits": False, "segments": True, "visual_evidence": False}
        result = apply_validation_rules(opts)
        assert result == opts

    def test_both_implications_apply(self):
        from yt_transcriber.tui import apply_validation_rules

        opts = {"summarize": False, "post_kits": True, "segments": False, "visual_evidence": True}
        result = apply_validation_rules(opts)
        assert result["summarize"] is True
        assert result["segments"] is True

    def test_does_not_mutate_input(self):
        from yt_transcriber.tui import apply_validation_rules

        opts = {"summarize": False, "post_kits": True, "segments": False, "visual_evidence": False}
        opts_copy = dict(opts)
        apply_validation_rules(opts)
        assert opts == opts_copy  # original untouched

    def test_playlist_options_no_segments_keys(self):
        """Playlist options dict has no segments/visual_evidence keys; rules must not crash."""
        from yt_transcriber.tui import apply_validation_rules

        opts = {"summarize": False, "post_kits": True}
        result = apply_validation_rules(opts)
        assert result["summarize"] is True
```

- [ ] **Step 2: Run tests, expect failure**

```bash
.venv/Scripts/python.exe -m pytest tests/yt_transcriber/test_tui.py::TestApplyValidationRules -v
```

Expected: FAIL with `ImportError: cannot import name 'apply_validation_rules'`.

- [ ] **Step 3: Add `apply_validation_rules` to `yt_transcriber/tui.py`**

Append to `yt_transcriber/tui.py` (after `detect_input_type`):

```python
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
```

- [ ] **Step 4: Run tests, expect pass**

```bash
.venv/Scripts/python.exe -m pytest tests/yt_transcriber/test_tui.py::TestApplyValidationRules -v
```

Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add yt_transcriber/tui.py tests/yt_transcriber/test_tui.py
git commit -m "feat: add tui.apply_validation_rules"
```

---

## Task 5: tui.py — format_command_preview (TDD)

**Files:**
- Modify: `yt_transcriber/tui.py`
- Modify: `tests/yt_transcriber/test_tui.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/yt_transcriber/test_tui.py`:

```python
class TestFormatCommandPreview:
    def test_transcribe_minimal(self):
        from yt_transcriber.tui import format_command_preview

        opts = {"language": None, "summarize": False, "post_kits": False, "segments": False, "visual_evidence": False}
        out = format_command_preview("transcribe", "https://youtu.be/abc", opts)

        assert "transcribe" in out
        assert "https://youtu.be/abc" in out
        assert "--summarize" not in out
        assert "--language" not in out

    def test_transcribe_with_language_and_summarize(self):
        from yt_transcriber.tui import format_command_preview

        opts = {"language": "es", "summarize": True, "post_kits": False, "segments": False, "visual_evidence": False}
        out = format_command_preview("transcribe", "https://youtu.be/abc", opts)

        assert "--language es" in out
        assert "--summarize" in out

    def test_transcribe_with_post_kits_and_visual(self):
        from yt_transcriber.tui import format_command_preview

        opts = {"language": "en", "summarize": True, "post_kits": True, "segments": True, "visual_evidence": True}
        out = format_command_preview("transcribe", "/local/video.mp4", opts)

        assert "--post-kits" in out
        assert "--visual-evidence" in out
        assert "--segments" in out
        assert "/local/video.mp4" in out

    def test_playlist_minimal(self):
        from yt_transcriber.tui import format_command_preview

        opts = {"limit": None, "language": "es", "summarize": False, "post_kits": False}
        out = format_command_preview("playlist", "https://www.youtube.com/playlist?list=PLxxx", opts)

        assert "playlist" in out
        assert "--limit" not in out

    def test_playlist_with_limit(self):
        from yt_transcriber.tui import format_command_preview

        opts = {"limit": 5, "language": "en", "summarize": True, "post_kits": False}
        out = format_command_preview("playlist", "https://www.youtube.com/playlist?list=PLxxx", opts)

        assert "--limit 5" in out
        assert "--language en" in out
        assert "--summarize" in out
```

- [ ] **Step 2: Run tests, expect failure**

```bash
.venv/Scripts/python.exe -m pytest tests/yt_transcriber/test_tui.py::TestFormatCommandPreview -v
```

Expected: FAIL with `ImportError: cannot import name 'format_command_preview'`.

- [ ] **Step 3: Implement `format_command_preview`**

Append to `yt_transcriber/tui.py`:

```python
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
```

- [ ] **Step 4: Run tests, expect pass**

```bash
.venv/Scripts/python.exe -m pytest tests/yt_transcriber/test_tui.py::TestFormatCommandPreview -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add yt_transcriber/tui.py tests/yt_transcriber/test_tui.py
git commit -m "feat: add tui.format_command_preview"
```

---

## Task 6: tui.py — interactive prompts (no TDD, smoke-only)

Interactive `questionary` prompts are not unit-tested directly; we only verify the module imports and the prompt functions exist. End-to-end usability is verified manually in Task 10.

**Files:**
- Modify: `yt_transcriber/tui.py`
- Modify: `tests/yt_transcriber/test_tui.py`

- [ ] **Step 1: Add smoke tests**

Append to `tests/yt_transcriber/test_tui.py`:

```python
class TestSmoke:
    def test_module_imports(self):
        import yt_transcriber.tui  # noqa: F401

    def test_prompt_functions_exist(self):
        from yt_transcriber.tui import (
            prompt_transcribe_options,
            prompt_playlist_options,
            prompt_input_url,
            prompt_run_confirmation,
            prompt_run_again,
        )
        # Just check they are callables; we don't invoke them (would block on stdin)
        assert callable(prompt_transcribe_options)
        assert callable(prompt_playlist_options)
        assert callable(prompt_input_url)
        assert callable(prompt_run_confirmation)
        assert callable(prompt_run_again)
```

- [ ] **Step 2: Run tests, expect failure**

```bash
.venv/Scripts/python.exe -m pytest tests/yt_transcriber/test_tui.py::TestSmoke -v
```

Expected: FAIL — the prompt functions don't exist yet.

- [ ] **Step 3: Implement prompt helpers**

Append to `yt_transcriber/tui.py`:

```python
import questionary


# Tooltip strings (shown as `instruction=` below each prompt)
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
```

- [ ] **Step 4: Run smoke tests, expect pass**

```bash
.venv/Scripts/python.exe -m pytest tests/yt_transcriber/test_tui.py::TestSmoke -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add yt_transcriber/tui.py tests/yt_transcriber/test_tui.py
git commit -m "feat: add tui interactive prompts (questionary)"
```

---

## Task 7: tui.py — main loop (smoke + manual)

**Files:**
- Modify: `yt_transcriber/tui.py`

- [ ] **Step 1: Implement main and entry guard**

Append to `yt_transcriber/tui.py`:

```python
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
    """Pretty-print the dict returned by run_playlist_command."""
    successful = stats.get("successful", 0)
    failed = stats.get("failed", 0)
    print(f"\nPlaylist procesada — successful={successful}, failed={failed}")


def _run_transcribe(url: str, input_type: InputType) -> None:
    from yt_transcriber.cli import run_transcribe_command

    options = prompt_transcribe_options(input_type)
    options = apply_validation_rules(options)
    preview = format_command_preview("transcribe", url, options)

    if not prompt_run_confirmation(preview):
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

    if not prompt_run_confirmation(preview):
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
```

- [ ] **Step 2: Verify smoke import**

```bash
.venv/Scripts/python.exe -c "from yt_transcriber.tui import main; print(main)"
```

Expected: prints something like `<function main at 0x...>`. No import errors.

The interactive `python -m yt_transcriber.tui` invocation is verified manually in Task 11; here we only confirm the module imports cleanly.

- [ ] **Step 3: Run full test suite (no regressions)**

```bash
.venv/Scripts/python.exe -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add yt_transcriber/tui.py
git commit -m "feat: add tui main loop"
```

---

## Task 8: Update .env to tuned config

**Files:**
- Modify: `.env`

- [ ] **Step 1: Replace .env contents**

Overwrite `.env` with:

```dotenv
# =============================================================================
# Whisper (RTX 3060 Laptop 6GB VRAM)
# =============================================================================
WHISPER_MODEL_NAME=medium
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=int8_float16
WHISPER_BEAM_SIZE=5
WHISPER_VAD_FILTER=true

# =============================================================================
# LLM (Claude CLI)
# =============================================================================
SUMMARIZER_MODEL=sonnet
TRANSLATOR_MODEL=haiku
PRO_MODEL=opus
DEFAULT_LLM_MODEL=sonnet

# =============================================================================
# Cache (acelera reruns sobre el mismo video)
# =============================================================================
TRANSCRIPT_CACHE_ENABLED=true
TRANSCRIPT_CACHE_DIR=output/analysis/transcripts_cache/

# =============================================================================
# Output dirs
# =============================================================================
LOG_LEVEL=INFO
TEMP_DOWNLOAD_DIR=temp_files/
OUTPUT_TRANSCRIPTS_DIR=output/transcripts/
SUMMARY_OUTPUT_DIR=output/summaries/

# =============================================================================
# Features off por default — TUI los pregunta cuando aplican
# =============================================================================
TRANSCRIPT_SEGMENTS_ENABLED=false
VISUAL_EVIDENCE_ENABLED=false
VISUAL_EVIDENCE_MIN_SEGMENT_SECONDS=1.0
```

- [ ] **Step 2: Verify settings load OK**

```bash
.venv/Scripts/python.exe -c "from core.settings import settings; print('whisper=', settings.WHISPER_MODEL_NAME, settings.WHISPER_DEVICE, settings.WHISPER_COMPUTE_TYPE); print('cache=', settings.TRANSCRIPT_CACHE_ENABLED)"
```

Expected: prints `whisper= medium cuda int8_float16` and `cache= True`. No pydantic validation errors.

- [ ] **Step 3: No commit**

`.env` should be in `.gitignore`. If `git status` shows it tracked, do NOT commit (it may contain user-specific paths). Only commit if previously tracked.

---

## Task 9: Create launch_tui.bat

**Files:**
- Create: `launch_tui.bat`

- [ ] **Step 1: Create the batch file**

Create `launch_tui.bat` at repo root:

```batch
@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo ERROR: .venv no encontrado.
  echo Crear con:
  echo   python -m venv .venv
  echo   .venv\Scripts\pip install -e .[dev]
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m yt_transcriber.tui
pause
```

- [ ] **Step 2: Manual smoke test (Windows)**

Double-click `launch_tui.bat` in Explorer (or run `cmd.exe /c launch_tui.bat`). Expected:
- Window opens.
- Banner `yt-transcriber — TUI` prints.
- Prompt for URL appears.
- Press Ctrl+C → "Cancelado por el usuario."
- Window stays open (`pause`) until user presses any key.

- [ ] **Step 3: Commit**

```bash
git add launch_tui.bat
git commit -m "feat: add launch_tui.bat for windows double-click launch"
```

---

## Task 10: Add `ytt` alias to ~/ai_configs/shell/ai-tools.sh

**Files:**
- Modify: `~/ai_configs/shell/ai-tools.sh` (NOT the yt-transcriber repo)

This task touches a different repo (`~/ai_configs/`). Steps below assume that repo exists and is independent of this one.

- [ ] **Step 1: Locate the AI Tools section**

Find the section in `~/ai_configs/shell/ai-tools.sh` containing:

```bash
alias cc='claude'
alias ccy='claude --dangerously-skip-permissions'
alias gpt='codex'
alias gpty='codex --yolo'
alias gg='gemini'
alias ggy='gemini --yolo'
```

- [ ] **Step 2: Append ytt function after the gg/ggy aliases**

Add directly below `alias ggy='gemini --yolo'`:

```bash

# yt-transcriber TUI (subshell para no contaminar cwd)
ytt() {
  local proj="$HOME/IdeaProjects/yt-transcriber"
  if [[ ! -x "$proj/.venv/Scripts/python.exe" ]]; then
    echo "ERROR: $proj/.venv no existe. Crear con:" >&2
    echo "  cd $proj && python -m venv .venv && .venv/Scripts/pip install -e .[dev]" >&2
    return 1
  fi
  (cd "$proj" && "$proj/.venv/Scripts/python.exe" -m yt_transcriber.tui "$@")
}
```

- [ ] **Step 3: Reload bash profile and verify**

```bash
source ~/.bash_profile
type ytt
```

Expected: `ytt is a function` followed by the function body.

```bash
ytt
```

Expected: TUI banner + URL prompt. Press Ctrl+C to abort.

- [ ] **Step 4: Commit (in ~/ai_configs/)**

```bash
cd ~/ai_configs && git add shell/ai-tools.sh && git commit -m "feat: add ytt function for yt-transcriber TUI"
```

If the user prefers to keep `~/ai_configs/` uncommitted or has different conventions, skip this commit and inform the user.

---

## Task 11: Final verification — full test suite + manual smoke

**Files:** none (verification only)

- [ ] **Step 1: Full pytest suite**

```bash
.venv/Scripts/python.exe -m pytest -v
```

Expected: ALL tests pass (existing + new). Report counts: should be `existing_count + 25` new tests roughly (11 detect_input_type + 6 apply_validation_rules + 5 format_command_preview + 2 smoke + 3 run_playlist_command).

- [ ] **Step 2: Lint check (optional but recommended)**

```bash
.venv/Scripts/python.exe -m ruff check yt_transcriber/tui.py tests/yt_transcriber/test_tui.py
```

Expected: no errors.

- [ ] **Step 3: Manual TUI smoke (without actually transcribing)**

```bash
.venv/Scripts/python.exe -m yt_transcriber.tui
```

Steps in the TUI:
1. Paste a YouTube URL: `https://www.youtube.com/watch?v=dQw4w9WgXcQ`
2. Verify it detects `youtube_video`.
3. Walk through all prompts (pick defaults).
4. At the "¿Ejecutar?" prompt, answer **No** (we are not testing the actual transcription here, that's exercised by existing CLI tests).
5. At "¿Otra transcripción?" answer Yes.
6. Paste a playlist URL: `https://www.youtube.com/playlist?list=PLxxx`.
7. Verify it detects `youtube_playlist` and asks playlist questions.
8. Cancel at the run confirmation.
9. Answer No to "¿Otra?".
10. Process should exit cleanly with code 0.

Expected: no crashes, all prompts have visible tooltips/instructions, validation rules work (try setting `--post-kits=Yes`, the preview should also include `--summarize`).

- [ ] **Step 4: Verify `ytt` from a different cwd**

```bash
cd ~ && ytt
```

Expected: same TUI as above, cwd of the user's shell remains `~` after exit.

- [ ] **Step 5: Final commit (if anything pending)**

```bash
cd ~/IdeaProjects/yt-transcriber
git status
```

If clean: done. If something pending (e.g. `.env` was tracked and modified), evaluate per Task 8 step 3 and commit only if appropriate.

---

## Out-of-scope reminders

These were explicitly excluded in the spec; do NOT add them:

- Edición del `.env` desde la TUI.
- Modo batch (lista de URLs desde archivo).
- Historial de transcripciones recientes.
- Override de modelos por sesión.
- Theming, paginación, búsqueda en outputs.
- Notificaciones desktop.
- Empaquetado a `.exe`.
