# TUI yt-transcriber — UX refresh (V2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute task-by-task.

**Goal:** Refactor the existing TUI presentation layer to use `rich` for clear sections, line-separated tooltips, numbered steps, and a structured pre-run summary. Cosmetic-only — no logic changes.

**Architecture:** All changes inside `yt_transcriber/tui.py`. Adds a thin presentation layer on top of existing `questionary` prompts. Uses `rich` (already in deps).

**Tech Stack:** `rich>=13.7.0` (existing), `questionary>=2.0.0` (existing).

**Spec:** `docs/superpowers/specs/2026-04-30-tui-yt-transcriber-design.md` § "V2: UX/Presentation refresh".

---

## File Structure

| File | Action | Notes |
|---|---|---|
| `yt_transcriber/tui.py` | modify | Add helpers; refactor every prompt and runner; new banner/sections; structured summary block |
| `tests/yt_transcriber/test_tui.py` | unchanged | Existing 24 tests cover `detect_input_type`, `apply_validation_rules`, `format_command_preview`, smoke imports — none assert formatting; all keep passing |

No new files. No test changes required.

---

## Task 1: Helpers + prompt refactor

**Files:**
- Modify: `yt_transcriber/tui.py`

- [ ] **Step 1: Add rich imports + helpers at the top of the file**

Locate the imports block (after `from urllib.parse import ...`) and add:

```python
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

console = Console()
```

After all the `_T_*` tooltip constants and before `prompt_input_url`, insert the helpers block:

```python
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
```

- [ ] **Step 2: Refactor `prompt_input_url`**

Replace the existing function with:

```python
def prompt_input_url() -> str:
    """Ask for the input URL/path. Returns stripped non-empty string or raises KeyboardInterrupt."""
    while True:
        _hint(_T_URL)
        value = questionary.text("URL o ruta:").unsafe_ask()
        if value and value.strip():
            console.print()
            return value.strip()
```

Key change: `_hint(_T_URL)` above the prompt; `instruction=` argument removed; blank line after a valid response.

- [ ] **Step 3: Refactor `prompt_transcribe_options`**

Replace the existing function with:

```python
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
```

- [ ] **Step 4: Refactor `prompt_playlist_options`**

Replace the existing function with:

```python
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
```

- [ ] **Step 5: Simplify `prompt_run_confirmation` and update `prompt_run_again`**

Replace both with:

```python
def prompt_run_confirmation() -> bool:
    """Ask for go/no-go. The pre-run summary is printed by the caller."""
    return questionary.confirm("¿Ejecutar?", default=True).unsafe_ask()


def prompt_run_again() -> bool:
    """Ask if the user wants another run. Default Yes for chained workflows."""
    return questionary.confirm("¿Otra transcripción?", default=True).unsafe_ask()
```

`prompt_run_confirmation` no longer takes a `preview` argument — it is printed by `_run_transcribe` / `_run_playlist` as part of the "Resumen" section (see Task 2). The smoke test `test_prompt_functions_exist` only checks `callable(...)` so the signature change is safe.

- [ ] **Step 6: Run tests + smoke import**

```bash
.venv/Scripts/python.exe -m pytest tests/yt_transcriber/test_tui.py -v
.venv/Scripts/python.exe -c "from yt_transcriber.tui import (
    prompt_input_url, prompt_transcribe_options, prompt_playlist_options,
    prompt_run_confirmation, prompt_run_again, _banner, _section, _hint,
    _success, _skip, _warn, _info_line
); print('imports ok')"
```

Expected: all 24 tests in `test_tui.py` PASS; smoke prints `imports ok`.

If any test fails: STOP and report — likely the `instruction=` removal broke an assertion that was checking it, but per audit none of the existing tests assert that.

- [ ] **Step 7: Commit**

```bash
git add yt_transcriber/tui.py
git commit -m "feat(tui): add rich presentation helpers and refactor prompts (V2 step 1)"
```

---

## Task 2: Refactor runners, results, and main loop

**Files:**
- Modify: `yt_transcriber/tui.py`

- [ ] **Step 1: Refactor `_print_transcribe_results`**

Replace the existing function with:

```python
def _print_transcribe_results(result: tuple) -> None:
    """Pretty-print the tuple returned by run_transcribe_command."""
    transcript_path, summary_en, summary_es, post_kits = result
    if not any(result):
        _warn("No se generó ningún archivo. Revisa los logs arriba.")
        return
    if transcript_path:
        _success(f"Transcript:    {transcript_path}")
    if summary_en:
        _success(f"Summary EN:    {summary_en}")
    if summary_es:
        _success(f"Summary ES:    {summary_es}")
    if post_kits:
        _success(f"Post kits:     {post_kits}")
```

- [ ] **Step 2: Refactor `_print_playlist_results`**

Replace the existing function with:

```python
def _print_playlist_results(stats: dict) -> None:
    """Pretty-print the dict returned by run_playlist_command.

    Real per-video counters (after the playlist refactor): `successful`, `failed`,
    `files`, optional `error`.
    """
    successful = stats.get("successful", 0)
    failed = stats.get("failed", 0)
    files = stats.get("files", [])
    error = stats.get("error")

    total = successful + failed
    if total == 0 and not error:
        _skip("Playlist vacía o sin trabajo nuevo.")
        return

    _success(f"Playlist procesada — {successful}/{total} videos con éxito.")
    if failed:
        _warn(f"{failed} fallos. Revisa los logs arriba.")
    if error:
        _warn(f"Error fatal: {error}")
    if files:
        console.print()
        console.print(f"  [bold]Archivos generados ({len(files)}):[/bold]")
        for f in files[:10]:
            _success(str(f))
        if len(files) > 10:
            console.print(f"  [dim]... y {len(files) - 10} más[/dim]")
```

- [ ] **Step 3: Refactor `_run_transcribe`**

Replace the existing function with:

```python
def _run_transcribe(url: str, input_type: InputType) -> None:
    from yt_transcriber.cli import run_transcribe_command

    _section("Opciones de transcripción")
    options = prompt_transcribe_options(input_type)
    options = apply_validation_rules(options)

    _section("Resumen")
    _info_line("Subcomando:", "transcribe")
    _info_line("URL:", url)
    _info_line("Idioma:", options["language"] or "auto-detect")
    _info_line("Summarize:", "sí" if options["summarize"] else "no")
    _info_line("Post kits:", "sí" if options["post_kits"] else "no")
    _info_line("Segments:", "sí" if options["segments"] else "no")
    visual_label = (
        "sí" if options["visual_evidence"]
        else ("no" if input_type == InputType.LOCAL else "n/a (URL)")
    )
    _info_line("Visual:", visual_label)
    console.print()
    console.print("  [bold]CLI equivalente:[/bold]")
    console.print(f"    [dim]{format_command_preview('transcribe', url, options)}[/dim]")
    console.print()

    if not prompt_run_confirmation():
        _warn("Cancelado.")
        return

    _section("Ejecutando")
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

    _section("Resultado")
    _print_transcribe_results(result)
```

- [ ] **Step 4: Refactor `_run_playlist`**

Replace the existing function with:

```python
def _run_playlist(url: str) -> None:
    from yt_transcriber.cli import run_playlist_command

    _section("Opciones de playlist")
    options = prompt_playlist_options()
    options = apply_validation_rules(options)

    _section("Resumen")
    _info_line("Subcomando:", "playlist")
    _info_line("URL:", url)
    _info_line("Límite:", str(options["limit"]) if options["limit"] is not None else "todos")
    _info_line("Idioma:", options["language"])
    _info_line("Summarize:", "sí" if options["summarize"] else "no")
    _info_line("Post kits:", "sí" if options["post_kits"] else "no")
    console.print()
    console.print("  [bold]CLI equivalente:[/bold]")
    console.print(f"    [dim]{format_command_preview('playlist', url, options)}[/dim]")
    console.print()

    if not prompt_run_confirmation():
        _warn("Cancelado.")
        return

    _section("Ejecutando")
    stats = run_playlist_command(
        url=url,
        limit=options["limit"],
        language=options["language"],
        generate_summary=options["summarize"],
        generate_post_kits=options["post_kits"],
    )

    _section("Resultado")
    _print_playlist_results(stats)
```

- [ ] **Step 5: Refactor `main`**

Replace the existing function with:

```python
_INPUT_TYPE_LABELS = {
    InputType.YOUTUBE_VIDEO: "YouTube video",
    InputType.YOUTUBE_PLAYLIST: "YouTube playlist",
    InputType.DRIVE: "Google Drive",
    InputType.LOCAL: "Archivo local",
}


def main() -> int:
    """TUI entry point. Returns exit code."""
    _banner()

    while True:
        try:
            _section("Input")
            url = prompt_input_url()
            input_type = detect_input_type(url)

            if input_type == InputType.UNKNOWN:
                _warn(f"No reconozco '{url}' como YouTube/Drive/archivo local. Reintenta.")
                continue

            type_label = _INPUT_TYPE_LABELS.get(input_type, input_type.value)
            _success(f"Detectado: {type_label}")

            if input_type == InputType.YOUTUBE_PLAYLIST:
                _run_playlist(url)
            else:
                _run_transcribe(url, input_type)

            console.print()
            if not prompt_run_again():
                console.print("[dim]Hasta luego.[/dim]")
                return 0

        except KeyboardInterrupt:
            console.print()
            _warn("Cancelado por el usuario.")
            return 130
        except Exception as e:
            _warn(f"Error inesperado: {type(e).__name__}: {e}")
            console.print("[dim]    Continuando con el loop. Ctrl+C para salir.[/dim]")
            continue


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Run full suite + smoke**

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -c "from yt_transcriber.tui import main; print(main)"
```

Expected: `283 passed, 3 skipped` (or equivalent — same baseline as before this task). Smoke prints `<function main at 0x...>`.

- [ ] **Step 7: Commit**

```bash
git add yt_transcriber/tui.py
git commit -m "feat(tui): rich-styled sections, summary panel and result blocks (V2 step 2)"
```

---

## Task 3: Verification

**Files:** none (read-only validation)

- [ ] **Step 1: Full pytest**

```bash
.venv/Scripts/python.exe -m pytest -v 2>&1 | tail -40
```

Expected: 283 passed, 3 skipped. No regressions.

- [ ] **Step 2: Lint check**

```bash
.venv/Scripts/python.exe -m ruff check yt_transcriber/tui.py tests/yt_transcriber/test_tui.py
```

Expected: clean or only pre-existing `UP042` (`enum.StrEnum` suggestion). New issues introduced by V2 must be fixed via `ruff check --fix yt_transcriber/tui.py`.

- [ ] **Step 3: Visual smoke (manual, requires user)**

```bash
.venv/Scripts/python.exe -m yt_transcriber.tui
```

Walk through:
1. Paste a YouTube video URL.
2. Verify banner, "Input" section, "Detectado: YouTube video" with green ✓.
3. Walk all 5 prompts; confirm `[N/5]` prefix, dim tooltip above, `(Y/n)` indicator visible.
4. Confirm "Resumen" block lists all 7 fields + CLI equivalente.
5. Cancel at "¿Ejecutar?" (No).
6. Verify cancellation message in yellow.
7. Confirm "¿Otra transcripción?" appears with default Yes.
8. Try an invalid input (`https://example.com`) — verify warning and re-prompt.
9. Try a YouTube playlist URL — verify "YouTube playlist" detection and 4-step playlist flow.
10. Press Ctrl+C anywhere — verify clean exit with `Cancelado por el usuario.` warning.

If anything renders broken (encoding glitches, characters not visible, alignment off), report it as DONE_WITH_CONCERNS so we can decide a tweak.

- [ ] **Step 4: No commit needed**

Verification only.

---

## Out-of-scope reminders

These are explicitly out of scope (per spec V2):

- Spinners during execution (Whisper has tqdm).
- Tabular `rich.Table` for the summary (bullet list is enough).
- Themes / dark-light toggle.
- Internationalization.
- Configurable icons.
