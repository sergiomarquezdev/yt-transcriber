# Cleanup: post_kits + summary + reorg outputs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove unused `post_kits` and `summary` features (plus orphaned dead code), and reorganize outputs so each video's artifacts live together in `output/{stem}/` instead of being scattered across `output/transcripts/`, `output/summaries/`, `output/analysis/`.

**Architecture:** Pure refactor. The transcription pipeline (Whisper, yt-dlp, ffmpeg, segments JSON, visual evidence) is untouched in behavior. Only changes: (1) `process_transcription` returns `Path | None` instead of a 4-tuple; (2) outputs go to a per-video subdirectory derived from the existing stem `{normalized_title}_vid_{video_id}_job_{timestamp}`; (3) settings / CLI / TUI / tests strip every reference to the removed features.

**Tech Stack:** Python 3.12+, pytest, faster-whisper, yt-dlp, questionary, rich, pydantic-settings.

**Spec:** `docs/superpowers/specs/2026-05-06-cleanup-postkits-summary-reorg-outputs-design.md`

---

## File Map

**Files to DELETE:**
- `yt_transcriber/post_kits_generator.py`
- `yt_transcriber/summarizer.py`
- `yt_transcriber/models.py`
- `core/translator.py`
- `tests/yt_transcriber/test_post_kits_generator.py`
- `tests/yt_transcriber/test_summarizer.py`
- `tests/yt_transcriber/test_models.py`
- `tests/core/test_translator.py`

**Files to MODIFY:**
- `core/settings.py` — remove orphan settings, add `OUTPUT_BASE_DIR`
- `core/models.py` — remove `VideoSummary`, `TimestampedSection` (keep `TranscriptSegment`)
- `yt_transcriber/service.py` — drop `generate_summary_outputs`, change `process_transcription` signature, reorg outputs into per-video dir
- `yt_transcriber/cli.py` — drop `--summarize`/`--post-kits`, update wrappers, ajust playlist loop
- `yt_transcriber/tui.py` — drop summarize/post_kits prompts, update prompt counts and result printer
- `tests/yt_transcriber/test_service.py` — purge summary/post_kits/cache cases, ajust new layout
- `tests/yt_transcriber/test_cli.py` — purge `--summarize`/`--post-kits` paths, ajust new layout
- `tests/yt_transcriber/test_tui.py` — ajust prompt counts and validation rule
- `tests/core/test_settings.py` — purge deleted settings, add OUTPUT_BASE_DIR coverage
- `CLAUDE.md` — drop sections about removed features

**Files to verify mid-plan (delete if orphan):**
- `core/llm.py` and `tests/core/test_llm.py` — keep only if anything still calls `call_llm` / `is_model_configured` after the refactor

**Filesystem to GIT RM:**
- `output/transcripts/`, `output/summaries/`, `output/analysis/`

---

## Phase A — Delete dead modules

Goal: remove modules that have **no remaining productive consumers** before touching anything else. After this phase, the build temporarily breaks (service.py still imports them); we fix that in phase C.

### Task A1: Delete post_kits module + tests

**Files:**
- Delete: `yt_transcriber/post_kits_generator.py`
- Delete: `yt_transcriber/models.py`
- Delete: `tests/yt_transcriber/test_post_kits_generator.py`
- Delete: `tests/yt_transcriber/test_models.py`

- [ ] **Step 1: Confirm no production import outside the deletion set**

Run:
```bash
rg "post_kits_generator|yt_transcriber\.models|LinkedInPost|TwitterThread\b|PostKits\b" --type py
```
Expected: matches only inside files we are about to delete (`post_kits_generator.py`, `yt_transcriber/models.py`, the two test files), plus `service.py` and `cli.py` and `tui.py` (those will be cleaned up in later phases). No other surprises.

- [ ] **Step 2: Delete the four files**

```bash
git rm yt_transcriber/post_kits_generator.py yt_transcriber/models.py tests/yt_transcriber/test_post_kits_generator.py tests/yt_transcriber/test_models.py
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "refactor: remove unused post_kits feature (modules + tests)"
```

### Task A2: Delete summarizer module + tests

**Files:**
- Delete: `yt_transcriber/summarizer.py`
- Delete: `tests/yt_transcriber/test_summarizer.py`

- [ ] **Step 1: Confirm imports outside the deletion set**

Run:
```bash
rg "from yt_transcriber\.summarizer|yt_transcriber\.summarizer" --type py
```
Expected: matches only in `service.py`, `tests/yt_transcriber/test_summarizer.py` (will be deleted), and the `cli.py` reference inside `command_playlist` via `from yt_transcriber.service import generate_summary_outputs` (we cleanup in phase C/D).

- [ ] **Step 2: Delete files**

```bash
git rm yt_transcriber/summarizer.py tests/yt_transcriber/test_summarizer.py
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "refactor: remove unused summarizer feature (module + tests)"
```

### Task A3: Delete translator module + tests

**Files:**
- Delete: `core/translator.py`
- Delete: `tests/core/test_translator.py`

- [ ] **Step 1: Confirm `ScriptTranslator` only consumed by the summary path**

Run:
```bash
rg "ScriptTranslator|from core\.translator|core\.translator" --type py
```
Expected: matches in `core/translator.py` itself, `tests/core/test_translator.py`, and `yt_transcriber/service.py`. No third-party consumer.

- [ ] **Step 2: Delete files**

```bash
git rm core/translator.py tests/core/test_translator.py
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "refactor: remove orphan core.translator (only consumed by deleted summarizer)"
```

---

## Phase B — Clean settings

Goal: strip orphan settings from `core/settings.py` and introduce `OUTPUT_BASE_DIR`. Tests for settings drive this phase (TDD).

### Task B1: Add failing test for OUTPUT_BASE_DIR

**Files:**
- Modify: `tests/core/test_settings.py`

- [ ] **Step 1: Inspect the current test file to see existing patterns**

Run:
```bash
rg "def test_" tests/core/test_settings.py
```
Use the same fixture/style for the new test.

- [ ] **Step 2: Add a new failing test**

Append to `tests/core/test_settings.py`:

```python
def test_output_base_dir_default_is_output(monkeypatch):
    """OUTPUT_BASE_DIR defaults to Path('output/') and replaces the legacy split dirs."""
    # Force a clean re-import of settings so .env / module state doesn't leak
    monkeypatch.delenv("OUTPUT_BASE_DIR", raising=False)
    from importlib import reload

    import core.settings as settings_module

    reload(settings_module)
    assert settings_module.settings.OUTPUT_BASE_DIR == Path("output/")
```

If `from pathlib import Path` is missing at the top of the file, add it.

- [ ] **Step 3: Run and confirm it fails**

Run:
```bash
pytest tests/core/test_settings.py::test_output_base_dir_default_is_output -v
```
Expected: FAIL with `AttributeError: 'AppSettings' object has no attribute 'OUTPUT_BASE_DIR'` (or similar).

### Task B2: Strip orphan settings + introduce OUTPUT_BASE_DIR

**Files:**
- Modify: `core/settings.py`

- [ ] **Step 1: Apply the cleanup**

Replace `core/settings.py` with this (verbatim):

```python
"""Shared application settings for yt-transcriber.

This module contains the validated Pydantic settings for the CLI/TUI.
"""

import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env if present
load_dotenv()


class AppSettings(BaseSettings):
    """Validated application settings."""

    model_config = SettingsConfigDict(case_sensitive=False)

    # ========== WHISPER ==========
    WHISPER_MODEL_NAME: str = Field(
        default="base",
        description="Whisper model (base, small, medium, large-v3, distil-large-v3, ...)",
    )
    WHISPER_DEVICE: Literal["cpu", "cuda", "auto"] = Field(
        default="auto",
        description="Compute device (auto detects CUDA via CTranslate2)",
    )
    WHISPER_COMPUTE_TYPE: str = Field(
        default="default",
        description="CTranslate2 compute type (int8_float16 for GPU, int8 for CPU)",
    )
    WHISPER_BEAM_SIZE: int = Field(
        default=5,
        description="Beam search size (1=greedy, 5=default)",
    )
    WHISPER_VAD_FILTER: bool = Field(
        default=True,
        description="Silero VAD filter to skip silences",
    )

    # ========== PATHS ==========
    TEMP_DOWNLOAD_DIR: Path = Field(
        default=Path("temp_files/"),
        description="Directory for temporary files",
    )
    OUTPUT_BASE_DIR: Path = Field(
        default=Path("output/"),
        description="Base directory; each video gets its own subfolder here",
    )

    # ========== LOGGING / FFMPEG ==========
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    FFMPEG_LOCATION: str = Field(
        default="",
        description="Custom FFmpeg path (optional)",
    )

    # ========== TRANSCRIPT ARTIFACTS ==========
    TRANSCRIPT_SEGMENTS_ENABLED: bool = Field(
        default=False,
        description="Emit timestamped segments JSON sidecar",
    )
    VISUAL_EVIDENCE_ENABLED: bool = Field(
        default=False,
        description="Extract one frame per transcript segment (local files only)",
    )
    VISUAL_EVIDENCE_MIN_SEGMENT_SECONDS: float = Field(
        default=1.0,
        description="Minimum segment duration to consider visual evidence",
    )

    # ========== YT-DLP ==========
    YT_SEARCH_TIMEOUT_SECONDS: int = Field(
        default=180,
        description="Timeout (seconds) for yt-dlp search/info calls",
    )


# Build the global instance
try:
    settings = AppSettings()
except Exception as e:
    sys.stderr.write(f"CRITICAL: Failed to load/validate settings: {e}\n")
    sys.exit(1)


__all__ = ["AppSettings", "settings"]
```

- [ ] **Step 2: Run the new test (should pass)**

Run:
```bash
pytest tests/core/test_settings.py::test_output_base_dir_default_is_output -v
```
Expected: PASS.

- [ ] **Step 3: Run the full settings test file and remove dead test cases**

Run:
```bash
pytest tests/core/test_settings.py -v
```

For every test that fails because it asserts on a deleted setting (`SUMMARY_OUTPUT_DIR`, `TRANSCRIPT_CACHE_*`, `POST_KITS_*`, `SUMMARIZER_MODEL`, `PATTERN_ANALYZER_MODEL`, `QUERY_OPTIMIZER_MODEL`, `SCRIPT_OUTPUT_DIR`, `ANALYSIS_OUTPUT_DIR`, `TEMP_BATCH_DIR`, `TRENDS_OUTPUT_DIR`, `SERPAPI_API_KEY`, `REDDIT_*`, `OUTPUT_TRANSCRIPTS_DIR`, `CLAUDE_CLI_*`, `DEFAULT_LLM_MODEL`, `PRO_MODEL`, `TRANSLATOR_MODEL`), delete the test (the setting no longer exists). Keep tests that cover settings still present.

- [ ] **Step 4: Re-run, confirm green**

Run:
```bash
pytest tests/core/test_settings.py -v
```
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add core/settings.py tests/core/test_settings.py
git commit -m "refactor(settings): drop orphan settings, introduce OUTPUT_BASE_DIR"
```

---

## Phase C — Refactor service.py

Goal: `process_transcription` returns `Path | None`, writes everything for a single video into `output/{stem}/`, and no longer references summary/post-kits/cache/translator. Drive each change with a test.

### Task C1: Failing test for per-video output directory

**Files:**
- Modify: `tests/yt_transcriber/test_service.py`

- [ ] **Step 1: Read the existing fixture so the new test can reuse it**

Run:
```bash
rg "mock_dependencies" tests/yt_transcriber/test_service.py -n
```

- [ ] **Step 2: Update `mock_dependencies` fixture to the new settings shape**

In `tests/yt_transcriber/test_service.py`, replace the `mock_dependencies` fixture (around lines 15-33) with:

```python
@pytest.fixture
def mock_dependencies(self, temp_dir):
    """Set up common mocks for process_transcription."""
    with patch("yt_transcriber.service.settings") as mock_settings:
        mock_settings.TEMP_DOWNLOAD_DIR = temp_dir / "temp"
        mock_settings.OUTPUT_BASE_DIR = temp_dir / "output"
        mock_settings.TRANSCRIPT_SEGMENTS_ENABLED = False
        mock_settings.VISUAL_EVIDENCE_ENABLED = False
        mock_settings.VISUAL_EVIDENCE_MIN_SEGMENT_SECONDS = 1.0

        (temp_dir / "temp").mkdir(exist_ok=True)
        (temp_dir / "output").mkdir(exist_ok=True)

        yield mock_settings
```

- [ ] **Step 3: Add a new failing test below `class TestProcessTranscription`**

```python
def test_transcript_lives_in_per_video_subdir(
    self, mock_dependencies, mock_whisper_model, temp_dir
):
    """The transcript .txt is written under output/<stem>/<stem>.txt — never directly under output/."""
    with patch("yt_transcriber.service.download_and_extract_audio") as mock_dl, patch(
        "yt_transcriber.service.transcribe_audio_file"
    ) as mock_tr:
        mock_dl.return_value = MagicMock(
            audio_path=temp_dir / "fake.wav",
            video_path=None,
            video_id="ABC123",
        )
        mock_tr.return_value = MagicMock(
            text="hello world",
            language="en",
            segments=[],
        )

        result = process_transcription(
            youtube_url="https://www.youtube.com/watch?v=ABC123",
            title="Example Video",
            model=mock_whisper_model,
        )

        assert result is not None, "expected a Path, got None"
        assert result.suffix == ".txt"
        # The transcript must sit one level deep: output/<stem>/<stem>.txt
        assert result.parent.parent == mock_dependencies.OUTPUT_BASE_DIR
        assert result.parent.name == result.stem
        assert "ABC123" in result.stem
```

- [ ] **Step 4: Run the new test (it should fail because process_transcription still has the old layout)**

Run:
```bash
pytest tests/yt_transcriber/test_service.py::TestProcessTranscription::test_transcript_lives_in_per_video_subdir -v
```
Expected: FAIL.

### Task C2: Refactor `process_transcription` to new layout + new signature

**Files:**
- Modify: `yt_transcriber/service.py`

- [ ] **Step 1: Replace `yt_transcriber/service.py` with the new implementation**

Write `yt_transcriber/service.py` containing exactly:

```python
"""Transcriber Service - shared orchestration for the transcription pipeline.

This module centralizes the end-to-end flow used by both CLI and TUI:
Download -> Transcribe -> Save (transcript + optional segments + optional frames)

CLI keeps a thin wrapper that forwards to this service for testability and
backward compatibility (tests patch functions in yt_transcriber.cli).
"""

import logging
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from core.media_downloader import (
    DownloadError,
    download_and_extract_audio,
    extract_audio_from_local_file,
)
from core.media_transcriber import TranscriptionError, transcribe_audio_file
from core.settings import settings
from yt_transcriber import utils

logger = logging.getLogger(__name__)


def _resolve_segments_and_visual(
    segments_override: bool | None,
    visual_override: bool | None,
) -> tuple[bool, bool]:
    """Resolve effective segment/visual toggles with CLI/env precedence."""
    visual_enabled = (
        settings.VISUAL_EVIDENCE_ENABLED if visual_override is None else visual_override
    )

    if segments_override is not None:
        segments_enabled = segments_override
    elif visual_override is True:
        segments_enabled = True
    else:
        segments_enabled = settings.TRANSCRIPT_SEGMENTS_ENABLED

    return segments_enabled, visual_enabled


def _extract_visual_evidence(
    video_path: Path,
    segments: list[Any] | None,
    output_filename_base: str,
    output_dir: Path,
    ffmpeg_location: str | None,
) -> list[Path]:
    """Extract one frame (midpoint) per eligible transcript segment."""
    if not segments:
        logger.debug("No segments available for visual evidence extraction")
        return []

    min_duration = settings.VISUAL_EVIDENCE_MIN_SEGMENT_SECONDS
    ffmpeg_bin = ffmpeg_location or "ffmpeg"
    extracted_paths: list[Path] = []

    for idx, segment in enumerate(segments):
        duration = float(segment.end) - float(segment.start)
        if duration < min_duration:
            logger.debug(
                "Skipping visual evidence for segment %s: duration %.3fs < %.3fs",
                idx,
                duration,
                min_duration,
            )
            continue

        midpoint = float(segment.start) + (duration / 2.0)
        frame_path = output_dir / f"{output_filename_base}_frame_{idx}.jpg"
        cmd = [
            ffmpeg_bin,
            "-y",
            "-ss",
            f"{midpoint:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(frame_path),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            extracted_paths.append(frame_path)
        except Exception as e:
            logger.warning(
                "Could not extract visual evidence frame for segment %s (continuing): %s",
                idx,
                e,
            )

    return extracted_paths


def process_transcription(
    youtube_url: str,
    title: str,
    model: Any,
    language: str | None = None,
    ffmpeg_location: str | None = None,
    segments_override: bool | None = None,
    visual_override: bool | None = None,
) -> Path | None:
    """Download/extract -> transcribe -> save into a per-video output folder.

    Args:
        youtube_url: YouTube URL, Google Drive URL, or local file path.
        title: Video title (inferred from filename for local files when empty).
        model: Preloaded Whisper model.
        language: Optional language code; None = auto-detect.
        ffmpeg_location: Optional FFmpeg path.
        segments_override: Optional CLI override for segments JSON sidecar.
        visual_override: Optional CLI override for visual evidence extraction.

    Returns:
        Path to the saved transcript .txt, or None on failure.
    """
    segments_enabled, visual_enabled = _resolve_segments_and_visual(
        segments_override=segments_override,
        visual_override=visual_override,
    )

    # Detect input type
    is_local_file = False
    local_file_path: Path | None = None
    is_drive_url = False

    if not (youtube_url.startswith("http://") or youtube_url.startswith("https://")):
        potential_path = Path(youtube_url)
        if potential_path.exists() and potential_path.is_file():
            is_local_file = True
            local_file_path = potential_path
            logger.info(f"Detected local file: {local_file_path}")
            if not title or not title.strip():
                title = local_file_path.stem
    else:
        from core.media_downloader import is_google_drive_url

        is_drive_url = is_google_drive_url(youtube_url)
        if is_drive_url:
            logger.info(f"Detected Google Drive URL: {youtube_url}")

    if is_local_file:
        logger.info(f"Starting transcription for local file: {local_file_path}")
    elif is_drive_url:
        logger.info(f"Starting transcription for Google Drive file: {youtube_url}")
    else:
        logger.info(f"Starting transcription for URL: {youtube_url}")

    unique_job_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    settings.TEMP_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory(
            dir=settings.TEMP_DOWNLOAD_DIR, prefix=f"{unique_job_id}_"
        ) as temp_dir_str:
            job_temp_dir = Path(temp_dir_str)
            logger.info(f"Using temp dir: {job_temp_dir}")

            # 1) Download / extract audio
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

            # 3) Build per-video output dir
            normalized_title = utils.normalize_title_for_filename(title)
            video_id = download_result.video_id or "local"
            output_filename_base = f"{normalized_title}_vid_{video_id}_job_{unique_job_id}"
            output_dir = settings.OUTPUT_BASE_DIR / output_filename_base
            output_dir.mkdir(parents=True, exist_ok=True)

            # 4) Save transcript
            logger.info("Step 3: Saving transcript...")
            transcript_path = utils.save_transcription_to_file(
                transcription_text=transcription_result.text,
                output_filename_no_ext=output_filename_base,
                output_dir=output_dir,
                original_title=title,
            )

            if not transcript_path:
                raise OSError("Could not save transcript file.")

            logger.info(f"Transcript saved successfully to: {transcript_path}")
            print(f"\nTranscript saved to: {transcript_path}")

            # 5) Optional segments JSON
            if segments_enabled:
                segments_path = utils.derive_sibling_path(transcript_path, "_segments.json")
                utils.save_segments_json(
                    segments=transcription_result.segments,
                    language=transcription_result.language,
                    output_path=segments_path,
                )

            # 6) Optional visual evidence (local files only)
            if visual_enabled:
                if not is_local_file or not local_file_path:
                    logger.warning(
                        "Visual evidence is only supported for local files in V1; skipping extraction."
                    )
                else:
                    _extract_visual_evidence(
                        video_path=local_file_path,
                        segments=transcription_result.segments,
                        output_filename_base=output_filename_base,
                        output_dir=output_dir,
                        ffmpeg_location=ffmpeg_location,
                    )

            return transcript_path

    except (OSError, DownloadError, TranscriptionError) as e:
        logger.error(f"An error occurred during processing: {e}", exc_info=True)
        print(f"\nError: {e}", file=sys.stderr)
        return None
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        return None
```

- [ ] **Step 2: Run the new test (should now pass)**

Run:
```bash
pytest tests/yt_transcriber/test_service.py::TestProcessTranscription::test_transcript_lives_in_per_video_subdir -v
```
Expected: PASS.

### Task C3: Purge dead service tests + ajust the surviving ones

**Files:**
- Modify: `tests/yt_transcriber/test_service.py`

- [ ] **Step 1: Run the full service test file and inventory failures**

Run:
```bash
pytest tests/yt_transcriber/test_service.py -v 2>&1 | tee /tmp/service_test_run.txt
```

- [ ] **Step 2: Delete every test that asserts on summary, post_kits, or transcript cache**

Inside `tests/yt_transcriber/test_service.py`:
- Delete any test whose name or body mentions `summary`, `summarize`, `post_kits`, `post-kits`, `cache`, `reuse_transcripts`, `generate_summary_outputs`, `SUMMARY_OUTPUT_DIR`, `TRANSCRIPT_CACHE`.
- Delete imports for `generate_summary_outputs` and any post-kits/summary models.

- [ ] **Step 3: Update surviving tests for new return type and layout**

For every remaining test:
- Replace any `transcript_path, summary_en, summary_es, post_kits = process_transcription(...)` with `transcript_path = process_transcription(...)`.
- Replace assertions on `mock_settings.OUTPUT_TRANSCRIPTS_DIR` with `mock_settings.OUTPUT_BASE_DIR / "<stem>"` where `<stem>` is the expected per-video folder. If the test doesn't care about the exact stem, assert that the parent directory of the transcript is a child of `OUTPUT_BASE_DIR`.
- Drop any keyword arguments removed from the new signature (`generate_post_kits=`, `generate_summary=`, `reuse_transcripts=`).

- [ ] **Step 4: Re-run, iterate until green**

Run:
```bash
pytest tests/yt_transcriber/test_service.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add yt_transcriber/service.py tests/yt_transcriber/test_service.py
git commit -m "refactor(service): per-video output dir; drop summary/post_kits/cache flow"
```

---

## Phase D — Refactor cli.py

Goal: drop `--summarize`/`--post-kits` from both subcommands; update wrappers; ajust the playlist loop so each video's `.txt` lands in its own folder under `OUTPUT_BASE_DIR`.

### Task D1: Failing test for transcribe wrapper return type

**Files:**
- Modify: `tests/yt_transcriber/test_cli.py`

- [ ] **Step 1: Add a failing test that pins the new return type**

Append to `tests/yt_transcriber/test_cli.py`:

```python
def test_run_transcribe_command_returns_single_path(monkeypatch, tmp_path):
    """run_transcribe_command returns just the transcript path string (or None)."""
    from yt_transcriber import cli

    fake_path = tmp_path / "video" / "video.txt"
    fake_path.parent.mkdir(parents=True)
    fake_path.write_text("hi", encoding="utf-8")

    def fake_pt(**kwargs):
        return fake_path

    monkeypatch.setattr(cli, "process_transcription", fake_pt)
    monkeypatch.setattr(cli, "whisper_model_context", lambda: _fake_ctx())

    result = cli.run_transcribe_command(url="https://youtu.be/abc")
    assert result == str(fake_path)


class _fake_ctx:
    def __enter__(self):
        return object()

    def __exit__(self, *exc):
        return False
```

(If `_fake_ctx` clashes with an existing helper, rename it locally.)

- [ ] **Step 2: Run, confirm it fails**

Run:
```bash
pytest tests/yt_transcriber/test_cli.py::test_run_transcribe_command_returns_single_path -v
```
Expected: FAIL (current `run_transcribe_command` returns a 4-tuple).

### Task D2: Refactor cli.py

**Files:**
- Modify: `yt_transcriber/cli.py`

- [ ] **Step 1: Apply the cleanup**

Replace `yt_transcriber/cli.py` with:

```python
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
    from core.settings import settings as app_settings
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
        entry_dir = app_settings.OUTPUT_BASE_DIR / stem
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
                # Cleanup empty per-video folder so the user doesn't see ghosts
                try:
                    entry_dir.rmdir()
                except OSError:
                    pass
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
```

- [ ] **Step 2: Run the new test (should pass)**

Run:
```bash
pytest tests/yt_transcriber/test_cli.py::test_run_transcribe_command_returns_single_path -v
```
Expected: PASS.

### Task D3: Purge dead CLI tests + ajust survivors

**Files:**
- Modify: `tests/yt_transcriber/test_cli.py`

- [ ] **Step 1: Run the full file**

Run:
```bash
pytest tests/yt_transcriber/test_cli.py -v
```

- [ ] **Step 2: Delete tests for removed flags / behavior**

Delete every test in `tests/yt_transcriber/test_cli.py` that references `--summarize`, `--post-kits`, `summarize=`, `post_kits=`, `generate_summary`, `generate_post_kits`, `reuse_transcripts`, `summary_path_en`, `summary_path_es`, `post_kits_path`. Drop now-unused imports.

- [ ] **Step 3: Update survivors for new playlist layout**

For surviving `command_playlist` / `run_playlist_command` tests:
- Replace any assertion on `OUTPUT_TRANSCRIPTS_DIR` with `OUTPUT_BASE_DIR / <stem>` (where `<stem>` follows the `_vid_<id>_job_<ts>` pattern, or use a regex `re.match` if the timestamp is not pinned).
- Update `mock_settings` fixtures to set `OUTPUT_BASE_DIR` instead of `OUTPUT_TRANSCRIPTS_DIR`.

For surviving `run_transcribe_command` tests:
- Adjust expected return type from a 4-tuple to a single `str | None`.

- [ ] **Step 4: Re-run, iterate until green**

Run:
```bash
pytest tests/yt_transcriber/test_cli.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add yt_transcriber/cli.py tests/yt_transcriber/test_cli.py
git commit -m "refactor(cli): drop --summarize/--post-kits, per-video output dir for playlists"
```

---

## Phase E — Refactor tui.py

Goal: drop summarize/post_kits prompts; transcribe now asks 3 questions, playlist asks 2; result printer takes a `str | None`.

### Task E1: Failing test for new transcribe options shape

**Files:**
- Modify: `tests/yt_transcriber/test_tui.py`

- [ ] **Step 1: Add a failing test**

Append to `tests/yt_transcriber/test_tui.py`:

```python
def test_transcribe_options_have_no_summarize_or_post_kits(monkeypatch):
    """prompt_transcribe_options no longer asks about summarize/post_kits."""
    from yt_transcriber import tui

    answers_iter = iter([
        # [1/3] language
        type("Q", (), {"unsafe_ask": lambda self: "es"})(),
        # [2/3] segments
        type("Q", (), {"unsafe_ask": lambda self: False})(),
        # [3/3] visual evidence (only asked for LOCAL)
        type("Q", (), {"unsafe_ask": lambda self: False})(),
    ])

    def fake_select(*args, **kwargs):
        return next(answers_iter)

    def fake_confirm(*args, **kwargs):
        return next(answers_iter)

    monkeypatch.setattr("questionary.select", fake_select)
    monkeypatch.setattr("questionary.confirm", fake_confirm)

    options = tui.prompt_transcribe_options(tui.InputType.LOCAL)

    assert "summarize" not in options
    assert "post_kits" not in options
    assert options["language"] == "es"
    assert options["segments"] is False
    assert options["visual_evidence"] is False
```

- [ ] **Step 2: Run, confirm it fails**

Run:
```bash
pytest tests/yt_transcriber/test_tui.py::test_transcribe_options_have_no_summarize_or_post_kits -v
```
Expected: FAIL (current TUI still asks for summarize/post_kits).

### Task E2: Refactor tui.py

**Files:**
- Modify: `yt_transcriber/tui.py`

- [ ] **Step 1: Edit `_T_SUMMARIZE` and `_T_POST_KITS`**

In `yt_transcriber/tui.py`, delete the two lines:
```python
_T_SUMMARIZE = "Genera resúmenes EN + ES con Claude (incrementa tiempo y consume cuota Claude)"
_T_POST_KITS = "LinkedIn post + Twitter thread. Activa --summarize automáticamente."
```

- [ ] **Step 2: Replace `prompt_transcribe_options`**

Replace the body of `prompt_transcribe_options` with:

```python
def prompt_transcribe_options(input_type: InputType) -> dict:
    """Ask all transcribe-flow questions. Returns dict with options (pre-validation)."""
    total = 3

    # [1/3] Language
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

    # [2/3] Segments
    _hint(_T_SEGMENTS)
    segments = questionary.confirm(
        f"[2/{total}] ¿Sidecar de segmentos JSON?",
        default=False,
    ).unsafe_ask()
    console.print()

    # [3/3] Visual evidence (only if local)
    if input_type == InputType.LOCAL:
        _hint(_T_VISUAL_EVIDENCE)
        visual_evidence = questionary.confirm(
            f"[3/{total}] ¿Extraer frames clave (visual evidence)?",
            default=False,
        ).unsafe_ask()
        console.print()
    else:
        _skip(f"[3/{total}] Visual evidence: omitido (no aplica para URLs)")
        console.print()
        visual_evidence = False

    return {
        "language": language,
        "segments": segments,
        "visual_evidence": visual_evidence,
    }
```

- [ ] **Step 3: Replace `prompt_playlist_options`**

```python
def prompt_playlist_options() -> dict:
    """Ask all playlist-flow questions. Returns dict with options (pre-validation)."""
    total = 2

    # [1/2] Limit
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

    # [2/2] Language
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

    return {
        "limit": limit,
        "language": language,
    }
```

- [ ] **Step 4: Update `apply_validation_rules`**

Replace its body with:

```python
def apply_validation_rules(options: dict) -> dict:
    """Enforce CLI cross-flag implications on a copy of `options`.

    Rule: `visual_evidence=True` forces `segments=True`.
    """
    result = dict(options)
    if result.get("visual_evidence"):
        result["segments"] = True
    return result
```

- [ ] **Step 5: Update `format_command_preview`**

Replace its body with:

```python
def format_command_preview(subcommand: str, url: str, options: dict) -> str:
    """Build a human-readable equivalent CLI command for the user to confirm."""
    parts = ["yt-transcriber", subcommand, "-u", f'"{url}"']

    lang = options.get("language")
    if lang:
        parts.extend(["--language", lang])

    if subcommand == "playlist":
        limit = options.get("limit")
        if limit is not None:
            parts.extend(["--limit", str(limit)])

    if subcommand == "transcribe":
        if options.get("segments"):
            parts.append("--segments")
        if options.get("visual_evidence"):
            parts.append("--visual-evidence")

    return " ".join(parts)
```

- [ ] **Step 6: Update `_print_transcribe_results`**

Replace with:

```python
def _print_transcribe_results(transcript_path: str | None) -> None:
    """Pretty-print the path returned by run_transcribe_command."""
    if not transcript_path:
        _warn("No se generó ningún archivo. Revisa los logs arriba.")
        return
    _success(f"Transcript: {transcript_path}")
```

- [ ] **Step 7: Update `_run_transcribe` and `_run_playlist`**

Inside `_run_transcribe`, the "Resumen" block:
- Remove `_info_line("Summarize:", ...)` and `_info_line("Post kits:", ...)`.
The `run_transcribe_command` call should pass only the surviving kwargs:
```python
result = run_transcribe_command(
    url=url,
    language=options["language"],
    ffmpeg_location=None,
    segments_override=options["segments"],
    visual_override=options["visual_evidence"],
)
```

Inside `_run_playlist`, the "Resumen" block:
- Remove `_info_line("Summarize:", ...)` and `_info_line("Post kits:", ...)`.
The `run_playlist_command` call:
```python
stats = run_playlist_command(
    url=url,
    limit=options["limit"],
    language=options["language"],
)
```

- [ ] **Step 8: Run the new TUI test**

Run:
```bash
pytest tests/yt_transcriber/test_tui.py::test_transcribe_options_have_no_summarize_or_post_kits -v
```
Expected: PASS.

### Task E3: Purge dead TUI tests + ajust survivors

**Files:**
- Modify: `tests/yt_transcriber/test_tui.py`

- [ ] **Step 1: Run all TUI tests**

Run:
```bash
pytest tests/yt_transcriber/test_tui.py -v
```

- [ ] **Step 2: Delete tests that assume the old prompt sequence**

Delete tests asserting on `summarize`, `post_kits`, the old `[1/5]..[5/5]` numbering, or the `apply_validation_rules` rule `post_kits → summarize`.

- [ ] **Step 3: Update survivors**

- Tests on `apply_validation_rules`: keep only the case `visual_evidence=True` forces `segments=True`. Drop the `post_kits=True` case.
- Tests on `format_command_preview`: drop expectations of `--summarize`/`--post-kits`.
- Tests that mock `prompt_transcribe_options` answers: update count to 3, removing summarize/post_kits answers.
- Tests that mock `prompt_playlist_options`: update count to 2.
- Tests on `_print_transcribe_results`: pass a single string path instead of a 4-tuple.

- [ ] **Step 4: Re-run, iterate until green**

Run:
```bash
pytest tests/yt_transcriber/test_tui.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add yt_transcriber/tui.py tests/yt_transcriber/test_tui.py
git commit -m "refactor(tui): 3/2-question flow; drop summarize/post_kits prompts"
```

---

## Phase F — Trim core models, verify orphans, full suite

### Task F1: Remove `VideoSummary` and `TimestampedSection`

**Files:**
- Modify: `core/models.py`

- [ ] **Step 1: Confirm no production import survives**

Run:
```bash
rg "VideoSummary|TimestampedSection" --type py
```
Expected: zero matches outside `core/models.py` itself (we already removed translator/summarizer; if anything still references these, fix that first).

- [ ] **Step 2: Replace `core/models.py` with the trimmed version**

```python
"""Shared data models for yt-transcriber."""

from dataclasses import dataclass


@dataclass
class TranscriptSegment:
    """Structured transcript segment with timing metadata."""

    start: float
    end: float
    text: str

    def to_dict(self) -> dict:
        """Serialize segment for JSON sidecar output."""
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }


__all__ = ["TranscriptSegment"]
```

- [ ] **Step 3: Run model tests**

Run:
```bash
pytest tests/core/test_models.py -v
```
Delete any test that referenced `VideoSummary` or `TimestampedSection`. Re-run until green.

- [ ] **Step 4: Commit**

```bash
git add core/models.py tests/core/test_models.py
git commit -m "refactor(models): drop VideoSummary and TimestampedSection"
```

### Task F2: Verify `core/llm.py` has no consumers — delete if orphan

**Files:**
- Possibly delete: `core/llm.py`, `tests/core/test_llm.py`

- [ ] **Step 1: Grep for consumers**

Run:
```bash
rg "from core\.llm|core\.llm|call_llm\(|is_model_configured" --type py
```

- [ ] **Step 2: Decide**

- If matches are only inside `core/llm.py` itself and `tests/core/test_llm.py`: delete both files.
- If anything else references them (unexpected after this cleanup), stop and inspect; do not delete.

- [ ] **Step 3 (only if deleting): remove files and commit**

```bash
git rm core/llm.py tests/core/test_llm.py
git commit -m "refactor: drop core.llm (no remaining consumers)"
```

### Task F3: Run the full test suite

- [ ] **Step 1: Run pytest end-to-end**

Run:
```bash
pytest -v
```
Expected: all tests pass. If anything fails, fix the failure (likely a stale import or an `OUTPUT_TRANSCRIPTS_DIR` patch that slipped through) and commit the fix in a small follow-up commit.

- [ ] **Step 2: Final dead-reference grep**

Run:
```bash
rg "post_kits|summarizer|VideoSummary|LinkedInPost|TwitterThread|PostKits|TRANSCRIPT_CACHE|ScriptTranslator|SUMMARY_OUTPUT_DIR|SUMMARIZER_MODEL|PATTERN_ANALYZER_MODEL|QUERY_OPTIMIZER_MODEL|SCRIPT_OUTPUT_DIR|ANALYSIS_OUTPUT_DIR|TEMP_BATCH_DIR|TRENDS_OUTPUT_DIR|SERPAPI_API_KEY|REDDIT_CLIENT|OUTPUT_TRANSCRIPTS_DIR" --type py
```
Expected: zero matches. If any survive, hunt them down and remove.

- [ ] **Step 3: Commit any cleanup found in step 2 (if needed)**

```bash
git add -A
git commit -m "refactor: remove final stale references to deleted features"
```

---

## Phase G — Filesystem cleanup + docs

### Task G1: Git-rm legacy output directories

- [ ] **Step 1: Inspect what's there**

Run:
```bash
ls output/ 2>/dev/null && ls output/transcripts/ output/summaries/ output/analysis/ 2>/dev/null
```

- [ ] **Step 2: Remove them**

Run:
```bash
git rm -r --ignore-unmatch output/transcripts/ output/summaries/ output/analysis/
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove legacy output/ subdirectories (transcripts, summaries, analysis)"
```

### Task G2: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Replace `CLAUDE.md` with this content**

```markdown
# CLAUDE.md

YouTube CLI transcriber. Python 3.12+, faster-whisper (CTranslate2). Two subcommands: `transcribe` (Whisper, audio + Whisper transcription) and `playlist` (YouTube auto-subs via yt-dlp, no GPU).

## Output layout

Each invocation writes to its own folder under `OUTPUT_BASE_DIR` (default `output/`):

```
output/<stem>/
├── <stem>.txt                  (always)
├── <stem>_segments.json        (if --segments or TRANSCRIPT_SEGMENTS_ENABLED)
└── <stem>_frame_<n>.jpg ...    (if --visual-evidence on a local file)
```

`<stem> = {normalized_title}_vid_{video_id}_job_{timestamp}`. Reruns produce a new folder; no overwrites.

## Gotchas

- **Whisper context manager is mandatory**: always use `whisper_model_context()` from `whisper_context.py`. It handles CTranslate2 model load/unload + `gc.collect()`. Leaking models exhausts VRAM.
- **Playlist mode skips Whisper entirely**: `command_playlist` downloads YouTube auto-generated subtitles via yt-dlp (`skip_download=True`), cleans SRT/VTT to plain text. No audio, no GPU, no model loading.
- **Segment artifacts are additive and default-off**: `.txt` transcript is always canonical output; `_segments.json` is only emitted when `TRANSCRIPT_SEGMENTS_ENABLED=true` or `--segments` is passed.
- **Visual evidence is V1 local-only**: `--visual-evidence` implies segments, but frame extraction runs only for local-file inputs; URL/Drive/playlist paths skip frames.
- **`noplaylist: True` is hardcoded** in `download_and_extract_audio()`. Playlist support lives in separate functions (`extract_playlist_entries`, `download_auto_subtitles`).

## Error Hierarchy

```
DownloadError          # core/media_downloader.py
TranscriptionError     # core/media_transcriber.py
```

## Cross-File Workflows

**Add a new CLI subcommand:**
1. Add parser in `cli.py:main()` under `subparsers`
2. Create `command_<name>(args)` handler in `cli.py`
3. Add routing in the `if args.command ==` block at the bottom of `main()`

## Testing

- All external services are mocked (yt-dlp, faster-whisper)
- `core/settings.py` loads `.env` at import time via `load_dotenv()`. Tests that need specific env vars must use `monkeypatch.setenv` BEFORE importing settings, or patch `settings` directly
- `conftest.py` has `mock_whisper_model` fixture returning a MagicMock with `.transcribe()` -- use it instead of creating ad-hoc mocks
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: rewrite CLAUDE.md after post_kits/summary cleanup"
```

### Task G3: Smoke test (manual — required before declaring done)

- [ ] **Step 1: Run a real local-file transcription**

Pick a small `.mp4` (under 1 min) and run:
```bash
python -m yt_transcriber.cli transcribe -u <path-to-mp4> --segments
```
Expected:
- Exactly one new directory under `output/` named `<title>_vid_<id>_job_<timestamp>/`.
- Inside: `<stem>.txt` and `<stem>_segments.json`.
- No `output/transcripts/`, `output/summaries/`, or `output/analysis/`.

- [ ] **Step 2: Run a small playlist (1 video, --limit 1)**

```bash
python -m yt_transcriber.cli playlist -u <playlist-url> -n 1 -l es
```
Expected:
- Exactly one new directory under `output/` for the processed video.
- Inside: `<stem>.txt`.

- [ ] **Step 3: Smoke-test the TUI**

```bash
python -m yt_transcriber.tui
```
Expected:
- Transcribe flow asks 3 questions (language, segments, visual evidence) — never summarize/post-kits.
- Playlist flow asks 2 questions (limit, language) — never summarize/post-kits.
- CLI-equivalent preview never contains `--summarize` or `--post-kits`.

- [ ] **Step 4: Final commit (only if any docs/lint touch-ups appeared during smoke test)**

If steps 1-3 surfaced any tweak (e.g., a typo in a hint, an off-by-one in `[i/total]`), fix and commit. Otherwise skip.

```bash
git add -A
git commit -m "chore: post-smoke-test polish"
```

---

## Done criteria (mirrors spec section "Criterio de éxito")

- [ ] `pytest` passes.
- [ ] `python -m yt_transcriber.cli transcribe -u <url>` produces exactly one directory under `output/` containing the `.txt` and nothing else.
- [ ] With `--segments`, the `_segments.json` is in the same directory.
- [ ] With `--visual-evidence` on a local file, the `_frame_*.jpg` files are in the same directory.
- [ ] `python -m yt_transcriber.cli playlist -u <url> -n 3` produces 3 sibling directories under `output/`.
- [ ] TUI never shows summarize/post-kits prompts (3 questions for transcribe, 2 for playlist).
- [ ] Final dead-reference grep returns zero matches.
- [ ] `output/transcripts/`, `output/summaries/`, `output/analysis/` no longer exist.
