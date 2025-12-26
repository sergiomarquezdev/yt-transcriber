# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Install dependencies (editable mode)
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Run the CLI
yt-transcriber transcribe --url "https://www.youtube.com/watch?v=VIDEO_ID"
yt-transcriber transcribe --url "URL" --summarize          # With AI summaries
yt-transcriber transcribe --url "URL" --post-kits          # With LinkedIn/Twitter content

# Run all tests
pytest

# Run specific test file
pytest tests/core/test_llm.py

# Run specific test
pytest tests/core/test_llm.py::test_function_name -v

# Linting and type checking
ruff check .
ruff format .
mypy .
```

## Architecture Overview

### Two-Package Structure

- **`core/`** - Shared infrastructure used across pipelines:
  - `settings.py` - Pydantic-validated config from `.env` (AppSettings singleton)
  - `llm.py` - Multi-provider LLM dispatch (Gemini/OpenAI/Anthropic) with caching
  - `cache.py` - LLM response cache (in-memory LRU + disk persistence)
  - `media_downloader.py` - yt-dlp wrapper for YouTube/Google Drive/local files
  - `media_transcriber.py` - Whisper transcription wrapper
  - `translator.py` - Summary translation (EN→ES)

- **`yt_transcriber/`** - Main transcription pipeline:
  - `cli.py` - argparse CLI entry point, delegates to service
  - `service.py` - Orchestrates: Download → Transcribe → Summarize → Translate → Post Kits
  - `summarizer.py` - Gemini-based video summarization with structured output
  - `post_kits_generator.py` - LinkedIn posts + Twitter threads from summaries
  - `whisper_context.py` - Context manager for Whisper model lifecycle (memory cleanup)

### Key Data Flow

```
CLI/API → service.process_transcription()
  1. download_and_extract_audio() → WAV file
  2. transcribe_audio_file() → TranscriptionResult
  3. generate_summary() → VideoSummary (if --summarize)
  4. ScriptTranslator.translate_summary() → Spanish version
  5. generate_post_kits() → PostKits (if --post-kits)
```

### LLM Provider System

Models support provider prefixes: `"gemini:model"`, `"openai:model"`, `"anthropic:model"`. Default is Gemini. Configuration in `settings.py`:
- `SUMMARIZER_MODEL` - For video summarization (default: gemini-2.5-flash)
- `TRANSLATOR_MODEL` - For translations (default: gemini-2.5-flash-lite)
- `PRO_MODEL` - For premium tasks (default: gemini-2.5-pro)

### Memory Management

Whisper models are loaded/unloaded per video via `whisper_model_context()` to prevent VRAM leaks. Temp files use `tempfile.TemporaryDirectory` for automatic cleanup.

## Configuration

Copy `.env.example` to `.env`. Key settings:
- `GOOGLE_API_KEY` - Required for AI summaries
- `WHISPER_MODEL_NAME` - tiny/base/small/medium/large (default: base)
- `WHISPER_DEVICE` - cpu/cuda (auto-fallback if CUDA unavailable)
- `LLM_CACHE_ENABLED` - Toggle LLM response caching (default: true)

## Output Structure

```
output/
├── transcripts/    # Raw transcription .txt files
└── summaries/      # AI summaries and post kits .md files
```

## Testing Patterns

- Tests mock external services (yt-dlp, Whisper, Gemini APIs)
- Use `pytest-mock` for patching
- Fixtures in `tests/conftest.py`
- Test files mirror source structure: `tests/core/`, `tests/yt_transcriber/`
