# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YouTube Video Transcriber & Summarizer: CLI tool that transforms YouTube videos, Google Drive files, or local media into transcripts and AI-powered summaries with optional social media post generation.

**Stack**: Python 3.13+, Whisper (OpenAI), Gemini AI, yt-dlp, Pydantic Settings

## Commands

```bash
# Install (editable mode)
pip install -e .
pip install -e ".[dev]"  # with dev dependencies

# Run CLI
yt-transcriber transcribe --url "https://youtube.com/watch?v=..." [--language es] [--post-kits]
yt-transcriber summarize --url "..." [--post-kits]

# Testing
pytest                           # all tests
pytest tests/core/              # core module tests
pytest tests/yt_transcriber/    # transcriber tests
pytest -k "test_name"           # single test by name
pytest --cov=core --cov=yt_transcriber  # with coverage

# Linting
ruff check .
ruff check . --fix
mypy .
```

## Architecture

```
yt-transcriber/
├── core/                    # Shared infrastructure (settings, LLM, media ops)
│   ├── settings.py          # Pydantic settings (all env vars)
│   ├── llm.py               # Multi-provider LLM dispatch (Gemini/OpenAI/Anthropic) + caching
│   ├── cache.py             # Content-hash based LLM response cache
│   ├── media_downloader.py  # yt-dlp + FFmpeg audio extraction
│   ├── media_transcriber.py # Whisper transcription wrapper
│   ├── translator.py        # EN->ES translation via LLM
│   └── models.py            # Shared data models (VideoSummary, TimestampedSection)
│
├── yt_transcriber/          # Main application package
│   ├── cli.py               # argparse CLI entry point (transcribe/summarize commands)
│   ├── service.py           # Orchestration: download -> transcribe -> summarize -> translate
│   ├── summarizer.py        # Gemini-powered video summarization
│   ├── post_kits_generator.py # LinkedIn post + Twitter thread generation
│   ├── utils.py             # File operations, title normalization
│   └── models.py            # Application-specific models
│
└── tests/                   # pytest tests mirroring src structure
```

### Data Flow

1. **CLI** (`cli.py`) parses args, loads Whisper model, delegates to service
2. **Service** (`service.py`) orchestrates the pipeline:
   - Download/extract audio via `core.media_downloader`
   - Transcribe via `core.media_transcriber` (Whisper)
   - Generate EN summary via `yt_transcriber.summarizer`
   - Translate to ES via `core.translator`
   - Optional: generate post kits
3. **LLM calls** route through `core.llm.call_gemini_with_cache()` with content-hash caching

### Multi-Provider LLM Support

Models can specify provider prefix: `openai:gpt-4o-mini`, `anthropic:claude-3-haiku`, or plain `gemini-2.5-flash` (default Gemini).

### Configuration

All settings in `core/settings.py` via Pydantic. Key env vars:
- `GOOGLE_API_KEY` - Required for summaries (Gemini)
- `WHISPER_MODEL_NAME` - tiny/base/small/medium/large
- `WHISPER_DEVICE` - cpu/cuda
- `SUMMARIZER_MODEL` - Default: gemini-2.5-flash
- `LLM_CACHE_ENABLED` - Enable response caching

## Testing Patterns

- Fixtures in `tests/conftest.py` (mock Whisper model, temp dirs, sample data)
- Mock external APIs (yt-dlp, Whisper, Gemini) in tests
- Test files mirror source structure: `core/llm.py` -> `tests/core/test_llm.py`
