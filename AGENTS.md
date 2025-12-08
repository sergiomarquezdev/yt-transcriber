# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YouTube Video Transcriber & Summarizer: CLI tool that transforms YouTube videos, Google Drive files, or local media into transcripts and AI-powered summaries with optional social media post generation.

**Stack**: Python 3.13+, Whisper (OpenAI), Multi-provider LLM (Gemini/OpenAI/Anthropic), yt-dlp, Pydantic Settings

**Input sources**: YouTube URLs, Google Drive URLs, local video/audio files

## Commands

```bash
# Install (editable mode)
pip install -e .
pip install -e ".[dev]"  # with dev dependencies

# Run CLI
yt-transcriber transcribe --url "URL"                    # Only transcription (DEFAULT)
yt-transcriber transcribe --url "URL" --summarize        # + AI summaries (EN + ES)
yt-transcriber transcribe --url "URL" --post-kits        # + summaries + LinkedIn/Twitter

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
│   ├── cache.py             # Content-hash based LLM response cache (with in-memory LRU)
│   ├── media_downloader.py  # yt-dlp + FFmpeg audio extraction (with timeout protection)
│   ├── media_transcriber.py # Whisper transcription wrapper
│   ├── translator.py        # EN->ES translation via LLM
│   └── models.py            # Shared data models (VideoSummary, TimestampedSection)
│
├── yt_transcriber/          # Main application package
│   ├── cli.py               # argparse CLI entry point (transcribe command + flags)
│   ├── service.py           # Orchestration: download -> transcribe -> summarize -> translate
│   ├── whisper_context.py   # Context manager for Whisper model lifecycle (auto memory cleanup)
│   ├── summarizer.py        # Gemini-powered video summarization
│   ├── post_kits_generator.py # LinkedIn post + Twitter thread generation
│   ├── utils.py             # File operations, title normalization
│   └── models.py            # Application-specific models
│
└── tests/                   # pytest tests mirroring src structure
    └── test_performance_issues.py  # Performance optimization verification tests
```

### Data Flow

1. **CLI** (`cli.py`) parses args, delegates to service
2. **Service** (`service.py`) orchestrates the pipeline:
   - Uses `whisper_model_context()` context manager for automatic model cleanup
   - Uses `TemporaryDirectory` for automatic temp file cleanup
   - Download/extract audio via `core.media_downloader` (with 5min timeout protection)
   - Transcribe via `core.media_transcriber` (Whisper)
   - Generate EN summary via `yt_transcriber.summarizer`
   - Translate to ES via `core.translator`
   - Optional: generate post kits
3. **LLM calls** route through `core.llm.call_gemini_with_cache()` with:
   - Content-hash caching (disk)
   - In-memory LRU cache (fast repeated lookups)
   - Thread-safe rate limiting

### Multi-Provider LLM Support

`core.llm.call_gemini_with_cache()` intelligently routes to provider based on model name prefix:
- `openai:gpt-4o-mini` → OpenAI Chat Completions
- `anthropic:claude-3-haiku` → Anthropic Messages API
- `gemini-2.5-flash` (no prefix) → Google Gemini (default)

**Caching optimizations:**
- Disk cache: Content-hash based (model + prompt version + inputs)
- Memory cache: In-memory LRU for fast repeated lookups (reduces disk I/O)
- Probabilistic cleanup: 5% chance to clean expired files on write
- Thread-safe: Uses locks for concurrent access protection
- Cache keys include model name + prompt version + input data, ensuring:
  - Updated prompt version → cache invalidation
  - Same inputs + same version → instant cache hit (free)

### Configuration

All settings in `core/settings.py` via Pydantic BaseSettings. Key env vars:

**Whisper (transcription):**
- `WHISPER_MODEL_NAME` - tiny/base/small/medium/large (default: base)
- `WHISPER_DEVICE` - cpu/cuda (default: cpu)

**LLM API keys (at least one required for summaries):**
- `GOOGLE_API_KEY` - Gemini models
- `OPENAI_API_KEY` - OpenAI models (optional)
- `ANTHROPIC_API_KEY` - Anthropic models (optional)

**Model selection (use provider prefix for non-Gemini):**
- `SUMMARIZER_MODEL` - Default: gemini-2.5-flash
- `TRANSLATOR_MODEL` - Default: gemini-2.5-flash-lite
- `PRO_MODEL` - Default: gemini-2.5-pro

**Caching:**
- `LLM_CACHE_ENABLED` - Enable response caching (default: true)
- `LLM_CACHE_TTL_DAYS` - Cache lifetime (default: 7)
- `LLM_CACHE_DIR` - Cache storage (default: output/.llm_cache/)
- `TRANSCRIPT_CACHE_ENABLED` - Reuse transcripts by video_id (default: false)

**Prompt versioning (for cache invalidation):**
- `SUMMARIZER_PROMPT_VERSION` - Update to invalidate summary cache
- `TRANSLATOR_PROMPT_VERSION` - Update to invalidate translation cache
- `POST_KITS_PROMPT_VERSION` - Update to invalidate post kits cache

**Backward compatibility aliases:**
- `OUTPUT_TRENDS_DIR` → `TRENDS_OUTPUT_DIR`
- `LLM_CACHE_PATH` → `LLM_CACHE_DIR`

## Programmatic Usage

```python
from yt_transcriber.cli import run_transcribe_command

# Transcription only
transcript, _, _, _ = run_transcribe_command(url="path/to/video.mp4")

# With summaries
transcript, summary_en, summary_es, _ = run_transcribe_command(
    url="https://youtube.com/watch?v=VIDEO_ID",
    generate_summary=True
)

# With post kits (implies summary)
transcript, summary_en, summary_es, post_kits = run_transcribe_command(
    url="https://youtube.com/watch?v=VIDEO_ID",
    generate_post_kits=True
)
```

## Testing Patterns

**Fixtures** (`tests/conftest.py`):
- `mock_whisper_model` - Returns sample transcription without real Whisper
- `temp_dir`, `temp_cache_dir`, `temp_output_dir` - Isolated test directories
- `sample_transcript`, `sample_video_summary` - Realistic test data
- `mock_env_vars`, `clean_env` - Environment variable control
- `sample_llm_summary_response`, `sample_linkedin_response` - Mock LLM responses

**Patterns:**
- Mock external APIs (yt-dlp, Whisper, Gemini/OpenAI/Anthropic)
- Test files mirror source: `core/llm.py` → `tests/core/test_llm.py`
- Use fixtures for consistent test data and cleanup

## Performance Optimizations

The codebase includes several performance and resource management optimizations:

**Memory Management:**
- `whisper_context.py`: Context manager that auto-loads/unloads Whisper model, freeing RAM/VRAM
- Automatic `gc.collect()` and `torch.cuda.empty_cache()` on model cleanup
- In-memory LRU cache in `cache.py` reduces repeated disk I/O

**Resource Cleanup:**
- `TemporaryDirectory` in `service.py` ensures temp files are cleaned even on exceptions
- Probabilistic cache cleanup (5% chance on write) prevents unbounded growth

**Concurrency Safety:**
- Thread-safe rate limiter in `llm.py` with `deque(maxlen=...)` to prevent memory leaks
- `threading.Lock` protects shared state in concurrent LLM calls
- Sleep happens outside lock to avoid blocking other threads

**Process Protection:**
- FFmpeg timeout (5min) in `media_downloader.py` prevents hung processes
- Video ID extraction optimized: tries regex first, falls back to yt-dlp only if needed

**Verification Tests:**
- `tests/test_performance_issues.py` contains verification tests for all optimizations
- Run with: `pytest tests/test_performance_issues.py -v`
