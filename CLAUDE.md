# CLAUDE.md

YouTube CLI transcriber + summarizer. Python 3.12+, faster-whisper (CTranslate2), multi-provider LLM (Gemini default). Two subcommands: `transcribe` (Whisper) and `playlist` (YouTube auto-subs, no GPU).

## Gotchas

- **LLM dispatcher name is misleading**: `call_gemini_with_cache()` in `core/llm.py` routes to ALL providers (Gemini/OpenAI/Anthropic) despite the name. It's the single entry point for all LLM calls. Don't create separate call functions.
- **Provider prefix convention**: model names use `"provider:model"` format (`"openai:gpt-4o"`, `"anthropic:claude-3-haiku"`). No prefix defaults to Gemini. Parsed in `call_gemini_with_cache`.
- **Prompt versioning invalidates cache**: every caller passes a `prompt_version` string (from `settings.*_PROMPT_VERSION`). Bump the version when changing a prompt template, otherwise cached responses persist.
- **Whisper context manager is mandatory**: always use `whisper_model_context()` from `whisper_context.py`. It handles CTranslate2 model load/unload + `gc.collect()`. Leaking models exhausts VRAM.
- **Playlist mode skips Whisper entirely**: `command_playlist` downloads YouTube auto-generated subtitles via yt-dlp (`skip_download=True`), cleans SRT/VTT to plain text. No audio, no GPU, no model loading.
- **`noplaylist: True` is hardcoded** in `download_and_extract_audio()`. Playlist support lives in separate functions (`extract_playlist_entries`, `download_auto_subtitles`), not in the existing download path.

## Error Hierarchy

```
LLMError (base) -> LLMProviderError (API errors) | LLMConfigurationError (missing keys)
DownloadError          # core/media_downloader.py
TranscriptionError     # core/media_transcriber.py
TranslationError       # core/translator.py
SummarizationError     # yt_transcriber/summarizer.py
PostKitsError          # yt_transcriber/post_kits_generator.py
```

Callers pass `error_class` to `call_gemini_with_cache()` so it raises domain-specific exceptions.

## Cross-File Workflows

**Add a new LLM-powered feature:**
1. Add prompt version setting in `core/settings.py` (`*_PROMPT_VERSION`)
2. Create module with prompt template, call `call_gemini_with_cache()` from `core/llm.py`
3. Wire into `service.py` or `cli.py` as needed

**Add a new CLI subcommand:**
1. Add parser in `cli.py:main()` under `subparsers`
2. Create `command_<name>(args)` handler in `cli.py`
3. Add routing in the `if args.command ==` block at bottom of `main()`

## Testing

- All external services are mocked (yt-dlp, faster-whisper, Gemini/OpenAI/Anthropic APIs)
- `core/settings.py` loads `.env` at import time via `load_dotenv()`. Tests that need specific env vars must use `monkeypatch.setenv` BEFORE importing settings, or patch `settings` directly
- `conftest.py` has `mock_whisper_model` fixture returning a MagicMock with `.transcribe()` -- use it instead of creating ad-hoc mocks
