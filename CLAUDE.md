# CLAUDE.md

YouTube CLI transcriber + summarizer. Python 3.12+, faster-whisper (CTranslate2), Claude CLI for LLM calls (summaries, translations, post kits). Two subcommands: `transcribe` (Whisper) and `playlist` (YouTube auto-subs, no GPU).

## Gotchas

- **LLM via Claude CLI**: `call_llm()` in `core/llm.py` invokes `claude` CLI via `subprocess.run()`. Prompt goes via stdin (`input=`), response comes from stdout. No SDK dependencies, no API keys -- auth is via Claude subscription (Max/Pro).
- **Model names are Claude shortnames**: `"sonnet"`, `"haiku"`, `"opus"`. Configured in `settings.py` per feature (SUMMARIZER_MODEL, TRANSLATOR_MODEL, etc.).
- **No LLM cache**: removed. Each call spawns a new CLI process. Acceptable latency for current use cases.
- **Whisper context manager is mandatory**: always use `whisper_model_context()` from `whisper_context.py`. It handles CTranslate2 model load/unload + `gc.collect()`. Leaking models exhausts VRAM.
- **Playlist mode skips Whisper entirely**: `command_playlist` downloads YouTube auto-generated subtitles via yt-dlp (`skip_download=True`), cleans SRT/VTT to plain text. No audio, no GPU, no model loading.
- **`noplaylist: True` is hardcoded** in `download_and_extract_audio()`. Playlist support lives in separate functions (`extract_playlist_entries`, `download_auto_subtitles`), not in the existing download path.

## Error Hierarchy

```
LLMError (base) -> LLMProviderError (CLI errors) | LLMConfigurationError (CLI not found)
DownloadError          # core/media_downloader.py
TranscriptionError     # core/media_transcriber.py
TranslationError       # core/translator.py
SummarizationError     # yt_transcriber/summarizer.py
PostKitsError          # yt_transcriber/post_kits_generator.py
```

## Cross-File Workflows

**Add a new LLM-powered feature:**
1. Create module with prompt template, call `call_llm()` from `core/llm.py`
2. Wire into `service.py` or `cli.py` as needed

**Add a new CLI subcommand:**
1. Add parser in `cli.py:main()` under `subparsers`
2. Create `command_<name>(args)` handler in `cli.py`
3. Add routing in the `if args.command ==` block at bottom of `main()`

## Testing

- All external services are mocked (yt-dlp, faster-whisper, Claude CLI via `subprocess.run`)
- `core/settings.py` loads `.env` at import time via `load_dotenv()`. Tests that need specific env vars must use `monkeypatch.setenv` BEFORE importing settings, or patch `settings` directly
- `conftest.py` has `mock_whisper_model` fixture returning a MagicMock with `.transcribe()` -- use it instead of creating ad-hoc mocks
