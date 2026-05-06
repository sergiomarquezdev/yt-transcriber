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
