# Cleanup: post_kits + summary + reorg outputs por carpeta-por-vídeo

**Fecha**: 2026-05-06
**Tipo**: Refactor / cleanup (no nueva funcionalidad)
**Alcance**: yt-transcriber completo (core + yt_transcriber + tests + CLAUDE.md + filesystem `output/`)

## Objetivo

Reducir el aplicativo a su núcleo realmente usado (transcripción + segments + visual evidence) y simplificar la organización de outputs a una carpeta por vídeo. Eliminar todo el código residual que ha quedado sin consumidores.

## Decisiones (de la sesión de brainstorming)

| # | Decisión | Justificación |
|---|---|---|
| 1 | Borrar funcionalidad `post_kits` entera | El usuario no la utiliza; mantenerla añade superficie de fallo y prompts en TUI. |
| 2 | Borrar funcionalidad `summary` entera | Igual; además consume tokens de Claude CLI sin valor para el usuario. |
| 3 | Reorganizar outputs en `output/{stem}/` | Hoy los artefactos de un mismo vídeo están repartidos entre `output/transcripts/`, `output/summaries/`, `output/analysis/`. Una carpeta por vídeo es más navegable. |
| 4 | `stem = {titulo}_vid_{video_id}_job_{timestamp}` | Conservar el formato actual del filename garantiza unicidad por run sin colisiones (opción B en Q1). |
| 5 | Playlists: cada vídeo crea su carpeta plana bajo `output/`, sin carpeta padre de playlist | Mismo trato que `transcribe` (opción A en Q2). |
| 6 | Eliminar también el cache de transcripciones | `TRANSCRIPT_CACHE_ENABLED=False` por defecto y el usuario nunca reprocesa el mismo vídeo. |
| 7 | Eliminar `core/translator.py` | Sin `summary` no queda ningún consumidor productivo. El otro método (`translate_to_spanish`) importa de un paquete `yt_script_generator` que no existe en este repo. |
| 8 | Eliminar settings huérfanas | Settings sin consumidores en el código actual (residuos de un script generator/ideation engine no presente). |
| 9 | `git rm` de `output/transcripts/`, `output/summaries/`, `output/analysis/` | Empezar limpio (opción A en Q4). |
| 10 | Sin shim de retrocompatibilidad | Refactor en `main`, sin usuarios externos del módulo. CLAUDE.md global del usuario prohíbe atajos de back-compat innecesarios. |

## No-objetivos

- No tocar el pipeline de transcripción (Whisper, yt-dlp, ffmpeg, segments JSON, visual evidence). Se mueve solo el destino donde se escriben los artefactos.
- No añadir tests nuevos para la lógica que se borra.
- No cambiar nombres de flags ni de subcomandos que sobreviven (`transcribe`, `playlist`, `--language`, `--ffmpeg-location`, `--segments`, `--visual-evidence`, `--limit`).
- No tocar dependencias en `pyproject.toml` salvo que algún paquete quede sin uso tras el cleanup (verificar al final, no asumido aquí).

## Cambios de arquitectura

### `core/settings.py`

**Eliminar**:
- `OUTPUT_TRANSCRIPTS_DIR`, `SUMMARY_OUTPUT_DIR`, `SCRIPT_OUTPUT_DIR`, `ANALYSIS_OUTPUT_DIR`, `TEMP_BATCH_DIR`, `TRENDS_OUTPUT_DIR`
- `TRANSCRIPT_CACHE_ENABLED`, `TRANSCRIPT_CACHE_DIR`
- `SUMMARIZER_MODEL`, `PATTERN_ANALYZER_MODEL`, `QUERY_OPTIMIZER_MODEL`
- `SERPAPI_API_KEY`, `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`
- Todas las `POST_KITS_*` (8 settings)
- El alias retroactivo `OUTPUT_TRENDS_DIR` (ya no aplica)

**Añadir**:
- `OUTPUT_BASE_DIR: Path = Field(default=Path("output/"), description="Directorio base; cada vídeo crea una subcarpeta aquí")`

**Mantener** (verificado: todos tienen consumidores):
- `WHISPER_*`, `TEMP_DOWNLOAD_DIR`, `LOG_LEVEL`, `FFMPEG_LOCATION`
- `CLAUDE_CLI_PATH`, `CLAUDE_CLI_TIMEOUT`, `DEFAULT_LLM_MODEL` (lo usa `core/llm.py`; lo dejamos por si en el futuro vuelve un módulo LLM, no añade ruido)
- `PRO_MODEL` y `TRANSLATOR_MODEL`: revisar al editar — si tras borrar summarizer/translator nadie los usa, eliminarlos también. (Hipótesis actual: ambos quedan huérfanos y caen.)
- `TRANSCRIPT_SEGMENTS_ENABLED`, `VISUAL_EVIDENCE_ENABLED`, `VISUAL_EVIDENCE_MIN_SEGMENT_SECONDS`
- `YT_SEARCH_TIMEOUT_SECONDS` (lo usa `core/media_downloader.py`)

### `core/models.py`

**Eliminar**: `VideoSummary`, `TimestampedSection`. Actualizar `__all__`.
**Mantener**: `TranscriptSegment` (lo usa el segments JSON sidecar).

### `core/translator.py`

Borrar el archivo entero. Eliminar `tests/core/test_translator.py`.

### `yt_transcriber/summarizer.py`

Borrar el archivo entero. Eliminar `tests/yt_transcriber/test_summarizer.py`.

### `yt_transcriber/post_kits_generator.py`

Borrar el archivo entero. Eliminar `tests/yt_transcriber/test_post_kits_generator.py`.

### `yt_transcriber/models.py`

Borrar el archivo entero (solo contiene `LinkedInPost`, `TwitterThread`, `PostKits`). Eliminar `tests/yt_transcriber/test_models.py`.

### `yt_transcriber/service.py`

**Eliminar**:
- `generate_summary_outputs()` entera.
- Imports relacionados: `is_model_configured`, `ScriptTranslator`, `create_summary`, `TYPE_CHECKING` para `VideoSummary`.
- Parámetros `generate_post_kits`, `generate_summary`, `reuse_transcripts` de `process_transcription()`.
- Bloque entero de fast-path con `TRANSCRIPT_CACHE_ENABLED` / `reuse_transcripts`.
- Bloque "Generate summaries (only if requested)".
- La extracción temprana de `video_id` por regex/yt-dlp solo sobrevive si se sigue usando para construir `output_filename_base` (sí, se sigue usando — pero ahora sin el if cache).

**Cambiar firma**:
```python
def process_transcription(
    youtube_url: str,
    title: str,
    model: Any,
    language: str | None = None,
    ffmpeg_location: str | None = None,
    segments_override: bool | None = None,
    visual_override: bool | None = None,
) -> Path | None:
    """Returns the transcript path, or None on failure."""
```

**Cambiar destino de outputs** — el patrón nuevo:
```python
output_filename_base = f"{normalized_title}_vid_{download_result.video_id}_job_{unique_job_id}"
output_dir = settings.OUTPUT_BASE_DIR / output_filename_base
output_dir.mkdir(parents=True, exist_ok=True)
# transcript, segments JSON, frames todos van a output_dir
```

Aplicar también al fallback de `extract_audio_from_local_file` (cuando `is_local_file=True`) — ahí `download_result.video_id` puede ser `None` o un placeholder; verificar al implementar y, si es `None`, usar un fallback (`local`) para que el stem siga siendo único gracias a `_job_{unique_job_id}`.

### `yt_transcriber/cli.py`

- `process_transcription()` wrapper: ajustar firma (quitar params eliminados), cambiar tipo de retorno a `Path | None`.
- `command_transcribe`:
  - Quitar argumentos `--summarize` y `--post-kits` del parser.
  - Quitar la lógica `generate_summary = ... or args.post_kits`.
  - Pipeline pasa a recibir solo `transcript_path`. Logs simplificados.
- `command_playlist`:
  - Quitar argumentos `--summarize` y `--post-kits`.
  - Quitar el bloque `if generate_summary: ... generate_summary_outputs(...)`.
  - El `output_dir` para `download_auto_subtitles` cambia: por cada entrada construir `entry_dir = settings.OUTPUT_BASE_DIR / f"{normalized}_vid_{entry.video_id}_job_{job_id}"` y pasar ese al downloader. (Si `download_auto_subtitles` decide el filename internamente, comprobar que respeta `output_dir`.)
  - `files: list[str]` sigue siendo la lista de paths absolutos a los `.txt` generados.
- `run_transcribe_command()` y `run_playlist_command()`: ajustar firmas y tuplas de retorno (`run_transcribe_command` pasa de tupla de 4 a `str | None`).

### `yt_transcriber/tui.py`

- Quitar tooltips `_T_SUMMARIZE`, `_T_POST_KITS`.
- `prompt_transcribe_options()`: pasa de `5/5` a `3/3` (Idioma → Segments → Visual evidence). Quitar `summarize` y `post_kits` del dict de retorno.
- `prompt_playlist_options()`: pasa de `4/4` a `2/2` (Limit → Language). Quitar `summarize`/`post_kits`.
- `apply_validation_rules()`: queda solo la regla `visual_evidence → segments`. Considerar borrar la función si solo queda una regla, o dejarla por claridad (mantener: la firma compartida la usan tests).
- `format_command_preview()`: quitar `--summarize` y `--post-kits` del builder.
- `_print_transcribe_results()`: cambiar firma — recibe `str | None` (transcript path) en vez de tupla de 4.
- `_print_playlist_results()`: quitar referencias a summary/post_kits si las hubiera (revisar `stats`).
- `_run_transcribe()` y `_run_playlist()`: ajustar las llamadas a `run_transcribe_command`/`run_playlist_command` y los `_info_line` del bloque "Resumen".

### Tests a actualizar

- `tests/yt_transcriber/test_service.py`: borrar tests de `generate_summary_outputs`, summary flow, cache fast-path. Mantener tests de transcripción / segments / visual evidence pero ajustando los paths esperados al nuevo layout `output/{stem}/...`.
- `tests/yt_transcriber/test_cli.py`: borrar casos con `--summarize`/`--post-kits`; ajustar paths.
- `tests/yt_transcriber/test_tui.py`: actualizar prompts esperados (3/3 y 2/2).
- `tests/yt_transcriber/test_utils.py`: revisar; probablemente intacto.
- `tests/core/test_settings.py`: quitar asserts sobre settings borradas; añadir assert sobre `OUTPUT_BASE_DIR`.
- `tests/conftest.py`: el fixture `mock_whisper_model` no cambia; revisar si hay fixtures que setean dirs eliminados.

### Filesystem

```
git rm -r output/transcripts/ output/summaries/ output/analysis/
```

(Si el directorio `output/` queda vacío después del comando, dejarlo así — se recreará en el primer run.)

### `CLAUDE.md` (proyecto)

- Reescribir el primer párrafo: ya no hay "summarizer", "Post Kits", "translations".
- Borrar de la sección "Gotchas":
  - "LLM via Claude CLI" — revisar si queda algún consumidor de `call_llm` (en principio no; podría borrarse `core/llm.py` también). **Verificar al implementar.** Si nadie usa `call_llm`, eliminarlo y la sección entera del Claude CLI.
  - "Model names are Claude shortnames" — fuera si no hay LLM features.
- Actualizar "Error Hierarchy": quitar `LLMError*`, `TranslationError`, `SummarizationError`, `PostKitsError`. Quedan: `DownloadError`, `TranscriptionError`.
- Actualizar "Cross-File Workflows":
  - Borrar "Add a new LLM-powered feature" si no quedan features LLM.
  - "Add a new CLI subcommand" se mantiene.
- Actualizar "Testing": quitar referencia a SUMMARIZER y al mock de Claude CLI si ya no aplica.

## Layout de outputs (antes / después)

**Antes** (un vídeo de YouTube con todos los flags):
```
output/
├── transcripts/
│   ├── {stem}.txt
│   ├── {stem}_segments.json
│   └── {stem}_frame_0.jpg
├── summaries/
│   ├── {stem}_summary_EN.md
│   ├── {stem}_summary_ES.md
│   └── {stem}_post_kits.md
└── analysis/
    └── transcripts_cache/{video_id}.txt
```

**Después** (el mismo vídeo):
```
output/
└── {stem}/
    ├── {stem}.txt
    ├── {stem}_segments.json
    └── {stem}_frame_0.jpg
```

donde `stem = {normalized_title}_vid_{video_id}_job_{timestamp}`.

## Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| `download_auto_subtitles` decide su propio nombre/dir y rompe el layout esperado. | Leer la función al implementar; si no acepta `output_dir`, ajustar tests y/o pasar el dir esperado. |
| Algún test patchea settings borradas (`SUMMARY_OUTPUT_DIR`, etc.) y peta. | Buscar referencias antes de borrar (`Grep` sobre `SUMMARY_OUTPUT_DIR`, etc.) y limpiar. |
| `core/llm.py` queda sin consumidores → ¿borrarlo? | Verificar tras el cleanup. Si nadie llama `call_llm` ni `is_model_configured`, eliminarlo y `tests/core/test_llm.py`. Documentar la decisión en el plan. |
| `PRO_MODEL` / `TRANSLATOR_MODEL` quedan huérfanos. | Igual: verificar al editar `settings.py` y eliminar si nadie los usa. |
| Tests que asumen `Path | None | None | None` en retornos. | Refactor coordinado: cambiar firma + ajustar tests en el mismo paso. |

## Criterio de éxito

1. Tests pasan (`pytest`).
2. `python -m yt_transcriber.cli transcribe -u <url>` produce un único directorio bajo `output/` con el `.txt` dentro y nada más.
3. Con `--segments` añade el `_segments.json` en el mismo directorio.
4. Con `--visual-evidence` (vídeo local) añade los frames `_frame_*.jpg` en el mismo directorio.
5. `python -m yt_transcriber.cli playlist -u <url> -n 3` produce 3 carpetas bajo `output/`, una por vídeo, cada una con su `.txt`.
6. La TUI no muestra prompts de summarize/post-kits; transcribe pregunta 3 cosas; playlist pregunta 2.
7. `grep -r "post_kits\|summarizer\|VideoSummary\|LinkedInPost\|TwitterThread\|TRANSCRIPT_CACHE\|ScriptTranslator" --include="*.py"` no devuelve nada (excepto en el propio spec/plan en `docs/`).
8. Imports y configuración limpios: `python -c "from core.settings import settings; from yt_transcriber import service; from yt_transcriber import cli; from yt_transcriber import tui"` no falla.
9. `output/transcripts/`, `output/summaries/`, `output/analysis/` no existen tras el commit.

## Plan de implementación (alto nivel — el detalle paso a paso lo escribe writing-plans)

1. **Borrar módulos huérfanos** y sus tests (post_kits, summarizer, translator, models de post_kits, VideoSummary/TimestampedSection).
2. **Limpiar `core/settings.py`** (settings huérfanas + introducir `OUTPUT_BASE_DIR`).
3. **Refactor `service.py`** (firma nueva + reorg outputs por carpeta).
4. **Refactor `cli.py`** (parsers, wrappers, command handlers).
5. **Refactor `tui.py`** (prompts, validation, preview, results).
6. **Actualizar tests** restantes para el nuevo layout y firmas.
7. **`git rm` de `output/transcripts|summaries|analysis`** y commit.
8. **Verificar settings/módulos residuales** (`core/llm.py`, `PRO_MODEL`, `TRANSLATOR_MODEL`) y eliminar si están sin consumidores.
9. **Actualizar CLAUDE.md** acorde.
10. **Smoke test manual**: un `transcribe` y un `playlist -n 1` reales para confirmar layout.
