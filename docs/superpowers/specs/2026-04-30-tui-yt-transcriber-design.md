# Design: TUI para yt-transcriber

**Fecha**: 2026-04-30
**Estado**: aprobado (owner: smarq, brainstormed con Claude orchestrator)

## Goal

Reemplazar la invocación manual del CLI de `yt-transcriber` (que requiere conocer flags, escribir comandos largos en terminal) por una interfaz interactiva tipo wizard que se lanza con un alias corto (`ytt`) en Git Bash/WezTerm o doble-clic en un `.bat` en Windows. Objetivo de UX: dos clics o tres caracteres → pegar URL → pulsar Enter en defaults → transcripción corriendo.

No es una GUI ni un servidor web; es una TUI que vive en la misma terminal y delega 100% en el código existente del CLI.

## Non-Goals

- No reescribir el core (`service.py`, `core/*`, `cli.py`).
- No añadir features nuevos (cache, summarization, etc. ya existen).
- No cubrir edición de `.env` desde la TUI (el `.env` se afina una vez y se queda).
- No paquetizar como ejecutable standalone (.exe). Sigue siendo Python+venv.
- No internacionalizar la TUI (textos en español + términos técnicos en inglés, alineado con AGENTS.md global).

## Requirements

### Funcionales

1. **Single entry point**: `python -m yt_transcriber.tui` lanza la TUI desde cualquier cwd siempre que el venv esté activo.
2. **Detección automática de input**: dado un string, decidir si es:
   - YouTube playlist URL → ruta `playlist`
   - YouTube video URL (no playlist) → ruta `transcribe`
   - Google Drive URL → ruta `transcribe`
   - Ruta local existente → ruta `transcribe` (con visual-evidence habilitado como opción)
   - Otro → error con mensaje claro y reintento
3. **Prompts contextuales**: solo se preguntan flags que aplican al tipo detectado.
4. **Tooltips/ayuda inline**: cada prompt incluye una línea breve explicando qué hace la opción.
5. **Defaults sensatos**: pulsar Enter en cada prompt usa el default seguro (no costoso, no destructivo). Defaults se documentan abajo.
6. **Validaciones cruzadas automáticas**: si el usuario activa `--post-kits`, la TUI activa `--summarize` sola; si activa `--visual-evidence`, activa `--segments` sola.
7. **Preview pre-ejecución**: antes de lanzar, mostrar una línea de resumen tipo `transcribe -u "..." --language es --summarize` y pedir confirmación.
8. **Ejecución in-process**: invocar `command_transcribe(args)` o `command_playlist(args)` directamente (no subprocess), para que los logs salen al terminal padre y Ctrl+C interrumpe limpio.
9. **Loop opcional**: al terminar una transcripción, preguntar "¿Otra? [Y/n]". Default Yes para enlazar trabajos seguidos sin relanzar.
10. **Lanzadores**:
    - `launch_tui.bat` en raíz del repo (doble-clic en Windows).
    - Alias/función `ytt` en `~/ai_configs/shell/ai-tools.sh` (Git Bash + WezTerm).

### No funcionales

- **Cero deps pesadas**: solo añade `questionary` (~30KB pure-python).
- **Aislamiento**: toda la TUI vive en `yt_transcriber/tui.py` (un único archivo). Si se borra ese archivo, el CLI sigue funcionando idéntico.
- **Reuso del API programático existente**: `cli.py` ya expone `run_transcribe_command(url, language, ffmpeg_location, generate_post_kits, generate_summary, reuse_transcripts, segments_override, visual_override) -> tuple[str|None, str|None, str|None, str|None]`. La TUI lo usa directamente (no llama a `command_transcribe`, que hace `sys.exit`). Para playlist se añade `run_playlist_command(url, limit, language, generate_summary, generate_post_kits) -> dict` siguiendo el mismo patrón (no `sys.exit`, retorna stats). Esta es la única modificación no-aditiva en `cli.py`.
- **Compat Windows-first**: el `.bat` es prioridad. Un `launch_tui.sh` análogo se añade pero es secundario.

## Architecture

```
┌─────────────────────────────────────────────────┐
│ launch_tui.bat (doble-clic Windows)             │
│ ytt (alias bash)                                │
└──────────────────┬──────────────────────────────┘
                   │ activa venv + invoca
                   ▼
┌─────────────────────────────────────────────────┐
│ python -m yt_transcriber.tui                    │
│                                                 │
│  yt_transcriber/tui.py                          │
│   ├── detect_input_type(value) -> InputType     │
│   ├── prompt_transcribe_options(...) -> dict    │
│   ├── prompt_playlist_options(...) -> dict      │
│   ├── build_namespace(...) -> argparse.Namespace│
│   ├── format_command_preview(...) -> str        │
│   └── main() -> loop                            │
└──────────────────┬──────────────────────────────┘
                   │ import + call (in-process)
                   ▼
┌─────────────────────────────────────────────────┐
│ yt_transcriber/cli.py                           │
│   command_transcribe(args)                      │
│   command_playlist(args)                        │
└─────────────────────────────────────────────────┘
```

## Components

### `yt_transcriber/tui.py`

Funciones públicas (todas testeables aisladamente):

#### `detect_input_type(value: str) -> InputType`

Enum:
```python
class InputType(str, Enum):
    YOUTUBE_VIDEO = "youtube_video"
    YOUTUBE_PLAYLIST = "youtube_playlist"
    DRIVE = "drive"
    LOCAL = "local"
    UNKNOWN = "unknown"
```

Lógica (orden importante):

1. Si `Path(value).expanduser().exists()` → `LOCAL`.
2. Si contiene `playlist?list=` o `&list=` (cualquier dominio youtube) → `YOUTUBE_PLAYLIST`.
3. Si contiene `youtube.com/watch` o `youtu.be/` → `YOUTUBE_VIDEO`.
4. Si contiene `drive.google.com` o `docs.google.com` → `DRIVE`.
5. Else → `UNKNOWN`.

Notas:
- El check de path va primero porque rutas locales nunca contienen los markers de URL.
- Una URL tipo `https://www.youtube.com/watch?v=XXX&list=YYY` se detecta como playlist (ese es el comportamiento típico esperado: si tiene `list=`, el usuario quiere la playlist completa).

#### `prompt_transcribe_options(input_type: InputType) -> dict`

Devuelve dict con keys: `language`, `summarize`, `post_kits`, `segments`, `visual_evidence`.

Prompts (en orden):

| # | Pregunta | Tipo questionary | Default | Tooltip |
|---|---|---|---|---|
| 1 | Idioma del audio | `select` | `auto-detect` | Auto-detect funciona bien pero es un pelín más lento. Valores: auto-detect / es / en / otro |
| 1b | (si "otro") Código de idioma ISO | `text` | "" | Ej: `pt`, `fr`, `de`. Vacío = cancelar y volver. |
| 2 | Generar resúmenes (EN+ES) | `confirm` | False | Produce summary_EN.md + summary_ES.md vía Claude. |
| 3 | Generar post kits (LinkedIn + Twitter) | `confirm` | False | Activa resúmenes automáticamente. |
| 4 | Sidecar de segmentos JSON | `confirm` | False | _segments.json con timestamps por segmento. |
| 5 | Visual evidence (frames clave) | `confirm` | False | (solo si LOCAL) Extrae frames. Activa segmentos automáticamente. |

Reglas implícitas (aplicadas tras los prompts):
- Si `post_kits = True` → `summarize = True` (forzado).
- Si `visual_evidence = True` → `segments = True` (forzado).
- Si `input_type != LOCAL` → prompt 5 se omite y `visual_evidence = False`.

#### `prompt_playlist_options() -> dict`

Devuelve dict con keys: `limit`, `language`, `summarize`, `post_kits`.

| # | Pregunta | Tipo | Default | Tooltip |
|---|---|---|---|---|
| 1 | Cuántos videos (últimos N) | `text` | "" | Vacío = playlist completa. Número entero = últimos N. |
| 2 | Idioma de auto-subs | `select` | `es` | Idioma de los subtítulos automáticos a descargar. Valores: es / en / otro |
| 2b | (si "otro") Código ISO | `text` | "" | Ej: `pt`, `fr`. |
| 3 | Generar resúmenes | `confirm` | False | (igual que transcribe) |
| 4 | Generar post kits | `confirm` | False | Activa resúmenes automáticamente. |

Validación: si `limit` se introduce y no es entero positivo, repetir prompt.

#### Mapeo a wrappers programáticos

La TUI no construye `argparse.Namespace`. Llama directamente a:

- `run_transcribe_command(url=..., language=..., ffmpeg_location=None, generate_post_kits=..., generate_summary=..., reuse_transcripts=False, segments_override=..., visual_override=...)` — ya existente.
- `run_playlist_command(url=..., limit=..., language=..., generate_summary=..., generate_post_kits=...)` — a crear como espejo de `run_transcribe_command` (Task 2 del plan).

Ambos retornan datos estructurados (sin `sys.exit`), captura interna de excepciones, y la TUI imprime resultado + reanuda el loop.

#### `format_command_preview(subcommand: str, options: dict, url: str) -> str`

Para mostrar al usuario antes de ejecutar (string informativo, no se ejecuta):
```
yt-transcriber transcribe -u "https://www.youtube.com/watch?v=..." --language es --summarize --post-kits
```

#### `main()`

Loop:
1. Prompt URL/path (validado: no vacío).
2. Detect type. Si `UNKNOWN` → mensaje de error + reintento.
3. Según type → `prompt_transcribe_options` o `prompt_playlist_options`.
4. `build_namespace`.
5. Mostrar preview + confirm.
6. Llamar a `run_transcribe_command(...)` o `run_playlist_command(...)` (in-process).
7. Capturar `Exception` genérico (los wrappers ya capturan internamente, pero por defensa) → imprimir mensaje claro, no crash.
8. Listar archivos producidos:
   - Para transcribe: usar el tuple retornado por `run_transcribe_command`. Imprimir cada path no-`None`.
   - Para playlist: usar el dict retornado por `run_playlist_command` (`{"successful": N, "failed": N, "files": [...]}`).
9. Prompt "¿Otra transcripción? [Y/n]" → loop o salida.

KeyboardInterrupt en cualquier punto → salir limpio con mensaje "Cancelado por el usuario."

### `launch_tui.bat`

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

**Nota**: a fecha del spec, `.venv/` no existe en el repo. La creación queda fuera de alcance del plan automatizado (es responsabilidad del usuario; el `.bat` y el alias detectan la ausencia y dan instrucciones).

`pause` final para que la ventana no se cierre tras terminar (útil para revisar output).

### Alias `ytt`

En `~/ai_configs/shell/ai-tools.sh`, sección `# AI Tools`:

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

Función bash, no alias, porque necesita el `cd` en subshell + validación de `.venv`. Args opcionales se forwardean por si en el futuro queremos `ytt --debug`.

## Data Flow

```
usuario: pega URL
    ↓
detect_input_type → YOUTUBE_VIDEO
    ↓
prompt_transcribe_options → {language: "es", summarize: true, post_kits: false, segments: false, visual_evidence: false}
    ↓
build_namespace → Namespace(command="transcribe", url="...", language="es", summarize=True, ...)
    ↓
format_command_preview → "transcribe -u '...' --language es --summarize"
    ↓
confirm Y
    ↓
command_transcribe(ns)  ← reutiliza el código del CLI tal cual
    ↓
output/transcripts/...txt + output/summaries/...md
    ↓
TUI lista archivos producidos
    ↓
"¿Otra? [Y/n]" → loop
```

## Error Handling

| Error | Comportamiento TUI |
|---|---|
| Input vacío | Re-prompt del input. |
| Input `UNKNOWN` (no es URL ni path) | Mensaje "No reconozco ese input como YouTube/Drive/archivo local. Reintenta." → re-prompt. |
| `DownloadError` | Imprimir error, volver al loop principal (preguntar ¿otra?). |
| `TranscriptionError` | Idem. |
| `SummarizationError` / `PostKitsError` | Idem, indicando que el `.txt` ya está generado (no se pierde el trabajo principal). |
| `KeyboardInterrupt` | Salir limpio con `print("\nCancelado.")`. |
| `LLMConfigurationError` (Claude CLI no encontrado) | Mensaje específico: "Claude CLI no encontrado. Verifica `claude --version` o ajusta `CLAUDE_CLI_PATH` en `.env`." → loop. |
| Cualquier otra excepción | Imprimir traceback corto + tipo de excepción, volver al loop (no crashear todo el proceso). |

## `.env` afinado (one-time)

Reemplazo del `.env` actual con:

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

## Testing

`tests/test_tui.py`:

1. **`detect_input_type`** — table-driven:
   - Local path existente (usa `tmp_path` fixture) → `LOCAL`.
   - Local path inexistente → `UNKNOWN`.
   - `https://www.youtube.com/watch?v=abc123` → `YOUTUBE_VIDEO`.
   - `https://youtu.be/abc123` → `YOUTUBE_VIDEO`.
   - `https://www.youtube.com/playlist?list=PLxxx` → `YOUTUBE_PLAYLIST`.
   - `https://www.youtube.com/watch?v=abc&list=PLxxx` → `YOUTUBE_PLAYLIST` (ambigua, prefiere playlist).
   - `https://drive.google.com/file/d/xxx` → `DRIVE`.
   - `https://docs.google.com/document/d/xxx` → `DRIVE`.
   - `https://example.com` → `UNKNOWN`.
   - String vacío → `UNKNOWN`.

2. **`format_command_preview`** (dos tests, uno por subcomando):
   - Verifica que el string preview contiene los flags correctos según `options`.

3. **Validaciones cruzadas** (test unitario sobre la lógica post-prompts):
   - `post_kits=True` fuerza `summarize=True`.
   - `visual_evidence=True` fuerza `segments=True`.

4. **`run_playlist_command`** (test independiente, en `tests/yt_transcriber/test_cli.py`):
   - Mock `extract_playlist_entries` y `download_auto_subtitles`. Verifica que retorna dict con stats correctas y NO llama a `sys.exit`.
   - Verifica que excepciones internas son capturadas y reflejadas en `failed`.

4. **Smoke test**:
   - `import yt_transcriber.tui` no crashea.
   - `python -m yt_transcriber.tui --help` o equivalente: si `tui.py` tiene `if __name__ == "__main__": main()`, el smoke test es solo el import.

5. **No se testean los prompts interactivos** (`questionary` ya está testeado upstream; mockearlo añade complejidad sin valor).

Comandos de verificación:
- `pytest tests/yt_transcriber/test_tui.py -v` (TUI tests)
- `pytest tests/yt_transcriber/test_cli.py::TestRunPlaylistCommand -v` (wrapper tests)
- `pytest -v` (full suite, no debe romper tests existentes)

## Decisiones técnicas

| Decisión | Justificación |
|---|---|
| `questionary` vs `prompt_toolkit` puro vs `inquirer` | `questionary` envuelve `prompt_toolkit` con API simple, mantenido, MIT, ~30KB. `inquirer` está medio abandonado. `prompt_toolkit` directo es bajo nivel. |
| In-process vs subprocess | In-process: logs en vivo, Ctrl+C limpio, mismo intérprete (no hay overhead de spawn ni de re-importar Whisper). |
| Función bash `ytt` vs alias | Necesitamos `cd` en subshell para no romper el cwd del usuario. Alias no soporta esto. |
| `.bat` con `pause` | Sin `pause`, la ventana cierra al terminar y el usuario no ve los archivos producidos. |
| `medium` vs `large-v3` | Decidido en brainstorming: `medium` deja margen de VRAM para apps abiertas. |
| Loop opcional al final | Productividad: si el usuario tiene varias URLs seguidas, no relanza el comando cada vez. |

## Out of Scope (para futuras iteraciones)

- Edición del `.env` desde la TUI.
- Modo batch (leer URLs desde archivo).
- Historial de transcripciones recientes con re-run.
- Configuración de modelo Whisper/LLM por sesión (override del `.env`).
- Dark/light theme, paginación, búsqueda en outputs.
- Exportar a otros formatos (JSON, SRT directos sin pasar por `_segments.json`).
- Notificaciones desktop al terminar trabajos largos.

## Plan de implementación

Ver `plans/2026-04-30-tui-yt-transcriber-plan.md` (escrito tras aprobar este spec).
