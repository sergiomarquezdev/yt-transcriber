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
