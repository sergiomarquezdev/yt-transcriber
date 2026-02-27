"""Shared application settings for the entire YouTube Content Suite.

This module contains the validated Pydantic settings used across all modules
(transcriber, script generator, content ideation, frontend).

All environment variables, API keys, model names, and directory paths are
centralized here to provide a single source of truth for configuration.
"""

import contextlib
import os
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Cargar variables de entorno desde un archivo .env si existe
load_dotenv()


class AppSettings(BaseSettings):
    """
    Configuraciones de la aplicación, validadas con Pydantic.
    Lee variables de entorno y aplica valores por defecto.
    """

    model_config = SettingsConfigDict(case_sensitive=False)

    WHISPER_MODEL_NAME: str = Field(
        default="base",
        description="Modelo de Whisper (base, small, medium, large-v3, distil-large-v3, etc.)",
    )
    WHISPER_DEVICE: Literal["cpu", "cuda", "auto"] = Field(
        default="auto",
        description="Dispositivo para ejecutar Whisper (auto detecta CUDA via CTranslate2)",
    )
    WHISPER_COMPUTE_TYPE: str = Field(
        default="default",
        description="Tipo de computo CTranslate2 (int8_float16 para GPU, int8 para CPU, default auto)",
    )
    WHISPER_BEAM_SIZE: int = Field(
        default=5,
        description="Beam search size (1=greedy, 5=default)",
    )
    WHISPER_VAD_FILTER: bool = Field(
        default=True,
        description="Silero VAD filter para saltar silencios automaticamente",
    )
    TEMP_DOWNLOAD_DIR: Path = Field(
        default=Path("temp_files/"),
        description="Directorio para archivos temporales",
    )
    OUTPUT_TRANSCRIPTS_DIR: Path = Field(
        default=Path("output/transcripts/"),
        description="Directorio para transcripciones generadas",
    )
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Nivel de logging",
    )
    FFMPEG_LOCATION: str = Field(
        default="",
        description="Ruta personalizada a FFmpeg (opcional)",
    )

    # ========== CLAUDE CLI CONFIGURATION ==========

    CLAUDE_CLI_PATH: str = Field(
        default="claude",
        description="Path to Claude CLI executable",
    )
    CLAUDE_CLI_TIMEOUT: int = Field(
        default=180,
        description="Claude CLI timeout in seconds",
    )
    DEFAULT_LLM_MODEL: str = Field(
        default="sonnet",
        description="Default Claude model (opus/sonnet/haiku)",
    )

    # ========== MODEL SELECTION (Claude model names: opus/sonnet/haiku) ==========

    PRO_MODEL: str = Field(
        default="opus",
        description="Modelo premium para synthesis y script generation",
    )
    SUMMARIZER_MODEL: str = Field(
        default="sonnet",
        description="Modelo para resúmenes de video",
    )
    PATTERN_ANALYZER_MODEL: str = Field(
        default="sonnet",
        description="Modelo para análisis de patrones",
    )
    TRANSLATOR_MODEL: str = Field(
        default="haiku",
        description="Modelo para traducciones simples",
    )
    QUERY_OPTIMIZER_MODEL: str = Field(
        default="haiku",
        description="Modelo para optimización de queries",
    )

    # ========== DIRECTORY CONFIGURATION ==========

    SCRIPT_OUTPUT_DIR: Path = Field(
        default=Path("output/scripts/"),
        description="Directorio para guiones generados",
    )
    ANALYSIS_OUTPUT_DIR: Path = Field(
        default=Path("output/analysis/"),
        description="Directorio para análisis y síntesis",
    )
    TEMP_BATCH_DIR: Path = Field(
        default=Path("temp_batch/"),
        description="Directorio temporal para batch processing",
    )
    SUMMARY_OUTPUT_DIR: Path = Field(
        default=Path("output/summaries/"),
        description="Directorio para resúmenes generados",
    )

    # Transcript cache (to speed up iterative runs of the script generator)
    TRANSCRIPT_CACHE_ENABLED: bool = Field(
        default=False,
        description="Habilita cacheo/reutilización de transcripciones por video_id",
    )
    TRANSCRIPT_CACHE_DIR: Path = Field(
        default=Path("output/analysis/transcripts_cache/"),
        description="Directorio para cachear transcripciones por video_id",
    )

    # YouTube search configuration
    YT_SEARCH_TIMEOUT_SECONDS: int = Field(
        default=180,
        description="Timeout (segundos) para la búsqueda con yt-dlp",
    )

    # ========== CONTENT IDEATION ENGINE CONFIGURATION ==========

    # SerpAPI for search volume data
    SERPAPI_API_KEY: str = Field(
        default="",
        description="SerpAPI key para volumen de búsqueda (2500 requests/mes gratis)",
    )

    # Reddit API for trending discussions
    REDDIT_CLIENT_ID: str = Field(
        default="",
        description="Reddit API client ID (https://www.reddit.com/prefs/apps)",
    )
    REDDIT_CLIENT_SECRET: str = Field(
        default="",
        description="Reddit API client secret",
    )
    REDDIT_USER_AGENT: str = Field(
        default="yt-content-ideation/1.0",
        description="Reddit API user agent",
    )

    # Trends output directory
    TRENDS_OUTPUT_DIR: Path = Field(
        default=Path("output/trends/"),
        description="Directorio para análisis de tendencias",
    )

    # ========== POST KITS VALIDATION (PARAMETRIZABLE) ==========
    # LinkedIn
    POST_KITS_LINKEDIN_MIN_CHARS: int = Field(
        default=800,
        description="Mínimo de caracteres para el post de LinkedIn",
    )
    POST_KITS_LINKEDIN_MAX_CHARS: int = Field(
        default=1200,
        description="Máximo de caracteres para el post de LinkedIn",
    )
    POST_KITS_LINKEDIN_MIN_INSIGHTS: int = Field(
        default=4,
        description="Mínimo de insights en el post de LinkedIn",
    )
    POST_KITS_LINKEDIN_MAX_INSIGHTS: int = Field(
        default=8,
        description="Máximo de insights en el post de LinkedIn",
    )

    # Twitter / X
    POST_KITS_TWITTER_MIN_TWEETS: int = Field(
        default=8,
        description="Mínimo de tweets por hilo",
    )
    POST_KITS_TWITTER_MAX_TWEETS: int = Field(
        default=12,
        description="Máximo de tweets por hilo",
    )
    POST_KITS_TWITTER_MAX_CHARS_PER_TWEET: int = Field(
        default=280,
        description="Máximo de caracteres por tweet",
    )
    POST_KITS_TWITTER_MAX_HASHTAGS: int = Field(
        default=3,
        description="Máximo de hashtags permitidos (se añaden al último tweet)",
    )


# Crear una instancia global de las configuraciones validadas
try:
    settings = AppSettings()

    # Aliases de entorno para compatibilidad retroactiva
    # OUTPUT_TRENDS_DIR -> TRENDS_OUTPUT_DIR
    trends_alias = os.getenv("OUTPUT_TRENDS_DIR")
    if trends_alias:
        # Mantener el valor por defecto si el alias no es válido
        with contextlib.suppress(Exception):
            settings.TRENDS_OUTPUT_DIR = Path(trends_alias)

except Exception as e:
    sys.stderr.write(f"CRITICAL: Error al cargar o validar la configuración: {e}\n")
    sys.exit(1)


__all__ = ["AppSettings", "settings"]
