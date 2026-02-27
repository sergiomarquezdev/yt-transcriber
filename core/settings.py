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

    # YouTube Script Generator settings
    GOOGLE_API_KEY: str = Field(
        default="",
        description="Google Gemini API key para script generation",
    )
    OPENAI_API_KEY: str = Field(
        default="",
        description="OpenAI API key para modelos GPT (opcional)",
    )
    ANTHROPIC_API_KEY: str = Field(
        default="",
        description="Anthropic API key para modelos Claude (opcional)",
    )

    # ========== GEMINI MODEL CONFIGURATION (Optimized for Quality + Cost) ==========

    # For script synthesis and generation (CRITICAL - needs max quality)
    PRO_MODEL: str = Field(
        default="gemini-2.5-pro",
        description="Modelo premium para synthesis y script generation ($2.50/$15.00)",
    )

    # For video summarization (balanced quality, free tier)
    SUMMARIZER_MODEL: str = Field(
        default="gemini-2.5-flash",
        description="Modelo para resúmenes de video (free tier, excelente calidad)",
    )

    # For pattern analysis from transcripts (10 videos, free tier)
    PATTERN_ANALYZER_MODEL: str = Field(
        default="gemini-2.5-flash",
        description="Modelo para análisis de patrones (free tier, suficiente calidad)",
    )

    # For translations (summaries use lite, scripts use flash)
    TRANSLATOR_MODEL: str = Field(
        default="gemini-2.5-flash-lite",
        description="Modelo para traducciones simples ($0.075/$0.30)",
    )

    # For simple tasks (query optimization)
    QUERY_OPTIMIZER_MODEL: str = Field(
        default="gemini-2.5-flash-lite",
        description="Modelo para optimización de queries ($0.075/$0.30)",
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

    # ========== LLM CACHING & RATE LIMITING CONFIGURATION ==========

    # LLM Cache settings
    LLM_CACHE_ENABLED: bool = Field(
        default=True,
        description="Enable/disable LLM response caching",
    )
    LLM_CACHE_TTL_DAYS: int = Field(
        default=7,
        description="Cache time-to-live in days",
    )
    LLM_CACHE_DIR: Path = Field(
        default=Path("output/.llm_cache/"),
        description="Directory for LLM cache storage",
    )

    # Simple rate limiting (QPS) para llamadas LLM (0 = deshabilitado)
    LLM_QPS: int = Field(
        default=0,
        description="Límite de solicitudes por minuto a LLM (0 = sin límite)",
    )

    # Prompt versioning (for cache invalidation)
    POST_KITS_PROMPT_VERSION: str = Field(
        default="v1.0",
        description="Post Kits prompt template version",
    )
    SUMMARIZER_PROMPT_VERSION: str = Field(
        default="v1.0",
        description="Summarizer prompt template version",
    )
    PATTERN_ANALYZER_PROMPT_VERSION: str = Field(
        default="v1.0",
        description="Pattern Analyzer prompt template version",
    )
    SYNTHESIZER_PROMPT_VERSION: str = Field(
        default="v1.0",
        description="Pattern Synthesizer prompt template version",
    )
    SCRIPT_GENERATOR_PROMPT_VERSION: str = Field(
        default="v1.0",
        description="Script Generator prompt template version",
    )
    TRANSLATOR_PROMPT_VERSION: str = Field(
        default="v1.0",
        description="Translator prompt template version",
    )
    INSIGHT_GENERATOR_PROMPT_VERSION: str = Field(
        default="v1.0",
        description="Insight Generator prompt template version",
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

    # LLM_CACHE_PATH -> LLM_CACHE_DIR
    cache_alias = os.getenv("LLM_CACHE_PATH")
    if cache_alias:
        with contextlib.suppress(Exception):
            settings.LLM_CACHE_DIR = Path(cache_alias)

except Exception as e:
    sys.stderr.write(f"CRITICAL: Error al cargar o validar la configuración: {e}\n")
    sys.exit(1)


__all__ = ["AppSettings", "settings"]
