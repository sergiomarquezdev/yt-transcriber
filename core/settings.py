"""Shared application settings for yt-transcriber.

This module contains the validated Pydantic settings for the CLI/TUI.
"""

import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env if present
load_dotenv()


class AppSettings(BaseSettings):
    """Validated application settings."""

    model_config = SettingsConfigDict(case_sensitive=False)

    # ========== WHISPER ==========
    WHISPER_MODEL_NAME: str = Field(
        default="base",
        description="Whisper model (base, small, medium, large-v3, distil-large-v3, ...)",
    )
    WHISPER_DEVICE: Literal["cpu", "cuda", "auto"] = Field(
        default="auto",
        description="Compute device (auto detects CUDA via CTranslate2)",
    )
    WHISPER_COMPUTE_TYPE: str = Field(
        default="default",
        description="CTranslate2 compute type (int8_float16 for GPU, int8 for CPU)",
    )
    WHISPER_BEAM_SIZE: int = Field(
        default=5,
        description="Beam search size (1=greedy, 5=default)",
    )
    WHISPER_VAD_FILTER: bool = Field(
        default=True,
        description="Silero VAD filter to skip silences",
    )

    # ========== PATHS ==========
    TEMP_DOWNLOAD_DIR: Path = Field(
        default=Path("temp_files/"),
        description="Directory for temporary files",
    )
    OUTPUT_BASE_DIR: Path = Field(
        default=Path("output/"),
        description="Base directory; each video gets its own subfolder here",
    )

    # ========== LOGGING / FFMPEG ==========
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    FFMPEG_LOCATION: str = Field(
        default="",
        description="Custom FFmpeg path (optional)",
    )

    # ========== TRANSCRIPT ARTIFACTS ==========
    TRANSCRIPT_SEGMENTS_ENABLED: bool = Field(
        default=False,
        description="Emit timestamped segments JSON sidecar",
    )
    VISUAL_EVIDENCE_ENABLED: bool = Field(
        default=False,
        description="Extract one frame per transcript segment (local files only)",
    )
    VISUAL_EVIDENCE_MIN_SEGMENT_SECONDS: float = Field(
        default=1.0,
        description="Minimum segment duration to consider visual evidence",
    )

    # ========== YT-DLP ==========
    YT_SEARCH_TIMEOUT_SECONDS: int = Field(
        default=180,
        description="Timeout (seconds) for yt-dlp search/info calls",
    )


# Build the global instance
try:
    settings = AppSettings()
except Exception as e:
    sys.stderr.write(f"CRITICAL: Failed to load/validate settings: {e}\n")
    sys.exit(1)


__all__ = ["AppSettings", "settings"]
