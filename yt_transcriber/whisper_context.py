import gc
import logging
import os
import sys
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from core.settings import settings

# Ensure NVIDIA CUDA DLLs (cublas, cudnn) installed via pip are discoverable.
# On Windows, CTranslate2 loads them via the DLL search path at runtime.
_site_packages = next((p for p in sys.path if p.endswith("site-packages")), None)
if _site_packages:
    for _lib in ("nvidia/cublas/bin", "nvidia/cudnn/bin"):
        _dll_dir = os.path.join(_site_packages, _lib)
        if os.path.isdir(_dll_dir):
            os.add_dll_directory(_dll_dir)
            os.environ["PATH"] = _dll_dir + os.pathsep + os.environ.get("PATH", "")
        elif sys.platform == "win32":
            import logging as _logging

            _logging.getLogger(__name__).warning(
                f"Expected NVIDIA DLL dir not found: {_dll_dir}. "
                "If using CUDA, install with: "
                "pip install nvidia-cublas-cu12 'nvidia-cudnn-cu12==9.*'"
            )

logger = logging.getLogger(__name__)

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


def _resolve_compute_type(device: str, compute_type: str) -> str:
    """Auto-select optimal compute type based on device if set to 'default'."""
    if compute_type != "default":
        return compute_type
    return "int8_float16" if device == "cuda" else "int8"


def _resolve_device(device: str) -> str:
    """Resolve 'auto' device to 'cuda' or 'cpu' by probing CTranslate2."""
    if device != "auto":
        return device
    try:
        import ctranslate2

        supported = ctranslate2.get_supported_compute_types("cuda")
        if supported:
            return "cuda"
    except Exception:
        pass
    return "cpu"


@contextmanager
def whisper_model_context() -> Generator[Any]:
    """Context manager for loading and unloading the faster-whisper model.

    Ensures that memory is released after use.

    Yields:
        model: The loaded WhisperModel instance.
    """
    if WhisperModel is None:
        logger.critical("Dependency 'faster-whisper' not found.")
        raise ImportError("Missing dependency: faster-whisper")

    device = _resolve_device(settings.WHISPER_DEVICE)
    compute_type = _resolve_compute_type(device, settings.WHISPER_COMPUTE_TYPE)

    model = None
    try:
        logger.info(
            f"Loading faster-whisper model '{settings.WHISPER_MODEL_NAME}' "
            f"on {device} (compute_type={compute_type})..."
        )
        model = WhisperModel(
            settings.WHISPER_MODEL_NAME,
            device=device,
            compute_type=compute_type,
        )
        logger.info("Whisper model loaded successfully.")
        yield model
    except Exception as e:
        logger.critical(f"Failed to load Whisper model: {e}", exc_info=True)
        raise
    finally:
        if model:
            logger.info("Unloading Whisper model...")
            del model

        gc.collect()
        logger.info("Whisper model unloaded and memory released.")
