
import logging
import gc
from contextlib import contextmanager
from typing import Generator, Any

from core.settings import settings

logger = logging.getLogger(__name__)

# Try to import torch/whisper (may not be installed in all envs, handle gracefully)
try:
    import torch
    import whisper
except ImportError:
    torch = None
    whisper = None

@contextmanager
def whisper_model_context() -> Generator[Any, None, None]:
    """
    Context manager for loading and unloading the Whisper model.
    Ensures that memory (VRAM/RAM) is released after use.
    
    Yields:
        model: The loaded Whisper model.
    """
    if whisper is None or torch is None:
        logger.critical("Dependencies 'torch' or 'whisper' not found.")
        raise ImportError("Missing dependencies: torch, whisper")

    device = settings.WHISPER_DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    model = None
    try:
        logger.info(f"Loading Whisper model '{settings.WHISPER_MODEL_NAME}' on {device}...")
        model = whisper.load_model(settings.WHISPER_MODEL_NAME, device=device)
        logger.info("Whisper model loaded successfully.")
        yield model
    except Exception as e:
        logger.critical(f"Failed to load Whisper model: {e}", exc_info=True)
        raise
    finally:
        if model:
            logger.info("Unloading Whisper model...")
            del model
            
        # Force garbage collection
        gc.collect()
        
        # Release CUDA memory if used
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Whisper model unloaded and memory released.")
