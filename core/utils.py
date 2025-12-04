"""Core utilities for file and path operations.

These functions are used across multiple modules (yt_transcriber, yt_script_generator)
and are centralized here to avoid circular dependencies.
"""

import logging
import re
from pathlib import Path


logger = logging.getLogger(__name__)


def normalize_title_for_filename(text: str) -> str:
    """
    Normaliza un texto para ser usado como parte de un nombre de archivo.
    - Conserva alfanuméricos, espacios y guiones.
    - Elimina caracteres especiales y emojis.
    - Reemplaza espacios múltiples con un solo espacio.
    - Reemplaza espacios con guiones bajos.
    - Elimina guiones bajos al inicio/final.
    """
    if not text:
        return "untitled"

    # Eliminar caracteres especiales y emojis, pero mantener espacios
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)

    # Reemplazar espacios múltiples con un solo espacio
    text = re.sub(r"\s+", " ", text)

    # Reemplazar espacios con guiones bajos
    text = text.replace(" ", "_")

    # Eliminar guiones bajos al inicio y final
    text = text.strip("_")

    return text or "untitled"


def ensure_dir_exists(dir_path: Path) -> None:
    """
    Asegura que un directorio exista. Si no, lo crea.
    """
    if not dir_path.exists():
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directorio creado: {dir_path}")
        except OSError as e:
            logger.error(f"Error al crear el directorio {dir_path}: {e}")
            raise
    else:
        logger.debug(f"Directorio ya existe: {dir_path}")
