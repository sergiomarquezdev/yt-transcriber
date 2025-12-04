"""Utilities for yt_transcriber module.

Core functions (normalize_title_for_filename, ensure_dir_exists) are re-exported
from core.utils for backwards compatibility.
"""

import logging
import shutil
from pathlib import Path

# Re-export core utilities for backwards compatibility
from core.utils import ensure_dir_exists, normalize_title_for_filename


logger = logging.getLogger(__name__)

__all__ = [
    "normalize_title_for_filename",
    "ensure_dir_exists",
    "save_transcription_to_file",
    "cleanup_temp_files",
    "cleanup_temp_dir",
    "get_file_size_mb",
]


def save_transcription_to_file(
    transcription_text: str,
    output_filename_no_ext: str,
    output_dir: Path,
    original_title: str | None = None,
) -> Path | None:
    """
    Guarda el texto de la transcripción en un archivo .txt.

    Returns:
        La ruta completa al archivo guardado, o None si ocurre un error.
    """
    try:
        ensure_dir_exists(output_dir)

        safe_filename = "".join(
            c if c.isalnum() or c in (".", "_") else "_" for c in output_filename_no_ext
        ).strip(" .")

        if not safe_filename:
            safe_filename = f"default_transcription_{output_filename_no_ext[:10]}"

        file_path = output_dir / f"{safe_filename}.txt"

        content_to_write = transcription_text
        if original_title:
            content_to_write = f"# Original Video Title: {original_title}\n\n{transcription_text}"

        file_path.write_text(content_to_write, encoding="utf-8")
        logger.info(f"Transcripción guardada en: {file_path}")
        return file_path
    except Exception as e:
        logger.error(
            f"Error al guardar la transcripción para '{output_filename_no_ext}' en '{output_dir}': {e}",
            exc_info=True,
        )
        return None


def cleanup_temp_files(file_paths_to_delete: list[str | None]):
    """
    Elimina una lista de archivos temporalmente.

    Args:
        file_paths_to_delete: Una lista de rutas de archivos a eliminar.
                              Puede contener Nones, que serán ignorados.
    """
    cleaned_count = 0
    valid_paths_to_check = [Path(p) for p in file_paths_to_delete if p]

    for file_path in valid_paths_to_check:
        if file_path.exists():
            try:
                file_path.unlink()
                logger.info(f"Archivo temporal eliminado: {file_path}")
                cleaned_count += 1
            except OSError as e:
                logger.error(f"Error al eliminar el archivo temporal {file_path}: {e}")
        else:
            logger.warning(f"Se intentó limpiar el archivo temporal {file_path}, pero no existe.")
    logger.info(
        f"Limpieza de archivos temporales: {cleaned_count} archivo(s) eliminado(s) de {len(valid_paths_to_check)} solicitado(s) (existentes)."
    )


def cleanup_temp_dir(temp_dir_path: Path):
    """
    Elimina completamente el directorio temporal y todo su contenido.
    """
    try:
        if temp_dir_path.exists() and temp_dir_path.is_dir():
            shutil.rmtree(temp_dir_path)
            logger.info(f"Directorio temporal eliminado: {temp_dir_path}")
    except Exception as e:
        logger.error(
            f"Error al limpiar el directorio temporal {temp_dir_path}: {e}",
            exc_info=True,
        )


def get_file_size_mb(file_path: Path) -> float | None:
    """
    Obtiene el tamaño de un archivo en megabytes.
    """
    if file_path.exists():
        return file_path.stat().st_size / (1024 * 1024)
    return None
