"""TIF input/output utilities."""

from __future__ import annotations

import glob
import os
from typing import Iterable

import numpy as np
import tifffile as tiff

from .utils import logger


def list_tif_files(folder: str) -> list[str]:
    """List tif/tiff files in a folder."""
    files: list[str] = []
    for ext in ("*.tif", "*.tiff"):
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


def load_phase_tif(path: str) -> np.ndarray:
    """Load a single-band phase TIF as ``float32``.

    Multi-band inputs are reduced to the first band. NaN and Inf values are
    converted to 0 to keep TensorFlow pipelines stable.
    """
    image = tiff.imread(path).astype(np.float32)

    if image.ndim > 2:
        if image.shape[-1] == 1:
            image = np.squeeze(image, axis=-1)
        else:
            image = image[..., 0]

    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    return image.astype(np.float32)


def save_phase_tif(path: str, phase: np.ndarray) -> None:
    """Save a phase array to TIF."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tiff.imwrite(path, phase.astype(np.float32))


def load_data_pairs(noise_dir: str, clean_dir: str, max_files: int | None = None) -> tuple[list[str], list[str]]:
    """Load matched noisy/clean TIF paths by sorted order.

    This follows the behavior of the original script. If exact filename matching
    is required, rename files consistently or customize this function.
    """
    logger.info("Loading data from %s and %s", noise_dir, clean_dir)

    noise_files = list_tif_files(noise_dir)
    clean_files = list_tif_files(clean_dir)

    if not noise_files:
        raise FileNotFoundError(f"No tif files found in noisy directory: {noise_dir}")
    if not clean_files:
        raise FileNotFoundError(f"No tif files found in clean directory: {clean_dir}")

    pair_count = min(len(noise_files), len(clean_files))
    if max_files is not None:
        pair_count = min(pair_count, int(max_files))

    noise_files = noise_files[:pair_count]
    clean_files = clean_files[:pair_count]

    logger.info("Loaded %d matched InSAR image pairs", pair_count)
    return noise_files, clean_files


def ensure_same_length(*items: Iterable[object]) -> None:
    """Validate that several collections have the same length."""
    lengths = [len(x) for x in items]
    if len(set(lengths)) != 1:
        raise ValueError(f"Input collections must have the same length, got {lengths}")
