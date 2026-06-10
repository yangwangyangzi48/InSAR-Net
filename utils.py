"""General utilities."""

from __future__ import annotations

import logging
import os
import random

import numpy as np


def setup_logger(name: str = "insar_filter", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a project logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def set_global_seed(seed: int) -> None:
    """Set Python, NumPy, and TensorFlow seeds when TensorFlow is available."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


logger = setup_logger()
