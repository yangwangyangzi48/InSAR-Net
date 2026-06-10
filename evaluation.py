"""Evaluation metrics."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from .utils import logger


def wrapped_phase_rmse(phase_true: np.ndarray, phase_pred: np.ndarray) -> float:
    """RMSE between wrapped phases."""
    diff = np.abs(phase_true - phase_pred)
    wrapped_diff = np.minimum(diff, 2 * np.pi - diff)
    return float(np.sqrt(np.mean(wrapped_diff**2)))


def evaluate_model(model: tf.keras.Model, test_dataset) -> dict[str, float | int]:
    """Evaluate the model with phase-aware metrics."""
    logger.info("Evaluating model on test data...")

    test_maes: list[float] = []
    phase_rmses: list[float] = []

    for x_test, y_test in test_dataset:
        y_pred = model(x_test, training=False)
        mae = tf.reduce_mean(tf.abs(y_test - y_pred))
        test_maes.append(float(mae.numpy()))

        y_test_phase = tf.math.atan2(y_test[..., 0], y_test[..., 1])
        y_pred_phase = tf.math.atan2(y_pred[..., 0], y_pred[..., 1])
        phase_diff = tf.abs(y_test_phase - y_pred_phase)
        phase_diff = tf.minimum(phase_diff, 2 * np.pi - phase_diff)
        rmse = tf.sqrt(tf.reduce_mean(tf.square(phase_diff)))
        phase_rmses.append(float(rmse.numpy()))

    mean_mae = float(np.mean(test_maes)) if test_maes else np.nan
    mean_phase_rmse = float(np.mean(phase_rmses)) if phase_rmses else np.nan

    metrics = {
        "mae": mean_mae,
        "phase_rmse_rad": mean_phase_rmse,
        "phase_rmse_deg": float(mean_phase_rmse * 180 / np.pi) if np.isfinite(mean_phase_rmse) else np.nan,
        "batches_evaluated": int(len(test_maes)),
    }

    for key, value in metrics.items():
        logger.info("%s: %s", key, value)
    return metrics
