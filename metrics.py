import numpy as np
import tensorflow as tf
from .losses import phase_from_vector, wrap


def phase_mae(y_true, y_pred):
    diff = wrap(phase_from_vector(y_true) - phase_from_vector(y_pred))
    return tf.reduce_mean(tf.abs(diff))


def phase_rmse(y_true, y_pred):
    diff = wrap(phase_from_vector(y_true) - phase_from_vector(y_pred))
    return tf.sqrt(tf.reduce_mean(tf.square(diff)))


def np_phase_metrics(true_phase, pred_phase):
    diff = np.angle(np.exp(1j * (true_phase - pred_phase)))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    psnr = float(20.0 * np.log10(np.pi / max(rmse, 1e-8)))
    return {"mae": mae, "rmse": rmse, "psnr": psnr}
