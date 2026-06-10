import json
import os
import numpy as np
import tensorflow as tf
from .losses import phase_from_vector, wrap


def evaluate_dataset(model, dataset):
    values = []
    for x, y in dataset:
        pred = model(x, training=False)
        diff = wrap(phase_from_vector(y) - phase_from_vector(pred))
        values.append([float(tf.reduce_mean(tf.abs(diff)).numpy()), float(tf.sqrt(tf.reduce_mean(tf.square(diff))).numpy())])
    arr = np.array(values, dtype=np.float32)
    return {"phase_mae": float(np.mean(arr[:, 0])), "phase_rmse": float(np.mean(arr[:, 1])), "batches": int(len(values))}


def save_metrics(metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
