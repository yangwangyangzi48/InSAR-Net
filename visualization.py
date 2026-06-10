import os
import matplotlib.pyplot as plt
import numpy as np
from .features import target_to_phase


def save_prediction_figure(x, y, pred, path):
    phase_true = target_to_phase(y)
    phase_pred = target_to_phase(pred)
    phase_input = np.angle((x[..., 0] + 1j * x[..., 1]) * np.conj(x[..., 2] + 1j * x[..., 3]))
    err = np.abs(np.angle(np.exp(1j * (phase_true - phase_pred))))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(12, 8))
    panels = [(phase_input, "Input"), (phase_true, "Target"), (phase_pred, "Prediction"), (err, "Error")]
    for i, item in enumerate(panels):
        plt.subplot(2, 2, i + 1)
        plt.imshow(item[0], cmap="jet", vmin=-np.pi if i < 3 else 0, vmax=np.pi if i < 3 else np.pi)
        plt.title(item[1])
        plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
