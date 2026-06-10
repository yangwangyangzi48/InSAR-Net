"""Visualization utilities."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from .config import PhaseFilterConfig
from .evaluation import wrapped_phase_rmse
from .inference import predict_phase
from .io import load_phase_tif


def visualize_results(
    model,
    test_noise_files: list[str],
    test_clean_files: list[str],
    evaluation_dir: str,
    config: PhaseFilterConfig,
    num_samples: int = 5,
) -> None:
    """Generate visual comparison figures for random test samples."""
    os.makedirs(evaluation_dir, exist_ok=True)
    np.random.seed(config.random_seed)
    indices = np.random.choice(len(test_noise_files), min(num_samples, len(test_noise_files)), replace=False)

    for i, idx in enumerate(indices):
        noise_path = test_noise_files[idx]
        clean_path = test_clean_files[idx]

        clean_phase = load_phase_tif(clean_path)
        noisy_phase, model_output, enhanced_phase, coherence = predict_phase(model, noise_path, config)

        rmse_before = wrapped_phase_rmse(clean_phase, noisy_phase)
        rmse_after = wrapped_phase_rmse(clean_phase, model_output)
        rmse_enhanced = wrapped_phase_rmse(clean_phase, enhanced_phase)
        enhanced_improvement = (rmse_before - rmse_enhanced) / (rmse_before + 1e-8) * 100

        fig = plt.figure(figsize=(15, 10))

        plt.subplot(2, 3, 1)
        plt.imshow(noisy_phase, cmap="jet", vmin=-np.pi, vmax=np.pi)
        plt.colorbar()
        plt.title("Noisy Phase Input")

        plt.subplot(2, 3, 2)
        plt.imshow(clean_phase, cmap="jet", vmin=-np.pi, vmax=np.pi)
        plt.colorbar()
        plt.title("Clean Phase Target")

        plt.subplot(2, 3, 3)
        plt.imshow(model_output, cmap="jet", vmin=-np.pi, vmax=np.pi)
        plt.colorbar()
        plt.title(f"Model Output\nRMSE: {rmse_after:.4f} rad")

        plt.subplot(2, 3, 4)
        error_before = np.minimum(np.abs(clean_phase - noisy_phase), 2 * np.pi - np.abs(clean_phase - noisy_phase))
        plt.imshow(error_before, cmap="inferno", vmin=0, vmax=np.pi / 2)
        plt.colorbar()
        plt.title(f"Error Before\nRMSE: {rmse_before:.4f} rad")

        plt.subplot(2, 3, 5)
        plt.imshow(coherence, cmap="viridis", vmin=0, vmax=1)
        plt.colorbar()
        plt.title("Coherence Map")

        plt.subplot(2, 3, 6)
        plt.imshow(enhanced_phase, cmap="jet", vmin=-np.pi, vmax=np.pi)
        plt.colorbar()
        plt.title(f"Post-processed Output\nRMSE: {rmse_enhanced:.4f} rad")

        plt.suptitle(
            f"InSAR Phase Filter Results - Sample {i + 1}\nImprovement: {enhanced_improvement:.2f}%",
            fontsize=16,
        )
        plt.tight_layout()
        fig.savefig(os.path.join(evaluation_dir, f"result_{i + 1}.png"), dpi=200)
        plt.close(fig)
