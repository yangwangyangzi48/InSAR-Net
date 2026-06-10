"""Inference and post-processing."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter

from .config import PhaseFilterConfig
from .features import estimate_coherence, sincos_to_phase, wrap_phase
from .dataset import preprocess_input
from .io import load_phase_tif, save_phase_tif


def phase_filter_post_processing(phase_image: np.ndarray, coherence: np.ndarray | None = None, sigma: float = 1.5) -> np.ndarray:
    """Coherence-guided, phase-preserving Gaussian post-processing."""
    complex_phase = np.exp(1j * phase_image)
    if coherence is None:
        coherence = estimate_coherence(phase_image)

    filter_strength = np.clip(1.0 - coherence, 0.1, 0.9)
    sigma_var = float(sigma * np.mean(filter_strength))

    real_filtered = gaussian_filter(np.real(complex_phase), sigma=sigma_var)
    imag_filtered = gaussian_filter(np.imag(complex_phase), sigma=sigma_var)
    filtered_complex = real_filtered + 1j * imag_filtered
    filtered_complex = filtered_complex / (np.abs(filtered_complex) + 1e-8)
    return np.angle(filtered_complex).astype(np.float32)


def predict_phase(model: tf.keras.Model, noisy_phase_path: str, config: PhaseFilterConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Predict filtered phase for one TIF.

    Returns original noisy phase, model output, post-processed phase, and local
    coherence estimated from the original noisy phase.
    """
    model_input = preprocess_input(noisy_phase_path, config)
    original_phase = load_phase_tif(noisy_phase_path)

    pred = model.predict(np.expand_dims(model_input, axis=0), verbose=0)[0]
    model_phase = sincos_to_phase(pred)
    coherence = estimate_coherence(original_phase, window_size=config.coherence_window_size)
    refined_phase = wrap_phase(phase_filter_post_processing(model_phase, coherence))
    return original_phase, model_phase, refined_phase, coherence


def save_prediction_outputs(
    output_prefix: str,
    original_phase: np.ndarray,
    model_phase: np.ndarray,
    refined_phase: np.ndarray,
    coherence: np.ndarray,
) -> None:
    """Save prediction outputs as TIF files."""
    save_phase_tif(f"{output_prefix}_noisy.tif", original_phase)
    save_phase_tif(f"{output_prefix}_model_output.tif", model_phase)
    save_phase_tif(f"{output_prefix}_postprocessed.tif", refined_phase)
    save_phase_tif(f"{output_prefix}_coherence.tif", coherence)
