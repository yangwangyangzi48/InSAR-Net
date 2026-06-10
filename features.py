"""Phase preprocessing and feature construction."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    """Wrap phase to [-pi, pi]."""
    return (phase + np.pi) % (2 * np.pi) - np.pi


def estimate_coherence(phase_image: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Estimate local coherence from a wrapped phase image.

    This is a local phase-consistency proxy computed as the magnitude of the
    local mean complex phasor.
    """
    complex_image = np.exp(1j * phase_image)
    real_filtered = uniform_filter(np.real(complex_image), size=window_size)
    imag_filtered = uniform_filter(np.imag(complex_image), size=window_size)
    return np.abs(real_filtered + 1j * imag_filtered).astype(np.float32)


def crop_or_pad_center(image: np.ndarray, target_size: int) -> np.ndarray:
    """Center crop or zero-pad an image to ``target_size x target_size``."""
    h, w = image.shape
    if h == target_size and w == target_size:
        return image.astype(np.float32)

    if h >= target_size and w >= target_size:
        start_h = (h - target_size) // 2
        start_w = (w - target_size) // 2
        out = image[start_h : start_h + target_size, start_w : start_w + target_size]
        return out.astype(np.float32)

    out = np.zeros((target_size, target_size), dtype=np.float32)
    start_h = max(0, (target_size - h) // 2)
    start_w = max(0, (target_size - w) // 2)
    h_to_copy = min(h, target_size)
    w_to_copy = min(w, target_size)
    out[start_h : start_h + h_to_copy, start_w : start_w + w_to_copy] = image[:h_to_copy, :w_to_copy]
    return out


def phase_to_sincos(phase: np.ndarray) -> np.ndarray:
    """Convert phase to two channels: [sin, cos]."""
    return np.stack([np.sin(phase), np.cos(phase)], axis=-1).astype(np.float32)


def sincos_to_phase(sincos: np.ndarray) -> np.ndarray:
    """Convert [sin, cos] channels to wrapped phase."""
    return np.arctan2(sincos[..., 0], sincos[..., 1]).astype(np.float32)


def normalize_phase_features(
    image: np.ndarray,
    include_coherence: bool = True,
    include_frequency: bool = True,
    coherence_window_size: int = 5,
) -> np.ndarray:
    """Build model input features from wrapped phase.

    Channels are: sin, cos, normalized gradient magnitude, optional coherence,
    and optional normalized log-amplitude spectrum.
    """
    image = image.astype(np.float32)

    grad_y, grad_x = np.gradient(image)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2).astype(np.float32)
    max_grad = float(np.max(grad_magnitude))
    if max_grad > 0:
        grad_magnitude /= max_grad

    channels = [
        np.sin(image)[..., np.newaxis].astype(np.float32),
        np.cos(image)[..., np.newaxis].astype(np.float32),
        grad_magnitude[..., np.newaxis].astype(np.float32),
    ]

    if include_coherence:
        coherence = estimate_coherence(image, window_size=coherence_window_size)
        channels.append(np.clip(coherence, 0.0, 1.0)[..., np.newaxis].astype(np.float32))

    if include_frequency:
        fft_image = np.fft.fft2(image)
        fft_shifted = np.fft.fftshift(fft_image)
        log_amplitude = np.log(np.abs(fft_shifted) + 1e-8)
        min_val = float(np.min(log_amplitude))
        max_val = float(np.max(log_amplitude))
        norm_amp = (log_amplitude - min_val) / (max_val - min_val + 1e-8)
        channels.append(norm_amp[..., np.newaxis].astype(np.float32))

    return np.concatenate(channels, axis=-1).astype(np.float32)
