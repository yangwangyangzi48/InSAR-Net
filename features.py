import numpy as np


def wrap_phase(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def complex_interferogram(z1_real, z1_imag, z2_real, z2_imag):
    real = z1_real * z2_real + z1_imag * z2_imag
    imag = z1_imag * z2_real - z1_real * z2_imag
    return real, imag


def slc_pair_to_tensor(z1_real, z1_imag, z2_real, z2_imag):
    return np.stack([z1_real, z1_imag, z2_real, z2_imag], axis=-1).astype(np.float32)


def phase_to_target(phase):
    phase = wrap_phase(phase)
    return np.stack([np.cos(phase), np.sin(phase)], axis=-1).astype(np.float32)


def target_to_phase(target):
    return np.arctan2(target[..., 1], target[..., 0]).astype(np.float32)


def normalize_complex_pair(x):
    y = x.astype(np.float32).copy()
    for i in range(y.shape[-1]):
        band = y[..., i]
        scale = np.percentile(np.abs(band), 99.5)
        if scale > 0:
            y[..., i] = np.clip(band / scale, -1.0, 1.0)
    return y
