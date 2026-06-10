"""TensorFlow Dataset construction."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from .config import PhaseFilterConfig
from .features import crop_or_pad_center, normalize_phase_features, phase_to_sincos
from .io import load_phase_tif


def preprocess_input(path: str, config: PhaseFilterConfig) -> np.ndarray:
    """Load noisy phase and convert it to model input features."""
    image = load_phase_tif(path)
    image = crop_or_pad_center(image, config.image_size)
    return normalize_phase_features(
        image,
        include_coherence=config.include_coherence,
        include_frequency=config.include_frequency,
        coherence_window_size=config.coherence_window_size,
    )


def preprocess_target(path: str, config: PhaseFilterConfig) -> np.ndarray:
    """Load clean phase and convert it to [sin, cos] target."""
    image = load_phase_tif(path)
    image = crop_or_pad_center(image, config.image_size)
    return phase_to_sincos(image)


def data_augmentation(noise_img: tf.Tensor, clean_img: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply synchronized geometric augmentation to input and target."""
    num_channels = tf.shape(noise_img)[-1]
    stacked = tf.concat([noise_img, clean_img], axis=-1)

    if tf.random.uniform(()) > 0.5:
        stacked = tf.image.flip_left_right(stacked)
    if tf.random.uniform(()) > 0.5:
        stacked = tf.image.flip_up_down(stacked)

    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    stacked = tf.image.rot90(stacked, k=k)

    # Add slight noise only to the input feature channels.
    if tf.random.uniform(()) > 0.7:
        noise_sigma = 0.03
        random_noise = tf.random.normal(tf.shape(stacked), mean=0.0, stddev=noise_sigma)
        input_noise = random_noise[..., :num_channels] * 0.5
        target_noise = random_noise[..., num_channels:] * 0.0
        stacked = stacked + tf.concat([input_noise, target_noise], axis=-1)

    return stacked[..., :num_channels], stacked[..., num_channels:]


def create_dataset(
    noise_files: list[str],
    clean_files: list[str],
    config: PhaseFilterConfig,
    is_training: bool = True,
) -> tf.data.Dataset:
    """Create a TensorFlow dataset for training, validation, or testing."""

    def _process_path(noise_path: tf.Tensor, clean_path: tf.Tensor):
        def _load_pair(noise_path_np, clean_path_np):
            noise_path_str = noise_path_np.decode("utf-8")
            clean_path_str = clean_path_np.decode("utf-8")
            x = preprocess_input(noise_path_str, config)
            y = preprocess_target(clean_path_str, config)
            return x.astype(np.float32), y.astype(np.float32)

        x, y = tf.py_function(_load_pair, [noise_path, clean_path], [tf.float32, tf.float32])
        x.set_shape([config.image_size, config.image_size, config.input_channels])
        y.set_shape([config.image_size, config.image_size, config.output_channels])
        return x, y

    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(noise_files), tf.constant(clean_files)))
    dataset = dataset.map(_process_path, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        buffer_size = min(len(noise_files), 1000)
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=config.random_seed, reshuffle_each_iteration=True)
        dataset = dataset.map(data_augmentation, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(config.batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
