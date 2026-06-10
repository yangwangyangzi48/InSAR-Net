import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from .io import read_complex_tif, read_phase_tif, center_crop_or_pad, match_triplets
from .features import slc_pair_to_tensor, phase_to_target, normalize_complex_pair


def split_triplets(triplets, validation_split, test_split, seed):
    train_val, test = train_test_split(triplets, test_size=test_split, random_state=seed)
    val_ratio = validation_split / max(1.0 - test_split, 1e-6)
    train, val = train_test_split(train_val, test_size=val_ratio, random_state=seed)
    return train, val, test


def load_sample(slc1_path, slc2_path, target_path, image_size):
    slc1_path = slc1_path.numpy().decode("utf-8")
    slc2_path = slc2_path.numpy().decode("utf-8")
    target_path = target_path.numpy().decode("utf-8")
    z1r, z1i = read_complex_tif(slc1_path)
    z2r, z2i = read_complex_tif(slc2_path)
    phase = read_phase_tif(target_path)
    x = slc_pair_to_tensor(z1r, z1i, z2r, z2i)
    x = center_crop_or_pad(x, image_size)
    x = normalize_complex_pair(x)
    y = center_crop_or_pad(phase[..., None], image_size)[..., 0]
    y = phase_to_target(y)
    return x.astype(np.float32), y.astype(np.float32)


def augment(x, y):
    merged = tf.concat([x, y], axis=-1)
    if tf.random.uniform(()) > 0.5:
        merged = tf.image.flip_left_right(merged)
    if tf.random.uniform(()) > 0.5:
        merged = tf.image.flip_up_down(merged)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    merged = tf.image.rot90(merged, k)
    return merged[..., :4], merged[..., 4:]


def create_dataset(triplets, image_size, batch_size, training):
    slc1 = [p[0] for p in triplets]
    slc2 = [p[1] for p in triplets]
    target = [p[2] for p in triplets]
    ds = tf.data.Dataset.from_tensor_slices((slc1, slc2, target))

    def mapper(a, b, c):
        x, y = tf.py_function(lambda aa, bb, cc: load_sample(aa, bb, cc, image_size), [a, b, c], [tf.float32, tf.float32])
        x.set_shape([image_size, image_size, 4])
        y.set_shape([image_size, image_size, 2])
        return x, y

    ds = ds.map(mapper, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(min(len(triplets), 2048))
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=training)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_datasets(config):
    triplets = match_triplets(config.slc1_dir, config.slc2_dir, config.target_dir, config.max_files)
    train, val, test = split_triplets(triplets, config.validation_split, config.test_split, config.seed)
    train_ds = create_dataset(train, config.image_size, config.batch_size, True)
    val_ds = create_dataset(val, config.image_size, config.batch_size, False)
    test_ds = create_dataset(test, config.image_size, config.batch_size, False)
    return train_ds, val_ds, test_ds, train, val, test
