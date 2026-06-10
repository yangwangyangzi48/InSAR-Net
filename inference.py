import numpy as np
import tensorflow as tf
from .io import read_complex_tif, center_crop_or_pad, write_tif
from .features import slc_pair_to_tensor, normalize_complex_pair, target_to_phase
from .model import custom_objects


def load_model(path):
    return tf.keras.models.load_model(path, compile=False, custom_objects=custom_objects())


def prepare_input(slc1_path, slc2_path, image_size):
    z1r, z1i = read_complex_tif(slc1_path)
    z2r, z2i = read_complex_tif(slc2_path)
    x = slc_pair_to_tensor(z1r, z1i, z2r, z2i)
    x = center_crop_or_pad(x, image_size)
    x = normalize_complex_pair(x)
    return x[None, ...]


def predict_phase(model, slc1_path, slc2_path, image_size):
    x = prepare_input(slc1_path, slc2_path, image_size)
    pred = model.predict(x, verbose=0)[0]
    return target_to_phase(pred)


def predict_to_tif(model_path, slc1_path, slc2_path, output_path, image_size):
    model = load_model(model_path)
    phase = predict_phase(model, slc1_path, slc2_path, image_size)
    write_tif(output_path, phase.astype(np.float32))
    return output_path
