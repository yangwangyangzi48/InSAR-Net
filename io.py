import glob
import os
import numpy as np
import tifffile as tiff


def list_tifs(directory):
    files = []
    for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        files.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(files)


def basename_key(path):
    name = os.path.basename(path)
    return os.path.splitext(name)[0]


def match_triplets(slc1_dir, slc2_dir, target_dir, max_files=0):
    slc1 = {basename_key(p): p for p in list_tifs(slc1_dir)}
    slc2 = {basename_key(p): p for p in list_tifs(slc2_dir)}
    target = {basename_key(p): p for p in list_tifs(target_dir)}
    keys = sorted(set(slc1).intersection(slc2).intersection(target))
    if max_files and max_files > 0:
        keys = keys[:max_files]
    return [(slc1[k], slc2[k], target[k]) for k in keys]


def read_tif(path):
    arr = tiff.imread(path).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def write_tif(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tiff.imwrite(path, arr.astype(np.float32))


def read_complex_tif(path):
    arr = read_tif(path)
    if arr.ndim == 2:
        real = arr
        imag = np.zeros_like(arr, dtype=np.float32)
    elif arr.ndim == 3 and arr.shape[-1] >= 2:
        real = arr[..., 0]
        imag = arr[..., 1]
    elif arr.ndim == 3 and arr.shape[0] >= 2:
        real = arr[0]
        imag = arr[1]
    else:
        arr = np.squeeze(arr)
        real = arr.astype(np.float32)
        imag = np.zeros_like(real, dtype=np.float32)
    return real.astype(np.float32), imag.astype(np.float32)


def read_phase_tif(path):
    arr = read_tif(path)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
        if arr.ndim > 2:
            arr = arr[..., 0]
    return arr.astype(np.float32)


def center_crop_or_pad(arr, size):
    h, w = arr.shape[:2]
    if h >= size and w >= size:
        r0 = (h - size) // 2
        c0 = (w - size) // 2
        return arr[r0:r0 + size, c0:c0 + size, ...]
    out_shape = (size, size) + arr.shape[2:]
    out = np.zeros(out_shape, dtype=arr.dtype)
    r0 = max((size - h) // 2, 0)
    c0 = max((size - w) // 2, 0)
    rr = min(h, size)
    cc = min(w, size)
    out[r0:r0 + rr, c0:c0 + cc, ...] = arr[:rr, :cc, ...]
    return out
