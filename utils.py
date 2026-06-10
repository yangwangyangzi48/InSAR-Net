import json
import logging
import os
import random
import time
import numpy as np
import tensorflow as tf


def get_logger(name="insarnet"):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(name)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


def make_run_dirs(output_dir):
    root = os.path.join(output_dir, "run_" + timestamp())
    paths = {
        "root": root,
        "models": os.path.join(root, "models"),
        "logs": os.path.join(root, "logs"),
        "figures": os.path.join(root, "figures"),
        "predictions": os.path.join(root, "predictions"),
        "metrics": os.path.join(root, "metrics")
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
