import os
import tensorflow as tf
from .config import TrainConfig, apply_environment_defaults
from .data import build_datasets
from .evaluation import evaluate_dataset, save_metrics
from .model import build_insarnet, custom_objects
from .training import compile_model, train
from .utils import get_logger, make_run_dirs, save_json, set_seed


def run(config):
    logger = get_logger()
    set_seed(config.seed)
    if config.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    paths = make_run_dirs(config.output_dir)
    save_json(config.to_dict(), os.path.join(paths["root"], "config.json"))
    train_ds, val_ds, test_ds, train_files, val_files, test_files = build_datasets(config)
    logger.info("Training samples: %d", len(train_files))
    logger.info("Validation samples: %d", len(val_files))
    logger.info("Test samples: %d", len(test_files))
    if config.resume_model:
        model = tf.keras.models.load_model(config.resume_model, compile=False, custom_objects=custom_objects())
    else:
        model = build_insarnet(config.image_size, config.base_channels, config.fpn_channels, config.rdb_layers, config.rdb_growth_rate)
    model = compile_model(model, config)
    model, history = train(model, train_ds, val_ds, config, paths)
    metrics = evaluate_dataset(model, test_ds)
    save_metrics(metrics, os.path.join(paths["metrics"], "test_metrics.json"))
    logger.info("Results saved to %s", paths["root"])
    return model, metrics, paths


def default_run():
    return run(apply_environment_defaults(TrainConfig()))
