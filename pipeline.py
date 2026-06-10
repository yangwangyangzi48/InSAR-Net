"""End-to-end training pipeline."""

from __future__ import annotations

import json
import os

from sklearn.model_selection import train_test_split

from .config import PhaseFilterConfig, RunPaths
from .dataset import create_dataset
from .evaluation import evaluate_model
from .io import load_data_pairs
from .models import build_improved_phase_filter_model
from .training import compile_model, plot_training_history, train_model
from .utils import logger, set_global_seed
from .visualization import visualize_results


def run_training_pipeline(
    config: PhaseFilterConfig,
    paths: RunPaths,
    resume_model: str | None = None,
    visualize: bool = True,
):
    """Run data loading, training, evaluation, and visualization."""
    set_global_seed(config.random_seed)

    noise_files, clean_files = load_data_pairs(paths.noise_dir, paths.clean_dir, max_files=config.max_files)

    train_noise, test_noise, train_clean, test_clean = train_test_split(
        noise_files,
        clean_files,
        test_size=config.test_split,
        random_state=config.random_seed,
    )

    val_ratio = config.validation_split / max(1e-8, 1.0 - config.test_split)
    train_noise, val_noise, train_clean, val_clean = train_test_split(
        train_noise,
        train_clean,
        test_size=val_ratio,
        random_state=config.random_seed,
    )

    logger.info(
        "Data split: %d training, %d validation, %d test samples",
        len(train_noise),
        len(val_noise),
        len(test_noise),
    )

    train_dataset = create_dataset(train_noise, train_clean, config, is_training=True)
    val_dataset = create_dataset(val_noise, val_clean, config, is_training=False)
    test_dataset = create_dataset(test_noise, test_clean, config, is_training=False)

    model = build_improved_phase_filter_model(
        image_size=config.image_size,
        input_channels=config.input_channels,
    )
    model = compile_model(model, config)
    model.summary()

    model, history = train_model(
        model,
        train_dataset,
        val_dataset,
        config=config,
        model_dir=paths.model_dir,
        resume_model=resume_model,
    )

    plot_training_history(history, paths.evaluation_dir, config)
    metrics = evaluate_model(model, test_dataset)

    with open(os.path.join(paths.evaluation_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    if visualize:
        visualize_results(model, test_noise, test_clean, paths.evaluation_dir, config, num_samples=5)

    logger.info("InSAR phase filtering results saved to %s", paths.output_dir)
    return model, metrics
