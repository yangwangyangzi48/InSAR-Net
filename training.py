"""Training routines."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .config import PhaseFilterConfig
from .losses import improved_phase_loss2
from .utils import logger


def compile_model(model: tf.keras.Model, config: PhaseFilterConfig) -> tf.keras.Model:
    """Compile model with Adam and phase-aware loss."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(optimizer=optimizer, loss=improved_phase_loss2(), metrics=["mae"])
    logger.info("Model compiled with improved phase loss.")
    return model


def cosine_decay_with_warmup(total_epochs: int, warmup_epochs: int, initial_lr: float, min_lr: float):
    """Cosine decay learning-rate scheduler with linear warmup."""
    warmup_epochs = max(1, int(warmup_epochs))
    total_epochs = max(warmup_epochs + 1, int(total_epochs))

    def lr_fn(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(initial_lr * (epoch + 1) / warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return float((initial_lr - min_lr) * cosine_decay + min_lr)

    return lr_fn


def train_model(
    model: tf.keras.Model,
    train_dataset,
    val_dataset,
    config: PhaseFilterConfig,
    model_dir: str,
    initial_epoch: int = 0,
    resume_model: str | None = None,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Train or resume the model."""
    if resume_model and os.path.exists(resume_model):
        logger.info("Loading existing model from %s", resume_model)
        model = tf.keras.models.load_model(resume_model, compile=False)
        model = compile_model(model, config)
    elif resume_model:
        logger.warning("Resume model not found: %s. Training from scratch.", resume_model)

    warmup_epochs = int(config.warmup_ratio * config.epochs)
    lr_schedule_fn = cosine_decay_with_warmup(
        config.epochs, warmup_epochs, config.learning_rate, config.min_learning_rate
    )

    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(lr_schedule_fn),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "best_model.h5"),
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=config.tensorboard_log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq="epoch",
        ),
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(os.path.join(model_dir, "final_model.h5"))
    return model, history


def plot_training_history(history, evaluation_dir: str, config: PhaseFilterConfig) -> None:
    """Save loss, MAE, and learning-rate curves."""
    os.makedirs(evaluation_dir, exist_ok=True)
    warmup_epochs = int(config.warmup_ratio * config.epochs)
    lr_schedule_fn = cosine_decay_with_warmup(
        config.epochs, warmup_epochs, config.learning_rate, config.min_learning_rate
    )

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history.history.get("loss", []), label="Training Loss")
    plt.plot(history.history.get("val_loss", []), label="Validation Loss")
    plt.title("Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history.history.get("mae", []), label="Training MAE")
    plt.plot(history.history.get("val_mae", []), label="Validation MAE")
    plt.title("MAE History")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()

    lr_schedule = [lr_schedule_fn(epoch) for epoch in range(config.epochs)]
    plt.subplot(1, 3, 3)
    plt.plot(lr_schedule, label="Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.yscale("log")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(evaluation_dir, "training_history.png"), dpi=200)
    plt.close()
