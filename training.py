import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from .losses import phase_consistency_loss
from .metrics import phase_mae, phase_rmse


def build_optimizer(config):
    return tf.keras.optimizers.Adam(learning_rate=config.learning_rate)


def compile_model(model, config):
    model.compile(optimizer=build_optimizer(config), loss=phase_consistency_loss(config.lambda_cmp, config.lambda_ang, config.lambda_grad, config.lambda_cons), metrics=[phase_mae, phase_rmse])
    return model


def cosine_schedule(total_epochs, warmup_epochs, initial_lr, min_lr):
    def fn(epoch):
        if epoch < warmup_epochs:
            return initial_lr * float(epoch + 1) / float(max(warmup_epochs, 1))
        progress = float(epoch - warmup_epochs) / float(max(total_epochs - warmup_epochs, 1))
        return min_lr + 0.5 * (initial_lr - min_lr) * (1.0 + np.cos(np.pi * progress))
    return fn


def callbacks(config, paths):
    lr_fn = cosine_schedule(config.epochs, max(int(config.epochs * 0.06), 1), config.learning_rate, config.min_learning_rate)
    return [
        tf.keras.callbacks.LearningRateScheduler(lr_fn),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(paths["models"], "best_model.h5"), monitor="val_loss", save_best_only=True, mode="min"),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(os.path.join(paths["logs"], "history.csv")),
        tf.keras.callbacks.TensorBoard(log_dir=paths["logs"])
    ], lr_fn


def train(model, train_ds, val_ds, config, paths):
    cbs, lr_fn = callbacks(config, paths)
    history = model.fit(train_ds, validation_data=val_ds, epochs=config.epochs, callbacks=cbs)
    model.save(os.path.join(paths["models"], "final_model.h5"))
    plot_history(history, lr_fn, config.epochs, paths["figures"])
    return model, history


def plot_history(history, lr_fn, epochs, out_dir):
    keys = list(history.history.keys())
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(history.history.get("loss", []), label="loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.legend()
    plt.subplot(1, 3, 2)
    if "phase_rmse" in keys:
        plt.plot(history.history.get("phase_rmse", []), label="phase_rmse")
        plt.plot(history.history.get("val_phase_rmse", []), label="val_phase_rmse")
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot([lr_fn(i) for i in range(epochs)], label="lr")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_history.png"), dpi=200)
    plt.close()
