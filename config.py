"""Configuration utilities for InSAR phase filtering."""

from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass


@dataclass
class PhaseFilterConfig:
    """Runtime configuration for model training and evaluation."""

    epochs: int = 10
    batch_size: int = 4
    image_size: int = 256
    max_files: int = 100
    validation_split: float = 0.2
    test_split: float = 0.1
    learning_rate: float = 5e-4
    min_learning_rate: float = 5e-7
    warmup_ratio: float = 0.06
    include_coherence: bool = True
    include_frequency: bool = True
    coherence_window_size: int = 5
    early_stopping_patience: int = 20
    random_seed: int = 42
    tensorboard_log_dir: str = "tf_logs"

    @property
    def input_channels(self) -> int:
        """Input channels: sin, cos, gradient, optional coherence and spectrum."""
        channels = 3
        if self.include_coherence:
            channels += 1
        if self.include_frequency:
            channels += 1
        return channels

    @property
    def output_channels(self) -> int:
        """Output channels: sin and cos."""
        return 2

    @classmethod
    def from_environment(cls) -> "PhaseFilterConfig":
        """Return recommended defaults for AutoDL/cloud or local debugging."""
        if os.path.exists("/root/autodl-tmp"):
            return cls(
                epochs=64,
                batch_size=32,
                image_size=256,
                max_files=17000,
                validation_split=0.2,
                test_split=0.1,
                learning_rate=1e-5,
                min_learning_rate=5e-7,
                include_coherence=True,
                include_frequency=True,
                tensorboard_log_dir="/root/tf-logs",
            )
        return cls()


@dataclass
class RunPaths:
    """Input and output directories for a single run."""

    noise_dir: str
    clean_dir: str
    output_dir: str
    model_dir: str
    prediction_dir: str
    evaluation_dir: str


def default_base_dir() -> str:
    """Default dataset root used by the original script."""
    if os.path.exists("/root/autodl-tmp"):
        return "/root/autodl-tmp/PSPF11pro/simulation_results"
    return r"D:\YWWY\Projects\PSim\PS11\simulation_results"


def setup_run_paths(
    base_dir: str | None = None,
    noise_dir: str | None = None,
    clean_dir: str | None = None,
    output_root: str | None = None,
) -> RunPaths:
    """Create and return input/output paths.

    Parameters
    ----------
    base_dir:
        Dataset root. If ``noise_dir`` or ``clean_dir`` is not specified, this
        folder is expected to contain ``noisy_phase_shuffle`` and
        ``clean_phase_shuffle``.
    noise_dir, clean_dir:
        Optional explicit directories for noisy and clean wrapped phase tifs.
    output_root:
        Root folder for run outputs. Defaults to ``./results``.
    """
    base_dir = base_dir or default_base_dir()
    noise_dir = noise_dir or os.path.join(base_dir, "noisy_phase_shuffle")
    clean_dir = clean_dir or os.path.join(base_dir, "clean_phase_shuffle")

    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_root = output_root or os.path.join(os.getcwd(), "results")
    output_dir = os.path.join(output_root, f"results_{timestamp}")
    model_dir = os.path.join(output_dir, "model")
    prediction_dir = os.path.join(output_dir, "predictions")
    evaluation_dir = os.path.join(output_dir, "evaluation")

    for path in (output_root, output_dir, model_dir, prediction_dir, evaluation_dir):
        os.makedirs(path, exist_ok=True)

    return RunPaths(
        noise_dir=noise_dir,
        clean_dir=clean_dir,
        output_dir=output_dir,
        model_dir=model_dir,
        prediction_dir=prediction_dir,
        evaluation_dir=evaluation_dir,
    )
