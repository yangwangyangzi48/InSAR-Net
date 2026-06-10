import os
from dataclasses import dataclass, asdict


@dataclass
class TrainConfig:
    slc1_dir: str = "data/slc1"
    slc2_dir: str = "data/slc2"
    target_dir: str = "data/clean_phase"
    output_dir: str = "runs"
    image_size: int = 256
    batch_size: int = 8
    epochs: int = 80
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    validation_split: float = 0.2
    test_split: float = 0.1
    max_files: int = 0
    seed: int = 42
    base_channels: int = 64
    fpn_channels: int = 128
    rdb_layers: int = 4
    rdb_growth_rate: int = 32
    lambda_cmp: float = 0.4
    lambda_ang: float = 0.1
    lambda_grad: float = 0.3
    lambda_cons: float = 0.2
    mixed_precision: bool = False
    resume_model: str = ""

    def to_dict(self):
        return asdict(self)


def apply_environment_defaults(config):
    if os.path.exists("/root/autodl-tmp"):
        config.batch_size = max(config.batch_size, 16)
        config.epochs = max(config.epochs, 80)
    return config
