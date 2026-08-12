"""Config loading for training / evaluation runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    dataset: str = "mnist"          # "mnist" or "emnist"
    emnist_split: str = "balanced"  # only used when dataset == "emnist"
    root: str = "data"
    image_size: int = 28
    batch_size: int = 128
    num_workers: int = 0
    train_subset: int | None = None  # cap number of training samples (smoke runs)
    test_subset: int | None = None


@dataclass
class ModelConfig:
    arch: str = "resnet18"  # resnet18 / resnet34 / resnet50 / resnet101 / resnet152
    image_channels: int = 1


@dataclass
class TrainConfig:
    epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 42
    device: str = "cpu"
    out_dir: str = "models"
    run_name: str = "run"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> Config:
    """Load a YAML config file into a Config, applying defaults for missing keys."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return Config(
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**raw.get("train", {})),
    )
