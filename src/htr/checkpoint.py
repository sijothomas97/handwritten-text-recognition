"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from .config import Config, DataConfig, ModelConfig, TrainConfig
from .model import build_model


def load_checkpoint(path: str | Path):
    """Load a training checkpoint; returns (model, config, raw_checkpoint_dict)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    raw = ckpt["config"]
    cfg = Config(
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**raw["train"]),
    )
    model = build_model(cfg.model, ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, ckpt
