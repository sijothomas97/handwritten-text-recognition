"""Model factory wiring the existing from-scratch ResNet architecture."""

from __future__ import annotations

from algorithms import (
    ResNet,
    resnet18_config,
    resnet34_config,
    resnet50_config,
    resnet101_config,
    resnet152_config,
)

from .config import ModelConfig

_ARCHS = {
    "resnet18": resnet18_config,
    "resnet34": resnet34_config,
    "resnet50": resnet50_config,
    "resnet101": resnet101_config,
    "resnet152": resnet152_config,
}


def build_model(cfg: ModelConfig, num_classes: int) -> ResNet:
    if cfg.arch not in _ARCHS:
        raise ValueError(f"Unknown arch {cfg.arch!r}; choose one of {sorted(_ARCHS)}")
    return ResNet(_ARCHS[cfg.arch], num_classes, image_channels=cfg.image_channels)
