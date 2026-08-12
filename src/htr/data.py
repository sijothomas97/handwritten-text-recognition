"""Config-driven dataset loading (torchvision MNIST / EMNIST, auto-download)."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import DataConfig

# Per-dataset normalisation stats (single grayscale channel).
_STATS = {
    "mnist": ((0.1307,), (0.3081,)),
    "emnist": ((0.1751,), (0.3332,)),
}


def build_transform(cfg: DataConfig, train: bool = False) -> transforms.Compose:
    """Deterministic preprocessing: grayscale -> resize -> tensor -> normalise."""
    mean, std = _STATS.get(cfg.dataset, _STATS["mnist"])
    ops = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    return transforms.Compose(ops)


def _dataset(cfg: DataConfig, train: bool):
    tfm = build_transform(cfg, train=train)
    if cfg.dataset == "mnist":
        return datasets.MNIST(cfg.root, train=train, download=True, transform=tfm)
    if cfg.dataset == "emnist":
        return datasets.EMNIST(
            cfg.root, split=cfg.emnist_split, train=train, download=True, transform=tfm
        )
    raise ValueError(f"Unknown dataset: {cfg.dataset!r} (expected 'mnist' or 'emnist')")


def num_classes(cfg: DataConfig) -> int:
    if cfg.dataset == "mnist":
        return 10
    if cfg.dataset == "emnist":
        return {"byclass": 62, "bymerge": 47, "balanced": 47, "letters": 26,
                "digits": 10, "mnist": 10}[cfg.emnist_split]
    raise ValueError(f"Unknown dataset: {cfg.dataset!r}")


def _maybe_subset(ds, n: int | None, seed: int = 0):
    if n is None or n >= len(ds):
        return ds
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    return Subset(ds, idx)


def get_dataloaders(cfg: DataConfig, seed: int = 0) -> tuple[DataLoader, DataLoader]:
    train_ds = _maybe_subset(_dataset(cfg, train=True), cfg.train_subset, seed)
    test_ds = _maybe_subset(_dataset(cfg, train=False), cfg.test_subset, seed)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    return train_loader, test_loader
