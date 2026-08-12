"""Training CLI.

Usage:
    python -m htr.train --config configs/smoke.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .config import load_config
from .data import get_dataloaders, num_classes
from .engine import evaluate_model, train_one_epoch
from .model import build_model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a from-scratch ResNet classifier")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    torch.manual_seed(cfg.train.seed)
    device = torch.device(cfg.train.device)

    train_loader, test_loader = get_dataloaders(cfg.data, seed=cfg.train.seed)
    n_classes = num_classes(cfg.data)
    model = build_model(cfg.model, n_classes).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    out_dir = Path(cfg.train.out_dir) / cfg.train.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_acc = -1.0
    start = time.time()
    for epoch in range(1, cfg.train.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        test_metrics = evaluate_model(model, test_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
        }
        history.append(record)
        print(json.dumps(record))

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": cfg.to_dict(),
            "num_classes": n_classes,
        }
        torch.save(checkpoint, out_dir / "last.pt")
        if test_metrics["accuracy"] > best_acc:
            best_acc = test_metrics["accuracy"]
            torch.save(checkpoint, out_dir / "best.pt")

    metrics = {
        "config": cfg.to_dict(),
        "num_classes": n_classes,
        "epochs_run": cfg.train.epochs,
        "best_test_accuracy": best_acc,
        "final": history[-1] if history else None,
        "history": history,
        "wall_time_seconds": round(time.time() - start, 2),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved checkpoints and metrics.json to {out_dir}")


if __name__ == "__main__":
    main()
