"""Evaluation CLI: load a checkpoint and report test-set metrics.

Usage:
    python -m htr.evaluate --checkpoint models/smoke/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .checkpoint import load_checkpoint
from .data import get_dataloaders
from .engine import evaluate_model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--out", default=None, help="Optional path to write metrics JSON")
    args = parser.parse_args(argv)

    model, cfg, _ = load_checkpoint(args.checkpoint)
    device = torch.device(cfg.train.device)
    model.to(device)

    _, test_loader = get_dataloaders(cfg.data, seed=cfg.train.seed)
    metrics = evaluate_model(model, test_loader, device)
    result = {"checkpoint": str(args.checkpoint), **metrics}
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
