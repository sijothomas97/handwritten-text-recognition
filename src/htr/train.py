"""Train the CRNN with CTC loss on synthetic word images.

Usage:
    python -m htr.train --epochs 3 --train-size 2000 --val-size 200 \
        --out checkpoints/crnn.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import SyntheticWordDataset, collate
from .decode import greedy_decode
from .metrics import cer, wer
from .model import CRNN


def evaluate_model(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    refs: list[str] = []
    hyps: list[str] = []
    with torch.no_grad():
        for images, _, _, words in loader:
            log_probs = model(images.to(device))
            hyps.extend(greedy_decode(log_probs.cpu()))
            refs.extend(words)
    return cer(refs, hyps), wer(refs, hyps)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--train-size", type=int, default=2000)
    p.add_argument("--val-size", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("checkpoints/crnn.pt"))
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = SyntheticWordDataset(args.train_size, seed=args.seed)
    val_ds = SyntheticWordDataset(args.val_size, seed=args.seed + 10_000)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, collate_fn=collate,
        num_workers=args.num_workers,
    )

    model = CRNN().to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_cer = float("inf")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        for images, targets, target_lengths, _ in train_loader:
            images, targets = images.to(device), targets.to(device)
            log_probs = model(images)  # (T, B, C)
            input_lengths = torch.full(
                (images.size(0),), log_probs.size(0), dtype=torch.long
            )
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        val_cer, val_wer = evaluate_model(model, val_loader, device)
        avg_loss = total_loss / len(train_ds)
        print(
            f"epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  "
            f"val CER={val_cer:.4f}  val WER={val_wer:.4f}  "
            f"({time.time() - t0:.1f}s)"
        )
        if val_cer < best_cer:
            best_cer = val_cer
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "val_cer": val_cer,
                    "val_wer": val_wer,
                    "epoch": epoch,
                    "args": vars(args) | {"out": str(args.out)},
                },
                args.out,
            )
            print(f"  saved checkpoint -> {args.out} (CER {val_cer:.4f})")

    print(f"best val CER: {best_cer:.4f}")


if __name__ == "__main__":
    main()
