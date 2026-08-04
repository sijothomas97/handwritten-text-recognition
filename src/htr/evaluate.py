"""Evaluate a trained CRNN checkpoint on held-out synthetic words.

Usage:
    python -m htr.evaluate --checkpoint checkpoints/crnn.pt --size 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import SyntheticWordDataset, collate
from .decode import greedy_decode
from .metrics import cer, wer
from .model import CRNN


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/crnn.pt"))
    p.add_argument("--size", type=int, default=200)
    p.add_argument("--seed", type=int, default=99_000, help="held-out split seed")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--show", type=int, default=10, help="print N example predictions")
    args = p.parse_args(argv)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = CRNN()
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = SyntheticWordDataset(args.size, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate)

    refs: list[str] = []
    hyps: list[str] = []
    with torch.no_grad():
        for images, _, _, words in loader:
            hyps.extend(greedy_decode(model(images)))
            refs.extend(words)

    print(f"samples: {len(refs)}")
    print(f"CER: {cer(refs, hyps):.4f}")
    print(f"WER: {wer(refs, hyps):.4f}")
    for r, h in list(zip(refs, hyps))[: args.show]:
        mark = "ok " if r == h else "ERR"
        print(f"  [{mark}] ref={r!r:20s} hyp={h!r}")


if __name__ == "__main__":
    main()
