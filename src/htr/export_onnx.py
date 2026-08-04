"""Export a trained CRNN checkpoint to ONNX and verify parity.

Usage:
    python -m htr.export_onnx --checkpoint checkpoints/crnn.pt \
        --out checkpoints/crnn.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .data import IMG_H, IMG_W
from .model import CRNN


def export(checkpoint: Path, out: Path) -> None:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = CRNN()
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy = torch.randn(1, 1, IMG_H, IMG_W)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy,),
        str(out),
        input_names=["image"],
        output_names=["log_probs"],
        dynamic_axes={"image": {0: "batch"}, "log_probs": {1: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    # Parity check with onnxruntime.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = torch.randn(3, 1, IMG_H, IMG_W)
    with torch.no_grad():
        ref = model(x).numpy()
    got = sess.run(None, {"image": x.numpy()})[0]
    max_diff = float(np.abs(ref - got).max())
    if max_diff > 1e-3:
        raise RuntimeError(f"ONNX/PyTorch mismatch: max diff {max_diff}")
    print(f"exported {out} (parity max diff {max_diff:.2e})")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/crnn.pt"))
    p.add_argument("--out", type=Path, default=Path("checkpoints/crnn.onnx"))
    args = p.parse_args(argv)
    export(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
