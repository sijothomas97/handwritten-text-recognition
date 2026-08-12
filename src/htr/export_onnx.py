"""Export a trained checkpoint to ONNX.

Usage:
    python -m htr.export_onnx --checkpoint models/smoke/best.pt --out models/smoke/model.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .checkpoint import load_checkpoint


def export(checkpoint_path: str | Path, out_path: str | Path) -> Path:
    model, cfg, _ = load_checkpoint(checkpoint_path)
    dummy = torch.randn(
        1, cfg.model.image_channels, cfg.data.image_size, cfg.data.image_size
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy,),
        str(out_path),
        input_names=["image"],
        output_names=["logits", "features"],
        dynamic_axes={
            "image": {0: "batch"},
            "logits": {0: "batch"},
            "features": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export checkpoint to ONNX")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    path = export(args.checkpoint, args.out)
    print(f"Exported ONNX model to {path}")


if __name__ == "__main__":
    main()
