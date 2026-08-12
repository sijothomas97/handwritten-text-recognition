from pathlib import Path

import numpy as np
import pytest
import torch

from htr.config import Config, DataConfig, ModelConfig, TrainConfig
from htr.export_onnx import export
from htr.model import build_model

onnxruntime = pytest.importorskip("onnxruntime")

REPO = Path(__file__).resolve().parent.parent
TRAINED_CKPT = REPO / "models" / "smoke" / "best.pt"


@pytest.fixture(scope="module")
def fresh_checkpoint(tmp_path_factory):
    """A small untrained checkpoint so the test never depends on a prior run."""
    tmp = tmp_path_factory.mktemp("ckpt")
    cfg = Config(
        data=DataConfig(dataset="mnist", image_size=28),
        model=ModelConfig(arch="resnet18", image_channels=1),
        train=TrainConfig(),
    )
    torch.manual_seed(0)
    model = build_model(cfg.model, 10)
    path = tmp / "ckpt.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "config": cfg.to_dict(),
         "num_classes": 10, "epoch": 0},
        path,
    )
    return path


def _assert_parity(ckpt_path, tmp_path):
    from htr.checkpoint import load_checkpoint

    onnx_path = export(ckpt_path, tmp_path / "model.onnx")
    model, cfg, _ = load_checkpoint(ckpt_path)

    torch.manual_seed(123)
    x = torch.randn(4, cfg.model.image_channels, cfg.data.image_size, cfg.data.image_size)
    with torch.no_grad():
        torch_logits, torch_feats = model(x)

    sess = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits, onnx_feats = sess.run(None, {"image": x.numpy()})

    np.testing.assert_allclose(torch_logits.numpy(), onnx_logits, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(torch_feats.numpy(), onnx_feats, rtol=1e-3, atol=1e-4)


def test_onnx_matches_torch_fresh_model(fresh_checkpoint, tmp_path):
    _assert_parity(fresh_checkpoint, tmp_path)


@pytest.mark.skipif(not TRAINED_CKPT.exists(), reason="no trained smoke checkpoint present")
def test_onnx_matches_torch_trained_model(tmp_path):
    _assert_parity(TRAINED_CKPT, tmp_path)


def test_onnx_dynamic_batch(fresh_checkpoint, tmp_path):
    onnx_path = export(fresh_checkpoint, tmp_path / "model.onnx")
    sess = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    for batch in (1, 7):
        logits, feats = sess.run(None, {"image": np.random.randn(batch, 1, 28, 28).astype(np.float32)})
        assert logits.shape == (batch, 10)
        assert feats.shape == (batch, 512)
