import pytest
import torch

from htr.config import ModelConfig
from htr.model import build_model


@pytest.mark.parametrize("arch,feat_dim", [("resnet18", 512), ("resnet34", 512), ("resnet50", 2048)])
def test_forward_shapes(arch, feat_dim):
    model = build_model(ModelConfig(arch=arch, image_channels=1), num_classes=10)
    model.eval()
    x = torch.randn(3, 1, 28, 28)
    logits, features = model(x)
    assert logits.shape == (3, 10)
    assert features.shape == (3, feat_dim)


def test_forward_rgb_channels():
    model = build_model(ModelConfig(arch="resnet18", image_channels=3), num_classes=47)
    model.eval()
    logits, _ = model(torch.randn(2, 3, 32, 32))
    assert logits.shape == (2, 47)


def test_unknown_arch_raises():
    with pytest.raises(ValueError):
        build_model(ModelConfig(arch="resnet9000"), num_classes=10)


def test_forward_batch_size_one():
    model = build_model(ModelConfig(arch="resnet18", image_channels=1), num_classes=10)
    model.eval()
    logits, _ = model(torch.randn(1, 1, 28, 28))
    assert logits.shape == (1, 10)
