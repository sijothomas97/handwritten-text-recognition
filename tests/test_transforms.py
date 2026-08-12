import torch
from PIL import Image

from htr.config import DataConfig
from htr.data import build_transform, num_classes


def _apply(cfg, img):
    return build_transform(cfg)(img)


def test_transform_output_shape_and_dtype():
    cfg = DataConfig(dataset="mnist", image_size=28)
    img = Image.new("L", (28, 28), color=0)
    out = _apply(cfg, img)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 28, 28)
    assert out.dtype == torch.float32


def test_transform_resizes_arbitrary_input():
    cfg = DataConfig(dataset="mnist", image_size=28)
    img = Image.new("RGB", (100, 60), color=(255, 255, 255))
    out = _apply(cfg, img)
    assert out.shape == (1, 28, 28)  # RGB collapsed to 1 channel, resized


def test_transform_normalisation_values():
    cfg = DataConfig(dataset="mnist", image_size=28)
    black = _apply(cfg, Image.new("L", (28, 28), color=0))
    white = _apply(cfg, Image.new("L", (28, 28), color=255))
    # (0 - 0.1307) / 0.3081 and (1 - 0.1307) / 0.3081
    assert torch.allclose(black, torch.full_like(black, -0.1307 / 0.3081), atol=1e-4)
    assert torch.allclose(white, torch.full_like(white, (1 - 0.1307) / 0.3081), atol=1e-4)


def test_transform_is_deterministic():
    cfg = DataConfig(dataset="mnist", image_size=28)
    img = Image.new("L", (40, 40), color=128)
    assert torch.equal(_apply(cfg, img), _apply(cfg, img))


def test_num_classes():
    assert num_classes(DataConfig(dataset="mnist")) == 10
    assert num_classes(DataConfig(dataset="emnist", emnist_split="balanced")) == 47
    assert num_classes(DataConfig(dataset="emnist", emnist_split="byclass")) == 62
