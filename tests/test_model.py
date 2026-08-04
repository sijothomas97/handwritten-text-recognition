import torch

from htr.charset import NUM_CLASSES
from htr.data import IMG_H, IMG_W
from htr.model import CRNN


def test_forward_shape():
    model = CRNN()
    x = torch.randn(5, 1, IMG_H, IMG_W)
    out = model(x)
    assert out.shape == (model.time_steps, 5, NUM_CLASSES)
    assert out.shape[0] == IMG_W // 4


def test_output_is_log_probs():
    model = CRNN().eval()
    with torch.no_grad():
        out = model(torch.randn(2, 1, IMG_H, IMG_W))
    sums = out.exp().sum(dim=2)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_time_steps_cover_max_label():
    # CTC needs input length >= label length (>= 2L-1 with repeats).
    from htr.data import MAX_LABEL_LEN

    assert CRNN().time_steps >= 2 * MAX_LABEL_LEN - 1
