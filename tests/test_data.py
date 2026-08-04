import torch

from htr.charset import CHARS, decode_labels, encode
from htr.data import IMG_H, IMG_W, SyntheticWordDataset, collate


def test_charset_roundtrip():
    for word in ["hello", "zebra", "a", CHARS]:
        assert decode_labels(encode(word)) == word


def test_encode_rejects_unknown_chars():
    import pytest

    with pytest.raises(ValueError):
        encode("Hello!")


def test_dataset_shapes_and_ranges():
    ds = SyntheticWordDataset(8, seed=1)
    x, y, word = ds[0]
    assert x.shape == (1, IMG_H, IMG_W)
    assert x.dtype == torch.float32
    assert 0.0 <= x.min() and x.max() <= 1.0
    assert x.max() > 0.2  # actually contains ink
    assert y.tolist() == encode(word)
    assert 1 <= len(word) <= 10


def test_dataset_deterministic_per_index():
    a = SyntheticWordDataset(4, seed=7)
    b = SyntheticWordDataset(4, seed=7)
    xa, ya, wa = a[2]
    xb, yb, wb = b[2]
    assert wa == wb
    assert torch.equal(xa, xb)
    assert torch.equal(ya, yb)


def test_different_seeds_differ():
    a = SyntheticWordDataset(20, seed=1)
    b = SyntheticWordDataset(20, seed=2)
    words_a = [a[i][2] for i in range(20)]
    words_b = [b[i][2] for i in range(20)]
    assert words_a != words_b


def test_collate():
    ds = SyntheticWordDataset(4, seed=3)
    images, targets, target_lengths, words = collate([ds[i] for i in range(4)])
    assert images.shape == (4, 1, IMG_H, IMG_W)
    assert target_lengths.sum().item() == targets.numel()
    assert target_lengths.tolist() == [len(w) for w in words]
