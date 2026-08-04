import torch

from htr.charset import BLANK_IDX, CHAR_TO_IDX, NUM_CLASSES
from htr.decode import greedy_decode


def _log_probs_from_path(path: list[int], num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Build (T, 1, C) log-probs whose argmax follows `path`."""
    t = len(path)
    lp = torch.full((t, 1, num_classes), -10.0)
    for i, idx in enumerate(path):
        lp[i, 0, idx] = 0.0
    return lp


def test_collapses_repeats():
    c = CHAR_TO_IDX
    path = [c["h"], c["h"], c["e"], c["l"], c["l"], c["o"]]
    assert greedy_decode(_log_probs_from_path(path)) == ["helo"]


def test_blank_separates_repeats():
    c = CHAR_TO_IDX
    path = [c["l"], BLANK_IDX, c["l"], c["o"]]
    assert greedy_decode(_log_probs_from_path(path)) == ["llo"]


def test_leading_trailing_blanks():
    c = CHAR_TO_IDX
    path = [BLANK_IDX, BLANK_IDX, c["a"], BLANK_IDX]
    assert greedy_decode(_log_probs_from_path(path)) == ["a"]


def test_all_blanks_gives_empty():
    path = [BLANK_IDX] * 5
    assert greedy_decode(_log_probs_from_path(path)) == [""]


def test_batch_decode():
    c = CHAR_TO_IDX
    a = _log_probs_from_path([c["a"], c["b"], BLANK_IDX])
    b = _log_probs_from_path([BLANK_IDX, c["z"], c["z"]])
    lp = torch.cat([a, b], dim=1)  # (T, 2, C)
    assert greedy_decode(lp) == ["ab", "z"]
