"""CTC decoding (greedy / best-path)."""

from __future__ import annotations

import torch

from .charset import BLANK_IDX, IDX_TO_CHAR


def greedy_decode(log_probs: torch.Tensor) -> list[str]:
    """Best-path CTC decode.

    Args:
        log_probs: (T, B, C) log-probabilities (model output).
    Returns:
        One decoded string per batch element (repeats collapsed, blanks removed).
    """
    best = log_probs.argmax(dim=2).T  # (B, T)
    results = []
    for seq in best.tolist():
        chars = []
        prev = BLANK_IDX
        for idx in seq:
            if idx != BLANK_IDX and idx != prev:
                chars.append(IDX_TO_CHAR[idx])
            prev = idx
        results.append("".join(chars))
    return results
