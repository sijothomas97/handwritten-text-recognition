"""Character set and label encoding for CTC.

Index 0 is reserved for the CTC blank token.
"""

from __future__ import annotations

CHARS = "abcdefghijklmnopqrstuvwxyz"
BLANK_IDX = 0
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # 1..26
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  # + blank


def encode(text: str) -> list[int]:
    """Map a string to a list of label indices (no blank)."""
    try:
        return [CHAR_TO_IDX[c] for c in text]
    except KeyError as e:
        raise ValueError(f"Character {e.args[0]!r} not in charset") from None


def decode_labels(indices: list[int]) -> str:
    """Map label indices (no blanks/repeats handling) back to a string."""
    return "".join(IDX_TO_CHAR[i] for i in indices if i != BLANK_IDX)
