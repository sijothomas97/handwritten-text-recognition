"""CER / WER metrics based on Levenshtein edit distance."""

from __future__ import annotations

from collections.abc import Sequence


def levenshtein(a: Sequence, b: Sequence) -> int:
    """Edit distance between two sequences (insert/delete/substitute = 1)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(refs: list[str], hyps: list[str]) -> float:
    """Corpus character error rate: total char edits / total ref chars."""
    if len(refs) != len(hyps):
        raise ValueError("refs and hyps must have the same length")
    edits = sum(levenshtein(r, h) for r, h in zip(refs, hyps))
    total = sum(len(r) for r in refs)
    return edits / max(total, 1)


def wer(refs: list[str], hyps: list[str]) -> float:
    """Corpus word error rate: total word edits / total ref words.

    Each ref/hyp string is split on whitespace; for single-word samples
    this reduces to 1 - word accuracy.
    """
    if len(refs) != len(hyps):
        raise ValueError("refs and hyps must have the same length")
    edits = sum(levenshtein(r.split(), h.split()) for r, h in zip(refs, hyps))
    total = sum(len(r.split()) for r in refs)
    return edits / max(total, 1)
