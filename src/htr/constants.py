"""Shared image-size constants, kept dependency-free (no torch/PIL import)
so the lean inference-serving path (`htr.api`, `htr.preprocess`) doesn't
need to pull in the training-only dependencies that `htr.data` requires.
"""

from __future__ import annotations

IMG_H = 32
IMG_W = 128
MAX_LABEL_LEN = 10
