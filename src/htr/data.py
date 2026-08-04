"""Synthetic word-image dataset rendered on the fly with PIL.

Zero-download OCR bootstrapping: random words are rendered in a mix of
system fonts (handwriting-style ones preferred) with light geometric and
noise augmentation. Each sample index maps deterministically to a word +
rendering, so train/val splits (different seed bases) are reproducible.

An optional IAM hook (`IAMWordsDataset`) is provided for later use with
the real IAM words dataset; it is not required for training.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torch.utils.data import Dataset

from .charset import CHARS, encode
from .constants import IMG_H, IMG_W, MAX_LABEL_LEN

__all__ = [
    "IMG_H",
    "IMG_W",
    "MAX_LABEL_LEN",
    "WORDS",
    "available_fonts",
    "render_word",
    "SyntheticWordDataset",
    "collate",
    "IAMWordsDataset",
]

# A small embedded vocabulary of common English words (lowercase a-z only).
WORDS = (
    "the of and to in is was for that with his they this have from one had "
    "word but not what all were when your can said there use each which she "
    "how their will other about out many then them these some her would make "
    "like him into time has look two more write see number way could people "
    "than first water been call who oil its now find long down day did get "
    "come made may part over new sound take only little work know place year "
    "live back give most very after thing our just name good sentence man "
    "think say great where help through much before line right too mean old "
    "any same tell boy follow came want show also around form three small "
    "set put end does another well large must big even such because turn "
    "here why ask went men read need land different home move try kind hand "
    "picture again change off play spell air away animal house point page "
    "letter mother answer found study still learn should world high every "
    "near add food between own below country plant last school father keep "
    "tree never start city earth eye light thought head under story saw left "
    "night bright quick jump lazy vex fox dog zebra"
).split()

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
    "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
    "/System/Library/Fonts/Supplemental/Chalkboard.ttc",
    "/System/Library/Fonts/Supplemental/Apple Chancery.ttf",
    "/System/Library/Fonts/Supplemental/Noteworthy.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
    # Linux (CI) fallbacks
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def available_fonts() -> list[str]:
    fonts = [f for f in _FONT_CANDIDATES if os.path.exists(f)]
    if not fonts:
        raise RuntimeError("No usable TTF/TTC fonts found on this system")
    return fonts


def render_word(word: str, rng: random.Random, fonts: list[str]) -> np.ndarray:
    """Render one word to a float32 array of shape (IMG_H, IMG_W) in [0, 1].

    Dark ink on light background, normalized so ink ~= 1.0 after inversion.
    """
    font_path = rng.choice(fonts)
    font_size = rng.randint(20, 28)
    font = ImageFont.truetype(font_path, font_size)

    # Render on a generous canvas, then crop to ink and resize.
    canvas = Image.new("L", (IMG_W * 4, IMG_H * 4), color=255)
    draw = ImageDraw.Draw(canvas)
    ink = rng.randint(0, 60)
    draw.text((20, 20), word, fill=ink, font=font)

    bbox = canvas.point(lambda p: 255 - p).getbbox()
    if bbox is None:  # blank render (shouldn't happen) — return empty image
        return np.zeros((IMG_H, IMG_W), dtype=np.float32)
    canvas = canvas.crop(bbox)

    # Mild rotation for augmentation.
    angle = rng.uniform(-3.0, 3.0)
    canvas = canvas.rotate(angle, expand=True, fillcolor=255, resample=Image.BILINEAR)

    # Fit into IMG_H x IMG_W keeping aspect ratio, left-aligned with padding.
    w, h = canvas.size
    scale = min((IMG_H - 4) / h, (IMG_W - 4) / w)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    canvas = canvas.resize((new_w, new_h), Image.BILINEAR)

    out = Image.new("L", (IMG_W, IMG_H), color=255)
    x0 = rng.randint(1, max(1, IMG_W - new_w - 1))
    y0 = (IMG_H - new_h) // 2
    out.paste(canvas, (x0, y0))

    if rng.random() < 0.3:
        out = out.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.8)))

    arr = np.asarray(out, dtype=np.float32) / 255.0
    arr = 1.0 - arr  # ink -> high values
    # Additive noise
    noise = np.random.default_rng(rng.getrandbits(32)).normal(0, 0.03, arr.shape)
    arr = np.clip(arr + noise, 0.0, 1.0).astype(np.float32)
    return arr


class SyntheticWordDataset(Dataset):
    """Deterministic on-the-fly synthetic word images.

    Sample i is fully determined by (seed, i): same word, font, and
    augmentation every epoch/run. Use different seeds for train/val.
    """

    def __init__(self, size: int, seed: int = 0, random_word_prob: float = 0.2):
        self.size = size
        self.seed = seed
        self.random_word_prob = random_word_prob
        self.fonts = available_fonts()

    def __len__(self) -> int:
        return self.size

    def _word_for(self, rng: random.Random) -> str:
        if rng.random() < self.random_word_prob:
            n = rng.randint(2, MAX_LABEL_LEN)
            return "".join(rng.choice(CHARS) for _ in range(n))
        return rng.choice(WORDS)

    def __getitem__(self, idx: int):
        if not 0 <= idx < self.size:
            raise IndexError(idx)
        rng = random.Random(self.seed * 1_000_003 + idx)
        word = self._word_for(rng)
        img = render_word(word, rng, self.fonts)
        x = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        y = torch.tensor(encode(word), dtype=torch.long)
        return x, y, word


def collate(batch):
    """Pad labels and stack images. Returns (images, targets, target_lengths, words)."""
    xs, ys, words = zip(*batch)
    images = torch.stack(xs)
    target_lengths = torch.tensor([len(y) for y in ys], dtype=torch.long)
    targets = torch.cat(ys)
    return images, targets, target_lengths, list(words)


class IAMWordsDataset(Dataset):
    """Optional hook for the real IAM words dataset (not downloaded here).

    Expects the standard IAM layout:
        root/words.txt
        root/words/<form-prefix>/<form>/<word-id>.png
    Only 'ok'-segmented, charset-compatible words are kept.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        index = self.root / "words.txt"
        if not index.exists():
            raise FileNotFoundError(
                f"IAM index not found at {index}. Download the IAM words dataset "
                "(registration required at fki.tic.heia-fr.ch) and point root at it."
            )
        self.samples: list[tuple[Path, str]] = []
        for line in index.read_text(errors="ignore").splitlines():
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 9 or parts[1] != "ok":
                continue
            word_id, text = parts[0], parts[-1].lower()
            if not text or any(c not in CHARS for c in text) or len(text) > MAX_LABEL_LEN:
                continue
            a, b = word_id.split("-")[0], "-".join(word_id.split("-")[:2])
            path = self.root / "words" / a / b / f"{word_id}.png"
            if path.exists():
                self.samples.append((path, text))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, word = self.samples[idx]
        img = Image.open(path).convert("L").resize((IMG_W, IMG_H), Image.BILINEAR)
        arr = 1.0 - np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).unsqueeze(0)
        y = torch.tensor(encode(word), dtype=torch.long)
        return x, y, word
