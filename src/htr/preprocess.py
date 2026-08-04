"""Preprocessing for arbitrary uploaded/drawn images at inference time.

Mirrors the normalization used by ``htr.data.render_word`` for synthetic
training images: grayscale, ink inverted to high values in [0, 1], fit into
IMG_H x IMG_W keeping aspect ratio, centered vertically, left-padded.

Uploaded images may be photos/scans of handwriting on a light or dark
background, in RGB or with an alpha channel (e.g. PNG from an HTML canvas
with a transparent background) — this module normalizes all of those to the
same convention the model was trained on.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from .constants import IMG_H, IMG_W


def _flatten_alpha(img: Image.Image) -> Image.Image:
    """Composite an RGBA/LA image onto a white background, then grayscale."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("L")


def load_image(data: bytes) -> Image.Image:
    """Decode raw image bytes into a flattened grayscale PIL image."""
    img = Image.open(io.BytesIO(data))
    img.load()
    return _flatten_alpha(img)


def to_model_input(img: Image.Image) -> np.ndarray:
    """Convert a grayscale PIL image to a (1, IMG_H, IMG_W) float32 array.

    Auto-detects ink polarity (assumes the majority-color corners/border are
    background), inverts so ink -> high values, and fits the content into
    the model's expected canvas the same way training images are built.
    """
    arr = np.asarray(img, dtype=np.float32) / 255.0  # 0=black .. 1=white

    # Decide background level from the border pixels; handle both dark-on-light
    # (typical scan/photo) and light-on-dark (e.g. white ink on black canvas).
    border = np.concatenate(
        [arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]]
    )
    bg_level = float(np.median(border))
    if bg_level < 0.5:
        # Background is dark -> ink is already "high value"; keep as-is.
        ink = arr
    else:
        # Background is light -> invert so ink becomes high value.
        ink = 1.0 - arr

    # Threshold-free bbox: find where ink deviates meaningfully from 0.
    mask = ink > 0.08
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return np.zeros((1, IMG_H, IMG_W), dtype=np.float32)

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    cropped = ink[y0:y1, x0:x1]

    crop_img = Image.fromarray((cropped * 255).astype(np.uint8), mode="L")
    h, w = cropped.shape
    scale = min((IMG_H - 4) / h, (IMG_W - 4) / w)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = crop_img.resize((new_w, new_h), Image.BILINEAR)

    out = Image.new("L", (IMG_W, IMG_H), color=0)
    x_off = max(0, (IMG_W - new_w) // 2)
    y_off = max(0, (IMG_H - new_h) // 2)
    out.paste(resized, (x_off, y_off))

    out_arr = np.asarray(out, dtype=np.float32) / 255.0
    return out_arr[np.newaxis, :, :]  # (1, H, W)


def bytes_to_model_input(data: bytes) -> np.ndarray:
    """Full pipeline: raw image bytes -> (1, 1, IMG_H, IMG_W) batch array."""
    img = load_image(data)
    x = to_model_input(img)  # (1, H, W)
    return x[np.newaxis, :, :, :]  # (1, 1, H, W)


__all__ = ["load_image", "to_model_input", "bytes_to_model_input", "IMG_H", "IMG_W"]
