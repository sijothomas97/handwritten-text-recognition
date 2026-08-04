"""API tests using rendered synthetic word images, driven through the real
ONNX model (requires checkpoints/crnn.onnx to exist; skipped otherwise so
the suite still passes in environments that haven't exported a model yet).
"""

from __future__ import annotations

import base64
import io
import random
import zlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from htr.data import available_fonts, render_word
from htr.metrics import cer as char_error_rate

ONNX_PATH = Path(__file__).resolve().parents[1] / "checkpoints" / "crnn.onnx"

pytestmark = pytest.mark.skipif(
    not ONNX_PATH.exists(),
    reason=f"no exported model at {ONNX_PATH}; run `python -m htr.export_onnx` first",
)


def _word_png_bytes(word: str, seed: int) -> bytes:
    """Render a word with the same pipeline used for training, then save it
    as a normal light-background PNG (as a real upload would look), matching
    the polarity `htr.preprocess` expects to auto-detect and invert.
    """
    fonts = available_fonts()
    rng = random.Random(seed)
    arr = render_word(word, rng, fonts)  # ink = high values in [0, 1]
    img_arr = ((1.0 - arr) * 255).astype("uint8")  # back to dark-ink-on-light
    img = Image.fromarray(img_arr, mode="L").convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    from htr.api import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def word_image_bytes():
    return _word_png_bytes("hello", seed=12345)


def assert_close_transcription(word: str, text: str, max_cer: float = 0.34) -> None:
    """The model is smoke-trained (see README); tolerate a small edit-distance
    gap rather than demanding byte-exact output, so CI isn't flaky against a
    freshly (re)trained checkpoint while still catching real regressions
    (e.g. an empty/garbage transcription).
    """
    rate = char_error_rate([word], [text])
    assert rate <= max_cer, f"transcription {text!r} too far from {word!r} (CER {rate:.2f})"


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_index_page_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Handwritten Text Recognition" in resp.text


def test_predict_multipart_upload_returns_sane_transcription(client, word_image_bytes):
    resp = client.post(
        "/predict",
        files={"file": ("word.png", word_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert_close_transcription("hello", data["text"])
    assert 0.0 <= data["mean_confidence"] <= 1.0
    assert len(data["chars"]) == len(data["text"])
    for ch in data["chars"]:
        assert len(ch["char"]) == 1 and ch["char"].isalpha()
        assert 0.0 <= ch["confidence"] <= 1.0


def test_predict_base64_json_matches_multipart(client, word_image_bytes):
    b64 = base64.b64encode(word_image_bytes).decode()
    resp = client.post("/predict", json={"image_base64": b64})
    assert resp.status_code == 200
    multipart_resp = client.post(
        "/predict", files={"file": ("word.png", word_image_bytes, "image/png")}
    )
    assert resp.json() == multipart_resp.json(), "base64 and multipart paths must decode identically"


def test_predict_base64_data_uri_prefix_is_stripped(client, word_image_bytes):
    b64 = base64.b64encode(word_image_bytes).decode()
    resp = client.post("/predict", json={"image_base64": f"data:image/png;base64,{b64}"})
    assert resp.status_code == 200
    assert_close_transcription("hello", resp.json()["text"])


def test_predict_multiple_words_are_plausible(client):
    for word in ["quick", "jump", "zebra", "cat"]:
        # zlib.crc32 (unlike the built-in str hash) is stable across
        # processes/runs, keeping this test's rendering deterministic.
        seed = zlib.crc32(word.encode()) % 100000
        data_bytes = _word_png_bytes(word, seed=seed)
        resp = client.post("/predict", files={"file": ("w.png", data_bytes, "image/png")})
        assert resp.status_code == 200
        assert_close_transcription(word, resp.json()["text"])


def test_predict_rejects_bad_content_type(client):
    resp = client.post("/predict", content=b"not an image", headers={"content-type": "text/plain"})
    assert resp.status_code == 415


def test_predict_rejects_empty_multipart_file(client):
    resp = client.post("/predict", files={"file": ("empty.png", b"", "image/png")})
    assert resp.status_code == 400


def test_predict_rejects_invalid_base64(client):
    resp = client.post("/predict", json={"image_base64": "not-valid-base64!!"})
    assert resp.status_code == 400


def test_predict_rejects_missing_file_field(client):
    resp = client.post(
        "/predict",
        files={"wrong_field": ("word.png", b"123", "image/png")},
    )
    assert resp.status_code == 400
