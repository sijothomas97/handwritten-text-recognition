"""API tests for the FastAPI inference service.

Uses the trained ONNX model at models/smoke/model.onnx when present; otherwise
exports a fresh (untrained) model so the API contract is always testable.
Accuracy assertions only run against the trained model + local MNIST data.
"""

import base64
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from htr.service import create_app, preprocess  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TRAINED_ONNX = REPO / "models" / "smoke" / "model.onnx"
MNIST_RAW = REPO / "data" / "MNIST" / "raw" / "t10k-images-idx3-ubyte"


@pytest.fixture(scope="module")
def model_path(tmp_path_factory) -> Path:
    if TRAINED_ONNX.exists():
        return TRAINED_ONNX
    # Fall back to exporting an untrained model (CI without a prior train run).
    import torch

    from htr.config import Config
    from htr.export_onnx import export
    from htr.model import build_model

    tmp = tmp_path_factory.mktemp("model")
    cfg = Config()
    torch.manual_seed(0)
    model = build_model(cfg.model, 10)
    ckpt = tmp / "ckpt.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "config": cfg.to_dict(),
         "num_classes": 10, "epoch": 0},
        ckpt,
    )
    return export(ckpt, tmp / "model.onnx")


@pytest.fixture(scope="module")
def client(model_path) -> TestClient:
    return TestClient(create_app(model_path))


def sample_image_bytes(fmt: str = "PNG", invert: bool = False) -> bytes:
    """Generate a synthetic digit-like image: white stroke on black, 280x280."""
    img = Image.new("L", (280, 280), 255 if invert else 0)
    draw = ImageDraw.Draw(img)
    ink = 0 if invert else 255
    # A crude "1": vertical stroke with a serif, thick like canvas drawing.
    draw.line([(150, 40), (150, 240)], fill=ink, width=24)
    draw.line([(110, 80), (150, 40)], fill=ink, width=24)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def assert_valid_prediction(data: dict) -> None:
    assert data["label"] in [str(d) for d in range(10)]
    assert 0.0 <= data["confidence"] <= 1.0
    assert set(data["probabilities"]) == {str(d) for d in range(10)}
    assert abs(sum(data["probabilities"].values()) - 1.0) < 0.01
    assert data["latency_ms"] >= 0


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_file_upload(client):
    r = client.post(
        "/predict", files={"file": ("digit.png", sample_image_bytes(), "image/png")}
    )
    assert r.status_code == 200
    assert_valid_prediction(r.json())


def test_predict_base64_data_url(client):
    b64 = base64.b64encode(sample_image_bytes()).decode()
    r = client.post("/predict", json={"image_b64": "data:image/png;base64," + b64})
    assert r.status_code == 200
    assert_valid_prediction(r.json())


def test_predict_plain_base64(client):
    b64 = base64.b64encode(sample_image_bytes("JPEG")).decode()
    r = client.post("/predict", json={"image_b64": b64})
    assert r.status_code == 200
    assert_valid_prediction(r.json())


def test_inverted_input_matches_normal(client):
    """Black-on-white and white-on-black versions should predict the same label."""
    labels = []
    for invert in (False, True):
        r = client.post(
            "/predict",
            files={"file": ("d.png", sample_image_bytes(invert=invert), "image/png")},
        )
        assert r.status_code == 200
        labels.append(r.json()["label"])
    assert labels[0] == labels[1]


def test_predict_no_payload(client):
    assert client.post("/predict").status_code == 400


def test_predict_bad_base64(client):
    r = client.post("/predict", json={"image_b64": "not-base64!!!"})
    assert r.status_code == 400


def test_predict_not_an_image(client):
    r = client.post("/predict", files={"file": ("x.png", b"hello world", "image/png")})
    assert r.status_code == 400


def test_missing_model_returns_503(tmp_path):
    app = create_app(tmp_path / "nope.onnx")
    c = TestClient(app)
    assert c.get("/health").json()["model_loaded"] is False
    r = c.post("/predict", files={"file": ("d.png", sample_image_bytes(), "image/png")})
    assert r.status_code == 503


def test_index_serves_ui(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "canvas" in r.text


def test_preprocess_matches_training_transform():
    """Service preprocessing must equal the torchvision training transform."""
    torch = pytest.importorskip("torch")

    from htr.config import DataConfig
    from htr.data import build_transform

    rng = np.random.default_rng(0)
    # Dark-background image so the service's auto-invert stays off.
    arr = (rng.random((28, 28)) * 80).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    served = preprocess(img)
    trained = build_transform(DataConfig())(img).unsqueeze(0)
    np.testing.assert_allclose(served, trained.numpy(), rtol=1e-5, atol=1e-6)
    assert isinstance(served, np.ndarray) and served.dtype == np.float32
    del torch


def _load_mnist_samples(n: int):
    """Read n test images + labels straight from the raw IDX files."""
    images = np.frombuffer(MNIST_RAW.read_bytes(), dtype=np.uint8, offset=16)
    images = images.reshape(-1, 28, 28)
    labels_path = MNIST_RAW.parent / "t10k-labels-idx1-ubyte"
    labels = np.frombuffer(labels_path.read_bytes(), dtype=np.uint8, offset=8)
    return images[:n], labels[:n]


@pytest.mark.skipif(
    not (TRAINED_ONNX.exists() and MNIST_RAW.exists()),
    reason="needs trained model and local MNIST data",
)
def test_predict_real_mnist_samples(client):
    """The trained model should get most real MNIST test digits right."""
    images, labels = _load_mnist_samples(40)
    correct = 0
    for img_arr, label in zip(images, labels):
        buf = io.BytesIO()
        Image.fromarray(img_arr, mode="L").save(buf, format="PNG")
        r = client.post("/predict", files={"file": ("d.png", buf.getvalue(), "image/png")})
        assert r.status_code == 200
        correct += r.json()["label"] == str(label)
    assert correct / len(images) >= 0.7  # smoke model scores ~0.88
