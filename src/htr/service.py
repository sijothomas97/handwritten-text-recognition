"""FastAPI inference service serving the exported ONNX model.

Run locally:
    uvicorn htr.service:app --port 8000

The model path is taken from the HTR_MODEL_PATH env var (default:
models/smoke/model.onnx relative to the current working directory).
Preprocessing mirrors htr.data.build_transform (grayscale -> resize ->
scale to [0,1] -> normalise with MNIST stats) using only PIL + numpy so the
serving image does not need torch.
"""

from __future__ import annotations

import base64
import binascii
import io
import os
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from starlette.datastructures import UploadFile

# Must match htr.data._STATS["mnist"] and configs/*.yaml image_size.
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
IMAGE_SIZE = 28
LABELS = [str(d) for d in range(10)]

DEFAULT_MODEL_PATH = "models/smoke/model.onnx"
STATIC_DIR = Path(__file__).resolve().parent / "static"


class Base64Request(BaseModel):
    """JSON body for canvas submissions: a base64 (optionally data-URL) PNG."""

    image_b64: str


class Prediction(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]
    latency_ms: float


def preprocess(img: Image.Image) -> np.ndarray:
    """PIL image -> float32 array of shape (1, 1, 28, 28), normalised.

    Mirrors the training transform. Additionally auto-inverts light-background
    images (e.g. black ink on white paper/canvas) since MNIST is white-on-black.
    """
    img = img.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.mean() > 0.5:  # light background -> invert to match MNIST
        arr = 1.0 - arr
    arr = (arr - MNIST_MEAN) / MNIST_STD
    return arr[np.newaxis, np.newaxis, :, :]


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def _decode_base64(data: str) -> bytes:
    if "," in data and data.strip().startswith("data:"):
        data = data.split(",", 1)[1]
    try:
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {exc}") from exc


def create_app(model_path: str | os.PathLike | None = None) -> FastAPI:
    import onnxruntime as ort

    path = Path(model_path or os.environ.get("HTR_MODEL_PATH", DEFAULT_MODEL_PATH))
    app = FastAPI(title="Handwritten character recognition", version="1.0.0")
    app.state.model_path = path
    app.state.session = None
    if path.exists():
        app.state.session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )

    def _predict_bytes(raw: bytes) -> Prediction:
        if app.state.session is None:
            raise HTTPException(
                status_code=503,
                detail=f"Model not loaded (looked for {app.state.model_path}). "
                "Train and export it first: see README quickstart.",
            )
        try:
            img = Image.open(io.BytesIO(raw))
            img.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=400, detail=f"Not a valid image: {exc}") from exc
        start = time.perf_counter()
        x = preprocess(img)
        logits = app.state.session.run(["logits"], {"image": x})[0][0]
        probs = _softmax(logits)
        latency_ms = (time.perf_counter() - start) * 1000
        best = int(probs.argmax())
        return Prediction(
            label=LABELS[best],
            confidence=round(float(probs[best]), 4),
            probabilities={lab: round(float(p), 4) for lab, p in zip(LABELS, probs)},
            latency_ms=round(latency_ms, 2),
        )

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok" if app.state.session is not None else "degraded",
            "model_loaded": app.state.session is not None,
            "model_path": str(app.state.model_path),
        }

    @app.post("/predict", response_model=Prediction)
    async def predict(request: Request) -> Prediction:
        """Accepts either a multipart 'file' upload or JSON {'image_b64': ...}."""
        content_type = request.headers.get("content-type", "")
        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            file = form.get("file")
            if not isinstance(file, UploadFile):
                raise HTTPException(
                    status_code=400, detail="Multipart request must include a 'file' part"
                )
            return _predict_bytes(await file.read())
        if content_type.startswith("application/json"):
            try:
                body = Base64Request.model_validate(await request.json())
            except ValueError as exc:
                raise HTTPException(
                    status_code=400, detail=f"Expected JSON {{'image_b64': ...}}: {exc}"
                ) from exc
            return _predict_bytes(_decode_base64(body.image_b64))
        raise HTTPException(
            status_code=400,
            detail="Send either a multipart 'file' upload or JSON {'image_b64': ...}",
        )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
