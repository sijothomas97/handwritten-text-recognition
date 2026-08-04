"""FastAPI inference service for the CRNN-CTC handwriting recognizer.

Loads the exported ONNX model and serves:
    POST /predict         image upload (multipart/form-data, field "file")
                          or JSON body {"image_base64": "..."}
    GET  /health          liveness/readiness + model info
    GET  /                static demo page (draw or upload a word image)

Run:
    uvicorn htr.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import base64
import binascii
import os
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .charset import BLANK_IDX, IDX_TO_CHAR
from .preprocess import bytes_to_model_input

DEFAULT_MODEL_PATH = Path(os.environ.get("HTR_ONNX_PATH", "checkpoints/crnn.onnx"))
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Handwritten Text Recognition API",
    description="CRNN-CTC word recognizer (smoke-trained on synthetic data). "
    "POST an image to /predict to get a transcription with per-character confidence.",
    version="0.1.0",
)

_session: Optional[ort.InferenceSession] = None
_model_path: Optional[Path] = None


def get_session() -> ort.InferenceSession:
    global _session, _model_path
    if _session is None:
        if not DEFAULT_MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Model not found at {DEFAULT_MODEL_PATH}. Train and export it first: "
                    "`python -m htr.train ...` then `python -m htr.export_onnx ...`."
                ),
            )
        _session = ort.InferenceSession(
            str(DEFAULT_MODEL_PATH), providers=["CPUExecutionProvider"]
        )
        _model_path = DEFAULT_MODEL_PATH
    return _session


class PredictRequest(BaseModel):
    image_base64: str


class CharConfidence(BaseModel):
    char: str
    confidence: float


class PredictResponse(BaseModel):
    text: str
    mean_confidence: float
    chars: list[CharConfidence]


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def run_inference(image_batch: np.ndarray) -> PredictResponse:
    """image_batch: (1, 1, H, W) float32 array -> transcription + confidences."""
    session = get_session()
    (log_probs,) = session.run(None, {"image": image_batch.astype(np.float32)})
    # log_probs: (T, B, C) with B=1
    probs = np.exp(log_probs[:, 0, :])  # (T, C), already normalized (log_softmax)
    best_idx = probs.argmax(axis=1)  # (T,)
    best_prob = probs.max(axis=1)  # (T,)

    chars: list[CharConfidence] = []
    prev = BLANK_IDX
    for idx, p in zip(best_idx.tolist(), best_prob.tolist()):
        if idx != BLANK_IDX and idx != prev:
            chars.append(CharConfidence(char=IDX_TO_CHAR[idx], confidence=float(p)))
        prev = idx

    text = "".join(c.char for c in chars)
    mean_conf = float(np.mean([c.confidence for c in chars])) if chars else 0.0
    return PredictResponse(text=text, mean_confidence=mean_conf, chars=chars)


@app.get("/health")
def health():
    model_ok = DEFAULT_MODEL_PATH.exists()
    info = {"status": "ok" if model_ok else "degraded", "model_path": str(DEFAULT_MODEL_PATH)}
    if model_ok:
        try:
            get_session()
            info["model_loaded"] = True
        except HTTPException as e:
            info["status"] = "degraded"
            info["model_loaded"] = False
            info["detail"] = e.detail
    else:
        info["model_loaded"] = False
    return info


@app.post("/predict", response_model=PredictResponse)
async def predict(request: Request):
    """Accepts EITHER a multipart file upload (field "file") OR a JSON body
    {"image_base64": "..."}. Returns the greedy-decoded transcription plus a
    per-character confidence (softmax probability of the winning class at
    that decoded character's peak timestep).
    """
    content_type = request.headers.get("content-type", "")
    data: bytes

    if "multipart/form-data" in content_type:
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            raise HTTPException(status_code=400, detail="Missing multipart field 'file'")
        data = await upload.read()
    elif "application/json" in content_type:
        try:
            payload = await request.json()
            body = PredictRequest.model_validate(payload)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}") from None
        raw = body.image_base64
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[-1]
        try:
            data = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}") from None
    else:
        raise HTTPException(
            status_code=415,
            detail="Content-Type must be multipart/form-data (field 'file') "
            "or application/json ({'image_base64': '...'})",
        )

    if not data:
        raise HTTPException(status_code=400, detail="Empty image payload")

    try:
        image_batch = bytes_to_model_input(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}") from None

    return run_inference(image_batch)


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(STATIC_DIR / "index.html"))
