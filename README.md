# hand-written-text-reading

Hand-written text reading using deep learning models. From-scratch ResNet
classifiers (originally Colab notebooks under `nb/`) refactored into a
reproducible, config-driven training package.

## Setup

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python -r requirements.txt
uv pip install -p .venv/bin/python -e .
```

## Train / evaluate / export

```bash
# Smoke run: 1 epoch on a 4k MNIST subset (~1 min CPU), auto-downloads data/
.venv/bin/python -m htr.train --config configs/smoke.yaml

# Full MNIST run
.venv/bin/python -m htr.train --config configs/full_mnist.yaml

# Evaluate a checkpoint
.venv/bin/python -m htr.evaluate --checkpoint models/smoke/best.pt

# Export to ONNX
.venv/bin/python -m htr.export_onnx --checkpoint models/smoke/best.pt --out models/smoke/model.onnx
```

Checkpoints (`best.pt`, `last.pt`) and `metrics.json` are written to
`models/<run_name>/`. Datasets (MNIST, or EMNIST via config) are downloaded
automatically by torchvision into `data/`.

## Serve the API + UI

The FastAPI service loads the exported ONNX model and serves both the
`/predict` API and a static draw-or-upload HTML page.

```bash
# after training + exporting a model to models/smoke/model.onnx (see above)
.venv/bin/python -m uvicorn htr.service:app --port 8000
```

Then open http://127.0.0.1:8000/ in a browser to draw a digit on the canvas
or upload an image, or call the API directly:

```bash
# health check
curl http://127.0.0.1:8000/health

# predict from an image file (multipart upload)
curl -X POST http://127.0.0.1:8000/predict -F "file=@/path/to/digit.png"

# predict from base64 (what the canvas UI sends)
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"image_b64\": \"$(base64 -i /path/to/digit.png)\"}"
```

`/predict` returns `{"label": "7", "confidence": 0.98, "probabilities": {...}, "latency_ms": 3.9}`.
The model path is configurable via `HTR_MODEL_PATH` (defaults to
`models/smoke/model.onnx`, resolved relative to the current working directory).

## Docker

A lightweight image (FastAPI + ONNX Runtime only, no torch/torchvision) bakes
in the exported `models/smoke/model.onnx`:

```bash
docker build -t htr-service .
docker run -p 8000:8000 htr-service
curl http://127.0.0.1:8000/health
```

Rebuild the image after re-exporting a new model to pick up the change.

## Tests

```bash
.venv/bin/python -m pytest
```

Covers preprocessing transforms, model forward shapes (resnet18/34/50),
ONNX-vs-PyTorch output parity, and the FastAPI service (`/health`, `/predict`
via multipart + base64, error handling, and prediction accuracy against real
MNIST samples when a trained model is present).

## CI

`.github/workflows/ci.yml` runs on every push/PR: ruff lint, a smoke training
run to produce a real checkpoint, the full pytest suite, and a Docker build +
container health-check smoke test.

## Layout

- `src/algorithms/` — from-scratch ResNet blocks and configs (BasicBlock, Bottleneck, resnet18–152)
- `src/htr/` — config, data loaders, training/eval loops, CLIs, ONNX export, `service.py` (FastAPI app), `static/` (canvas + upload UI)
- `configs/` — YAML run configs
- `tests/` — model, transform, ONNX-parity, and API tests
- `Dockerfile` — serving-only image (`requirements-serve.txt`, no torch)
- `nb/` — original Colab notebooks (kept for reference)
