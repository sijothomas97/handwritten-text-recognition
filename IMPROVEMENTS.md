# Improvements — Handwritten Text Recognition

**Goal:** A deployed web app where a user draws or uploads handwriting and gets recognized text back in real time, served by a versioned PyTorch model behind a documented API with a live demo link.

## TL;DR — Path to production
- [x] Reproducible training/eval package with pinned deps
- [x] Upgrade to sequence OCR (CRNN-CTC / TrOCR) with CER/WER metrics
- [x] Export model (ONNX) + FastAPI serving
- [x] Draw/upload web UI
- [x] Tests + GitHub Actions CI + Docker
- [ ] Public live demo

## Current state
- `src/htr/` package: CRNN + CTC sequence OCR (word recognition), synthetic word-image dataset rendered on the fly with PIL system fonts (zero download), plus an optional `IAMWordsDataset` hook for real IAM data later.
- CLIs: `python -m htr.train`, `python -m htr.evaluate` (CER/WER + example predictions), `python -m htr.export_onnx` (with onnxruntime parity check).
- Verified smoke run (CPU, ~5-20 min depending on machine load): held-out synthetic CER 0.011-0.013 / WER 0.047-0.053 across two independent runs; checkpoint at `checkpoints/crnn.pt`, ONNX at `checkpoints/crnn.onnx` (parity max diff ~1e-5).
- **Serving**: `src/htr/api.py` — FastAPI `/predict` (multipart upload or base64 JSON) returns transcription + per-character confidence via ONNX Runtime; `/health` reports model load status; `src/htr/preprocess.py` normalizes arbitrary uploaded images (auto ink-polarity detection, crop-to-content, resize/pad to 32x128) to match training-time preprocessing.
- **UI**: `src/htr/static/index.html` — vanilla HTML/JS draw-on-canvas + file-upload page, no build step, calls `/predict` and renders per-character confidence chips.
- **Docker**: `Dockerfile` builds a lean CPU/onnxruntime-only serving image (no PyTorch at runtime); verified locally — built, ran, and correctly transcribed a test word end-to-end via `docker run` + `curl`.
- **CI**: `.github/workflows/ci.yml` — unit tests, then a fresh smoke-train + ONNX export (reproducibility check, not just "old artifacts still work"), then API tests against that freshly exported model, then a Docker build + container `/health` smoke test. Not run on GitHub itself in this phase (no push), but the equivalent commands were run locally end-to-end and passed.
- 31 pytest tests total (21 data/model/decode/metrics + 10 new API tests covering multipart upload, base64 JSON, data-URI prefix stripping, and error paths) — all passing. API content-correctness tests use CER-based tolerance (not byte-exact match) against the smoke-trained model so they aren't flaky across retrains.
- Verified end-to-end manually: rendered word images ("hello", "water", "quick", "jump", "zebra") posted to a running API instance all returned correct or near-correct transcriptions with high per-character confidence.
- Pinned deps in `pyproject.toml` + `requirements.txt` (Python 3.12 via `uv venv -p 3.12`); serving-only subset pinned separately in `docker-requirements.txt` for the Docker image.
- Legacy: Colab notebooks (ResNet/AlexNet/VGG single-char classification) in `nb/`, unwired ResNet in `src/algorithms/`.
- Not done (out of scope / requires infra this environment doesn't have): public live demo / cloud deployment — the Dockerfile and CI config are the deployable artifacts, not deployed anywhere.

## Key improvements
- **ML/model:** move beyond single-char classification to real line/word recognition (CRNN + CTC or a ViT-based encoder-decoder); train on IAM/EMNIST with proper CER/WER metrics.
- **Pipeline:** extract notebooks into a clean `src/` package — config-driven train/eval/infer CLI, checkpointing, experiment tracking.
- **API/backend:** FastAPI inference service loading an exported ONNX/TorchScript model, `/predict` endpoint with image upload.
- **Frontend:** React/Next.js page with a draw-on-canvas + file-upload UI that calls the API and shows the transcription.
- **Quality:** real `pytest` suite (data transforms, model forward, API contract), pinned deps, pre-commit lint/format.

## Latest tech to showcase
- PyTorch 2.x (`torch.compile`) + Hugging Face `transformers`/TrOCR as a strong baseline.
- ONNX Runtime or TorchScript export for fast, framework-free serving.
- FastAPI + Pydantic v2 backend, Next.js + TypeScript frontend.
- Docker multi-stage build, GitHub Actions CI (lint/test/build), deploy to HF Spaces or Fly.io/Render.
- Experiment tracking with Weights & Biases or MLflow.

## Roadmap
1. Refactor notebooks into a reproducible training/eval package with metrics and pinned deps.
2. Upgrade the model to sequence recognition (CRNN-CTC / TrOCR), export weights, add tests + CI.
3. Ship FastAPI + Next.js app in Docker with a public live demo and observability.
