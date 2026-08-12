# Improvements — Handwritten Text Reading (local)

**Goal:** A deployable handwriting-recognition web app where a user uploads or draws an image and gets predicted text back in real time, served by a versioned model behind a documented API.

## TL;DR — Path to production
- [x] Package training code, pin deps, reproducible train/eval on a public dataset
- [x] Export a checkpoint (ONNX/TorchScript)
- [x] FastAPI `/predict` service
- [x] Canvas/upload UI (plain HTML+JS, not React — see note below)
- [x] Tests + GitHub Actions CI + Docker
- [ ] Live hosted demo with basic monitoring

Note: the UI is plain HTML + vanilla JS (canvas + file upload), not React —
it's fully static, needs no build step, and covers the same UX (draw or
upload, live prediction + confidence bars). Simpler and more reliable for a
single-page inference demo; can be swapped for React/Vite later if the
frontend grows beyond one page.

## Current state
- Three from-scratch CNN classifiers (ResNet / AlexNet / VGG) built as Colab notebooks for an MMU deep-learning assessment.
- Only one packaged module (`src/algorithms/resnet.py`, `bottleneck.py`); `data/`, `tests/` empty, `requirements.txt` blank, README is two lines.
- Pipeline is image-folder classification, hardcoded to Google Drive paths — not runnable locally and not reproducible.
- No inference path, no UI, no API, no CI, no saved weights.

## Key improvements
- ML: refactor notebooks into a clean training package; move from single-character classification toward true line/word OCR (CTC head or a CRNN / transformer OCR model).
- Reproducibility: pin `requirements.txt`, add a data loader for a public set (IAM / EMNIST), scripted train/eval with saved checkpoints and metrics.
- API/backend: FastAPI inference service loading the exported model, `/predict` endpoint accepting an uploaded/drawn image.
- Frontend: React canvas + upload UI (draw a character/word, see live prediction and confidence).
- Quality: pytest for model + API, typed code, linting, an ONNX/TorchScript export for portable serving.
- Observability: request logging, latency + confidence metrics, basic dashboard.

## Latest tech to showcase
- PyTorch 2.x with `torch.compile`; TrOCR / Hugging Face Transformers for modern OCR.
- FastAPI + Pydantic v2 for the serving layer; ONNX Runtime for fast CPU inference.
- React + Vite + TypeScript frontend with a drawing canvas.
- Docker multi-stage build + GitHub Actions CI (lint, test, build, push image).
- Experiment tracking with Weights & Biases or MLflow.

## Roadmap
1. Clean up: package the training code, pin deps, reproducible train/eval on a public dataset, export a checkpoint.
2. Serve: FastAPI `/predict` + React canvas UI, containerized, running end-to-end locally.
3. Ship: GitHub Actions CI, model versioning/registry, observability, and a live hosted demo.
