# Handwritten Text Recognition

A small end-to-end sequence-OCR project: a CRNN-CTC model that reads a single
handwritten word from an image, served behind a FastAPI `/predict` endpoint
with a draw-or-upload web page on top.

**Honest status:** the shipped checkpoint is **smoke-trained on synthetic,
computer-rendered "handwriting"** (system handwriting-style fonts, not real
human handwriting), for a few minutes on a CPU. It reliably reads clean
rendered words in a similar style to its training data. It has **not** been
trained or evaluated on real handwriting (e.g. IAM) and will likely do worse
on messy human strokes. See [Scaling to real handwriting (IAM)](#scaling-to-real-handwriting-iam)
for what's needed to close that gap.

## What it does

1. **Model**: a CRNN (CNN feature extractor + bidirectional LSTM + linear
   head) trained with CTC loss to transcribe a single word image (32x128
   grayscale) into lowercase a-z text — no fixed-length label assumption,
   no per-character segmentation needed.
2. **Serving**: the trained model is exported to ONNX and served by a
   FastAPI app. `POST /predict` accepts either a multipart image upload or a
   base64-encoded image in a JSON body, and returns the transcription plus
   a per-character confidence score.
3. **UI**: a static HTML/JS page (no build step) where you can draw a word
   on a canvas or upload an image and see the transcription with
   per-character confidence highlighting.
4. **Tests + CI**: pytest covers the data pipeline, model, CTC decoding,
   metrics, and the API end-to-end (upload -> transcription); GitHub Actions
   runs the suite (including a fresh smoke-train + ONNX export + Docker
   build) on every push.

## Architecture

```
                     ┌─────────────────────────┐
  PNG/JPG bytes  --> │  htr.preprocess          │  grayscale, ink-polarity
  (upload/base64)    │  bytes_to_model_input()  │  auto-detect, crop-to-ink,
                     └────────────┬─────────────┘  resize/pad to 32x128
                                  │ (1,1,32,128) float32
                                  v
                     ┌─────────────────────────┐
                     │  ONNX Runtime session     │  CRNN forward pass
                     │  checkpoints/crnn.onnx    │  -> (T=32, B=1, C=27)
                     └────────────┬─────────────┘  per-timestep log-probs
                                  │
                                  v
                     ┌─────────────────────────┐
                     │  greedy CTC collapse       │  argmax per timestep,
                     │  (in htr.api)              │  drop blanks/repeats,
                     └────────────┬─────────────┘  keep peak softmax prob
                                  │                 as that char's confidence
                                  v
                     { "text": "hello",
                       "mean_confidence": 0.98,
                       "chars": [{"char":"h","confidence":0.99}, ...] }
```

Model architecture (`src/htr/model.py`):

```
input (B,1,32,128)
  -> 4x [Conv3x3 -> BatchNorm -> ReLU -> MaxPool]   (width /4, height /16)
  -> reshape to (B, T=32, 256)
  -> 2-layer bidirectional LSTM (hidden=128)
  -> Linear -> log_softmax
  -> output (T=32, B, 27)   # 26 letters + CTC blank
```

Training uses `nn.CTCLoss`; inference uses greedy (best-path) CTC decoding —
good enough at this vocabulary/quality level; a beam-search + language-model
decoder would help most on ambiguous real handwriting.

### Why synthetic data during training?

`src/htr/data.py` renders random English words on the fly using the
system's installed handwriting-style fonts (Bradley Hand, Chalkboard, Comic
Sans, Apple Chancery, Noteworthy on macOS; DejaVu/Liberation as a Linux CI
fallback) with small rotation/blur/noise augmentation. This needs **zero
dataset download**, is deterministic per `(seed, index)` so train/val splits
are reproducible, and is enough to validate the whole pipeline (data ->
model -> CTC loss -> decode -> metrics -> export -> serving) without
waiting on external data or GPU time. It is not a substitute for training on
real handwriting — see below.

## Quickstart

Requires Python 3.12 (via [uv](https://docs.astral.sh/uv/)).

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python -e ".[dev]"

# Train on synthetic word images rendered on the fly (no dataset download, ~5 min CPU)
.venv/bin/python -m htr.train --epochs 16 --train-size 3000 --lr 1e-3 --out checkpoints/crnn.pt

# Evaluate CER/WER on a held-out synthetic split
.venv/bin/python -m htr.evaluate --checkpoint checkpoints/crnn.pt --size 300

# Export to ONNX (with onnxruntime parity check)
.venv/bin/python -m htr.export_onnx --checkpoint checkpoints/crnn.pt --out checkpoints/crnn.onnx

# Run the API (loads checkpoints/crnn.onnx)
.venv/bin/python -m uvicorn htr.api:app --reload --port 8000
# -> open http://localhost:8000/ to draw/upload a word
# -> or curl -X POST http://localhost:8000/predict -F "file=@word.png"

# Tests
.venv/bin/python -m pytest -q
```

Smoke-scale reference result (CPU, ~5 min train): held-out synthetic CER
0.011 / WER 0.047 (see `checkpoints/crnn.pt` metadata; random-guess baseline
is ~1.0). Treat this as "the pipeline is correct and learns", not as a
claim about real-world handwriting accuracy.

### API

`POST /predict` — either:
- multipart form upload, field name `file`, or
- JSON body `{"image_base64": "<...>"}` (a `data:image/...;base64,` prefix
  is accepted and stripped).

Response:
```json
{
  "text": "hello",
  "mean_confidence": 0.98,
  "chars": [
    {"char": "h", "confidence": 0.997},
    {"char": "e", "confidence": 0.995},
    {"char": "l", "confidence": 0.996},
    {"char": "l", "confidence": 0.959},
    {"char": "o", "confidence": 0.965}
  ]
}
```

`GET /health` — liveness/readiness + whether the ONNX model loaded.
`GET /docs` — interactive OpenAPI/Swagger UI (from FastAPI, for free).

### Docker

```bash
docker build -t htr-api .
docker run --rm -p 8000:8000 htr-api
# open http://localhost:8000/
```

The image is CPU/onnxruntime-only (no PyTorch at runtime) and only copies
the exported `checkpoints/crnn.onnx` plus the small serving modules — you
must train + export a checkpoint before building the image (it will fail
the `COPY checkpoints/crnn.onnx` step otherwise, by design, so you don't
accidentally ship a stale/missing model).

### CI

`.github/workflows/ci.yml` runs on every push/PR:
1. Unit tests (data/model/decode/metrics — no checkpoint needed).
2. A fresh smoke-train (same recipe as the quickstart above) + ONNX export,
   so CI is proof the training pipeline still reproducibly works, not just
   that old code compiles.
3. API tests against that freshly exported model.
4. A Docker build + container smoke test (`/health` returns
   `model_loaded: true`).

## Scaling to real handwriting (IAM)

The model, training loop, CTC loss, and metrics are already
dataset-agnostic — swapping in real handwriting is a data-layer change, not
an architecture change. To do it:

1. **Get the data.** Register and download the
   [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
   (words images + `words.txt`; free for research/non-commercial use,
   registration required — not something this repo can auto-download).
2. **Use the existing hook.** `src/htr/data.py` already has
   `IAMWordsDataset(root)`: it parses the standard IAM `words.txt` layout,
   keeps only `ok`-segmented, charset-compatible (a-z, <=10 chars) samples,
   and returns the same `(image, label, word)` shape the synthetic dataset
   does — so `train.py`/`evaluate.py` work unchanged, just pass a different
   dataset.
3. **Widen the charset.** IAM includes uppercase, digits, and punctuation;
   `src/htr/charset.py` is intentionally a single small constant (`CHARS`)
   to make this a one-line change plus a retrain — but do it deliberately,
   since it changes the output layer size and invalidates old checkpoints.
4. **Expect to retrain longer, on more data, ideally with a GPU.** IAM's
   words split is ~115k word images; the CRNN here is small enough to train
   on a single GPU in a few hours, or CPU overnight. Increase
   `--train-size`/`--epochs` accordingly and watch validation CER/WER
   (`htr.evaluate`) rather than training loss.
5. **Consider line-level input**, not just isolated words, if the real goal
   is transcribing full handwritten pages — that needs a wider input image
   and a longer max label length (`MAX_LABEL_LEN`), plus likely a stronger
   decoder (beam search + character/word language model) since CTC greedy
   decoding degrades faster on longer, messier sequences.
6. **Re-run `export_onnx.py` and redeploy** — the FastAPI/ONNX serving path
   does not need to change; only the checkpoint and (if the charset changed)
   `NUM_CLASSES` do.

## Project layout

```
src/htr/
  charset.py        a-z + CTC blank encode/decode
  constants.py       shared IMG_H/IMG_W/MAX_LABEL_LEN (no torch/PIL import,
                      so the lean serving path doesn't pull in training deps)
  data.py            SyntheticWordDataset (on-the-fly PIL rendering) +
                      IAMWordsDataset hook + collate_fn
  model.py           CRNN (CNN + BiLSTM + linear)
  decode.py          greedy CTC decode (used by train/evaluate)
  metrics.py         Levenshtein-based CER/WER
  preprocess.py      arbitrary-image -> model-input normalization (serving)
  api.py             FastAPI app: /predict, /health, static UI
  static/index.html  draw/upload demo page (vanilla HTML/JS, no build step)
  train.py / evaluate.py / export_onnx.py   CLIs (python -m htr.<name>)
tests/               pytest: data, model, decode, metrics, API
checkpoints/         crnn.pt (PyTorch) + crnn.onnx (exported) — gitignored,
                      generate them via the quickstart above
Dockerfile           CPU/onnxruntime-only serving image
.github/workflows/ci.yml   test -> smoke-train -> export -> API test -> docker build
nb/, src/algorithms/  legacy: earlier Colab notebooks + a ResNet single-
                      character classifier, superseded by src/htr/ but kept
                      for history
```

## License

MIT. See [LICENSE](LICENSE) if present, or treat as MIT otherwise.
