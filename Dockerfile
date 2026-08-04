# Handwritten Text Recognition — inference API image.
#
# CPU-only, onnxruntime-based serving (no torch/training deps at runtime).
# Build:
#   docker build -t htr-api .
# Run:
#   docker run --rm -p 8000:8000 htr-api
# Then open http://localhost:8000/ or POST to http://localhost:8000/predict

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Only the runtime deps needed to *serve* the model (no torch/onnx-export
# toolchain) — keeps the image small and the attack surface minimal.
COPY docker-requirements.txt ./docker-requirements.txt
RUN pip install --no-cache-dir -r docker-requirements.txt

# App code and the exported ONNX model + the small charset/preprocess/api
# modules it depends on. Deliberately NOT copying htr/data.py, decode.py,
# train.py, evaluate.py, export_onnx.py — those pull in torch, which the
# onnxruntime-only serving path doesn't need.
COPY src/htr/__init__.py src/htr/constants.py src/htr/charset.py \
     src/htr/preprocess.py src/htr/api.py ./src/htr/
COPY src/htr/static ./src/htr/static
COPY checkpoints/crnn.onnx ./checkpoints/crnn.onnx

ENV HTR_ONNX_PATH=/app/checkpoints/crnn.onnx
ENV PYTHONPATH=/app/src

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()" || exit 1

CMD ["uvicorn", "htr.api:app", "--host", "0.0.0.0", "--port", "8000"]
