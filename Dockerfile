# Lightweight serving image: FastAPI + ONNX Runtime (no torch/torchvision).
FROM python:3.12-slim

WORKDIR /app

# System deps for Pillow (jpeg/png) — slim base needs libjpeg/zlib at runtime only
# via the wheel's bundled libs, so no apt packages are required for Pillow wheels.

COPY requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

# App code (package layout: src/htr is the service, src/algorithms is imported
# by htr.model but not by htr.service; included anyway since it's tiny).
COPY src/htr ./src/htr
COPY src/algorithms ./src/algorithms
COPY pyproject.toml README.md ./

RUN pip install --no-cache-dir --no-deps -e .

# Exported model baked into the image for a self-contained demo.
COPY models/smoke/model.onnx ./models/smoke/model.onnx

ENV HTR_MODEL_PATH=/app/models/smoke/model.onnx
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()" || exit 1

CMD ["uvicorn", "htr.service:app", "--host", "0.0.0.0", "--port", "8000"]
