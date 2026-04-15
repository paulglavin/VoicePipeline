FROM python:3.12-slim

# System libraries required by audio processing and model inference
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Step 1: install PyTorch (CUDA 12.8) for STT only ─────────────────
# VAD and speaker ID now use pure ONNX (onnxruntime); only Granite STT
# still requires torch.  torchaudio is no longer needed.
# Requires CUDA 12.8+ on the host and nvidia-container-toolkit.
# For CPU-only, replace the index URL with https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu128 \
    "torch>=2.7.0"

# ── Step 2: install the remaining requirements ─────────────────────────
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ─────────────────────────────────────────────────
COPY pipeline/ ./pipeline/
COPY api/      ./api/
COPY ui/       ./ui/

# Pre-create runtime directories so the container starts cleanly even
# before the bind mounts are attached (plain docker run without compose).
RUN mkdir -p /data/audio /data/embeddings /models

# ── Environment defaults (all overridable at runtime) ──────────────────
ENV DB_PATH=/data/speaker.db \
    AUDIO_DIR=/data/audio \
    EMBEDDING_DIR=/data/embeddings \
    MODEL_DIR=/models \
    WYOMING_HOST=0.0.0.0 \
    WYOMING_PORT=10300 \
    HOST=0.0.0.0 \
    PORT=8000

# HTTP management API + Wyoming STT
EXPOSE 8000 10300

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
