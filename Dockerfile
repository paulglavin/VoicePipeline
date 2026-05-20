FROM python:3.12-slim

# ENABLE_LOCAL_STT=true  — IBM Granite 4.0 1B Speech (requires CUDA 12.8+ at runtime, ~2 GB weights)
# ENABLE_FASTER_WHISPER=true — faster-whisper tiny/base/small (CPU or CUDA, ~40–490 MB weights)
# Both default to false; the slim build uses the remote OpenAI-compatible STT endpoint.
ARG ENABLE_LOCAL_STT=false
ARG ENABLE_FASTER_WHISPER=false

# System libraries required by audio processing and model inference
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-local-stt.txt requirements-faster-whisper.txt ./

# ── Base dependencies (no torch) ───────────────────────────────────
RUN pip install --no-cache-dir -r requirements.txt

# ── Local STT: PyTorch (CUDA 12.8) + transformers ──────────────────
# Skipped in the default slim build. Requires nvidia-container-toolkit
# and a CUDA 12.8-capable GPU on the host at runtime.
RUN if [ "$ENABLE_LOCAL_STT" = "true" ]; then \
        pip install --no-cache-dir --timeout 300 \
            --index-url https://download.pytorch.org/whl/cu128 \
            "torch>=2.7.0" "torchaudio>=2.7.0" && \
        pip install --no-cache-dir -r requirements-local-stt.txt; \
    fi

# ── Faster-whisper STT: CTranslate2 (CPU or CUDA, no torch needed) ─
# Lightweight alternative to Granite. Works on CPU-only hosts.
# Enable with: --build-arg ENABLE_FASTER_WHISPER=true
RUN if [ "$ENABLE_FASTER_WHISPER" = "true" ]; then \
        pip install --no-cache-dir --default-timeout=100 --retries 5 -r requirements-faster-whisper.txt; \
    fi

# ── Application source ─────────────────────────────────────────────
COPY pipeline/ ./pipeline/
COPY api/      ./api/
COPY ui/       ./ui/

# Pre-create runtime directories so the container starts cleanly even
# before the bind mounts are attached (plain docker run without compose).
RUN mkdir -p /data/audio /data/embeddings /models

# ── Environment defaults (all overridable at runtime) ──────────────
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
