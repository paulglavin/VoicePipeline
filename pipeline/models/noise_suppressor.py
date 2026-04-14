"""
noise_suppressor.py — DTLN ONNX noise suppression wrapper.

DTLN (Dual-signal Transformation LSTM Network) by Nils L. Westhausen.
Reference: https://github.com/breizhn/DTLN

Architecture: two sequential LSTM stages — frequency domain then time domain.
Processes audio in 32ms frames (512 samples at 16 kHz) with 50% overlap.
Causal, real-time capable, designed for 16 kHz speech.

Models are downloaded automatically on first run to MODEL_DIR/dtln/.
If download fails the suppressor runs in passthrough mode.

Manual setup (if auto-download fails):
    1. Clone DTLN: git clone https://github.com/breizhn/DTLN
    2. pip install tensorflow tf2onnx
    3. cd DTLN && python export_dtln_to_onnx.py
    4. Copy DTLN_model_1.onnx, DTLN_model_2.onnx to MODEL_DIR/dtln/
"""

import logging
import shutil
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_SR          = 16_000
_BLOCK_LEN   = 512    # 32 ms at 16 kHz
_BLOCK_SHIFT = 256    # 50% overlap

# HuggingFace repo that hosts the pre-exported DTLN ONNX models.
_HF_REPO  = "breizhn/DTLN"
_HF_FILES = ["DTLN_model_1.onnx", "DTLN_model_2.onnx"]


class NoiseSuppressor:
    """
    DTLN ONNX noise suppressor.

    process() is synchronous and blocking — call via asyncio.to_thread.
    """

    def __init__(self, model_dir: str) -> None:
        self._dir = Path(model_dir) / "dtln"
        self._dir.mkdir(parents=True, exist_ok=True)

        self._sess1 = None
        self._sess2 = None
        self._in1:  list[str] = []
        self._out1: list[str] = []
        self._in2:  list[str] = []
        self._out2: list[str] = []

        p1 = self._dir / "DTLN_model_1.onnx"
        p2 = self._dir / "DTLN_model_2.onnx"

        if not p1.exists() or not p2.exists():
            self._download_models(p1, p2)

        if p1.exists() and p2.exists():
            self._load_sessions(p1, p2)
        else:
            logger.warning(
                "DTLN: ONNX model files not found at %s — running in passthrough mode. "
                "See module docstring for manual setup instructions.",
                self._dir,
            )

    # ------------------------------------------------------------------
    # Setup helpers

    def _download_models(self, p1: Path, p2: Path) -> None:
        """Try to fetch DTLN ONNX files from HuggingFace Hub."""
        try:
            from huggingface_hub import hf_hub_download
            logger.info("DTLN: downloading models from HuggingFace (%s) ...", _HF_REPO)
            for fname, dest in zip(_HF_FILES, (p1, p2)):
                cached = hf_hub_download(repo_id=_HF_REPO, filename=fname)
                shutil.copy(cached, dest)
            logger.info("DTLN: models downloaded to %s", self._dir)
        except Exception as exc:
            logger.warning(
                "DTLN: auto-download failed (%s). "
                "Place DTLN_model_1.onnx and DTLN_model_2.onnx in %s to enable noise suppression.",
                exc,
                self._dir,
            )

    def _load_sessions(self, p1: Path, p2: Path) -> None:
        """Load both ONNX sessions and cache input/output names."""
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 2

            self._sess1 = ort.InferenceSession(str(p1), opts, providers=["CPUExecutionProvider"])
            self._sess2 = ort.InferenceSession(str(p2), opts, providers=["CPUExecutionProvider"])

            # Read names dynamically — robust across different DTLN export versions
            self._in1  = [i.name for i in self._sess1.get_inputs()]
            self._out1 = [o.name for o in self._sess1.get_outputs()]
            self._in2  = [i.name for i in self._sess2.get_inputs()]
            self._out2 = [o.name for o in self._sess2.get_outputs()]

            logger.info(
                "DTLN: loaded (model_1 inputs=%s, model_2 inputs=%s)",
                self._in1, self._in2,
            )
        except Exception as exc:
            logger.warning("DTLN: failed to load ONNX sessions (%s) — passthrough mode", exc)
            self._sess1 = None
            self._sess2 = None

    # ------------------------------------------------------------------
    # Properties

    @property
    def available(self) -> bool:
        return self._sess1 is not None and self._sess2 is not None

    # ------------------------------------------------------------------
    # Inference

    def process(self, pcm_bytes: bytes) -> bytes:
        """
        Denoise 16-bit LE PCM at 16 kHz mono.
        Returns the same format. Falls back to the input on any error.
        """
        if not self.available or len(pcm_bytes) < _BLOCK_LEN * 2:
            return pcm_bytes

        try:
            audio = (
                np.frombuffer(pcm_bytes, dtype=np.int16)
                .astype(np.float32) / 32768.0
            )
            return self._run_dtln(audio)
        except Exception as exc:
            logger.warning("DTLN inference error (%s) — returning original audio", exc)
            return pcm_bytes

    def _run_dtln(self, audio: np.ndarray) -> bytes:
        n = len(audio)

        # Pad so every input sample is covered by at least one complete block.
        # We need the padded length to be: (num_blocks - 1) * shift + block_len
        # where num_blocks * shift >= n  →  num_blocks = ceil(n / shift)
        num_blocks = max(1, -(-n // _BLOCK_SHIFT))          # ceil division
        padded_len = (num_blocks - 1) * _BLOCK_SHIFT + _BLOCK_LEN
        audio_padded = np.pad(audio, (0, max(0, padded_len - n)))

        # Initialise LSTM states (zero tensors matching each model's expected shape)
        h1 = np.zeros(self._sess1.get_inputs()[1].shape, dtype=np.float32)
        h2 = np.zeros(self._sess2.get_inputs()[1].shape, dtype=np.float32)

        out_buffer = np.zeros(_BLOCK_LEN, dtype=np.float32)
        output     = np.zeros(padded_len, dtype=np.float32)

        for i in range(num_blocks):
            start     = i * _BLOCK_SHIFT
            in_block  = audio_padded[start : start + _BLOCK_LEN]

            # ── Stage 1: frequency-domain masking ────────────────────────
            spec  = np.fft.rfft(in_block)
            mag   = np.abs(spec).reshape(1, 1, -1).astype(np.float32)
            phase = np.angle(spec)

            out1    = self._sess1.run(self._out1, {self._in1[0]: mag, self._in1[1]: h1})
            mask1, h1 = out1[0], out1[1]

            est_spec  = (mag * mask1).squeeze() * np.exp(1j * phase)
            est_frame = np.fft.irfft(est_spec).reshape(1, 1, -1).astype(np.float32)

            # ── Stage 2: time-domain masking ─────────────────────────────
            out2      = self._sess2.run(self._out2, {self._in2[0]: est_frame, self._in2[1]: h2})
            mask2, h2 = out2[0], out2[1]

            out_block = (est_frame * mask2)[0, 0]   # (512,)

            # ── 50% overlap-add synthesis ────────────────────────────────
            out_buffer[:_BLOCK_SHIFT] += out_block[:_BLOCK_SHIFT]
            output[start : start + _BLOCK_SHIFT] = out_buffer[:_BLOCK_SHIFT]
            out_buffer[:-_BLOCK_SHIFT]  = out_buffer[_BLOCK_SHIFT:]
            out_buffer[-_BLOCK_SHIFT:]  = 0.0
            out_buffer += np.concatenate([
                np.zeros(_BLOCK_SHIFT, dtype=np.float32),
                out_block[_BLOCK_SHIFT:],
            ])

        # Trim to original length and convert back to 16-bit PCM
        clean  = output[:n]
        out_i16 = (np.clip(clean, -1.0, 1.0) * 32767.0).astype(np.int16)
        return out_i16.tobytes()
