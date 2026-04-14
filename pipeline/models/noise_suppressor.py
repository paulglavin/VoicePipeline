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
    Download model_1.onnx and model_2.onnx from:
    https://github.com/breizhn/DTLN/tree/master/pretrained_model
    and place them in MODEL_DIR/dtln/
"""

import logging
import os
import urllib.request
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_SR          = 16_000
_BLOCK_LEN   = 512    # 32 ms at 16 kHz
_BLOCK_SHIFT = 256    # 50% overlap

# Wet/dry blend ratio — how much of the DTLN output to mix with the original.
#   1.0 = full DTLN (maximum suppression, most artefacts)
#   0.0 = passthrough (no suppression, no artefacts)
#   0.5 = balanced default — real-world noise reduction with minimal musical noise
# Override with the DTLN_MIX environment variable.
_DEFAULT_MIX = 0.5
_MIX = float(os.getenv("DTLN_MIX", str(_DEFAULT_MIX)))

# Official DTLN pretrained ONNX weights — breizhn/DTLN on GitHub
_GITHUB_BASE = (
    "https://raw.githubusercontent.com/breizhn/DTLN/master/pretrained_model/"
)
_ONNX_FILES = ["model_1.onnx", "model_2.onnx"]


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

        p1 = self._dir / "model_1.onnx"
        p2 = self._dir / "model_2.onnx"

        if not p1.exists() or not p2.exists():
            self._download_models(p1, p2)

        if p1.exists() and p2.exists():
            self._load_sessions(p1, p2)
        else:
            logger.warning(
                "DTLN: ONNX files not found at %s — passthrough mode. "
                "Place model_1.onnx and model_2.onnx from "
                "https://github.com/breizhn/DTLN/tree/master/pretrained_model "
                "in that directory to enable noise suppression.",
                self._dir,
            )

    # ------------------------------------------------------------------
    # Setup

    def _download_models(self, p1: Path, p2: Path) -> None:
        try:
            logger.info("DTLN: downloading pretrained models from GitHub ...")
            for fname, dest in zip(_ONNX_FILES, (p1, p2)):
                url = _GITHUB_BASE + fname
                logger.info("DTLN: fetching %s", url)
                urllib.request.urlretrieve(url, dest)
            logger.info("DTLN: models saved to %s", self._dir)
        except Exception as exc:
            for p in (p1, p2):
                p.unlink(missing_ok=True)
            logger.warning(
                "DTLN: auto-download failed (%s). "
                "Manually place model_1.onnx and model_2.onnx in %s.",
                exc, self._dir,
            )

    def _load_sessions(self, p1: Path, p2: Path) -> None:
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 2
            providers = ["CPUExecutionProvider"]

            self._sess1 = ort.InferenceSession(str(p1), opts, providers=providers)
            self._sess2 = ort.InferenceSession(str(p2), opts, providers=providers)

            # Log model signatures so mismatches are easy to diagnose
            for name, sess in (("model_1", self._sess1), ("model_2", self._sess2)):
                inputs  = [(i.name, i.shape) for i in sess.get_inputs()]
                outputs = [(o.name, o.shape) for o in sess.get_outputs()]
                logger.info("DTLN %s  inputs=%s  outputs=%s", name, inputs, outputs)
            logger.info("DTLN ready — wet/dry mix=%.2f (set DTLN_MIX env var to adjust)", _MIX)

        except Exception as exc:
            logger.warning("DTLN: failed to load ONNX sessions (%s) — passthrough mode", exc)
            self._sess1 = None
            self._sess2 = None

    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._sess1 is not None and self._sess2 is not None

    # ------------------------------------------------------------------
    # Inference

    def process(self, pcm_bytes: bytes) -> bytes:
        """
        Denoise 16-bit LE PCM at 16 kHz mono.
        Returns the same format. Falls back to input on any error.
        """
        if not self.available or len(pcm_bytes) < _BLOCK_LEN * 2:
            return pcm_bytes

        try:
            audio = (
                np.frombuffer(pcm_bytes, dtype=np.int16)
                .astype(np.float32) / 32768.0
            )
            enhanced = self._run_dtln(audio)

            # Safety: if DTLN produces NaN/Inf return the original
            if not np.all(np.isfinite(enhanced)):
                logger.warning("DTLN: non-finite output — returning original audio")
                return pcm_bytes

            # Wet/dry blend — reduces musical-noise artefacts while preserving
            # the noise-suppression benefit.  _MIX=1.0 → full DTLN output;
            # _MIX=0.0 → passthrough.  Tune with DTLN_MIX env var.
            mixed = _MIX * enhanced + (1.0 - _MIX) * audio

            out_i16 = (np.clip(mixed, -1.0, 1.0) * 32767.0).astype(np.int16)
            return out_i16.tobytes()

        except Exception as exc:
            logger.warning("DTLN inference error (%s) — returning original audio", exc)
            return pcm_bytes

    def _make_input_dict(self, sess) -> dict:
        """
        Build a zero-filled input dict for an ONNX session.
        Matches the reference DTLN implementation: None dims are replaced by 1.
        """
        return {
            inp.name: np.zeros(
                [d if isinstance(d, int) else 1 for d in inp.shape],
                dtype=np.float32,
            )
            for inp in sess.get_inputs()
        }

    def _run_dtln(self, audio: np.ndarray) -> np.ndarray:
        """Process a full utterance through both DTLN LSTM stages."""
        n = len(audio)

        # Pad so every sample is covered by at least one complete block
        num_blocks = max(1, -(-n // _BLOCK_SHIFT))            # ceil(n / shift)
        padded_len = (num_blocks - 1) * _BLOCK_SHIFT + _BLOCK_LEN
        audio_pad  = np.pad(audio, (0, max(0, padded_len - n)))

        # Fresh state dicts for this utterance (zero initial state)
        inputs1 = self._make_input_dict(self._sess1)
        inputs2 = self._make_input_dict(self._sess2)

        # Input/output names — cached from loaded sessions
        in_names1  = [i.name for i in self._sess1.get_inputs()]
        in_names2  = [i.name for i in self._sess2.get_inputs()]

        out_buffer = np.zeros(_BLOCK_LEN, dtype=np.float32)
        output     = np.zeros(padded_len, dtype=np.float32)

        for i in range(num_blocks):
            start    = i * _BLOCK_SHIFT
            in_block = audio_pad[start : start + _BLOCK_LEN]

            # ── Stage 1: frequency-domain masking ────────────────────────
            spec  = np.fft.rfft(in_block)                         # (257,) complex
            mag   = np.abs(spec).reshape(1, 1, -1).astype(np.float32)  # (1,1,257)
            phase = np.angle(spec)                                 # (257,)

            inputs1[in_names1[0]] = mag
            out1 = self._sess1.run(None, inputs1)
            mask1 = out1[0]                                        # (1,1,257)
            # Update all state tensors (inputs[1:] ← outputs[1:])
            for j, name in enumerate(in_names1[1:], start=1):
                inputs1[name] = out1[j]

            est_spec  = (mag * mask1).squeeze() * np.exp(1j * phase)  # (257,)
            est_frame = np.fft.irfft(est_spec).reshape(1, 1, -1).astype(np.float32)  # (1,1,512)

            # ── Stage 2: time-domain enhancement ─────────────────────────
            # model_2 outputs the enhanced signal directly (mask applied
            # internally) — do NOT multiply est_frame by it again.
            inputs2[in_names2[0]] = est_frame
            out2 = self._sess2.run(None, inputs2)
            out_block = out2[0][0, 0]                              # (512,) enhanced signal
            for j, name in enumerate(in_names2[1:], start=1):
                inputs2[name] = out2[j]

            # ── 50% overlap-add synthesis (matches DTLN reference) ───────
            out_buffer[:_BLOCK_SHIFT] += out_block[:_BLOCK_SHIFT]
            output[start : start + _BLOCK_SHIFT] = out_buffer[:_BLOCK_SHIFT]
            out_buffer[:-_BLOCK_SHIFT]  = out_buffer[_BLOCK_SHIFT:]
            out_buffer[-_BLOCK_SHIFT:]  = 0.0
            out_buffer += np.concatenate([
                np.zeros(_BLOCK_SHIFT, dtype=np.float32),
                out_block[_BLOCK_SHIFT:],
            ])

        return output[:n]
