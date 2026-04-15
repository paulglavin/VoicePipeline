"""
vad.py — SileroVAD ONNX wrapper.

Uses the silero-vad package's own OnnxWrapper for inference — state management
is handled by the package so we never have to match its internal tensor shapes.

The ONNX file is bundled inside the silero-vad pip package and copied to
MODEL_DIR on first run so it lives alongside the other cached models.

process() is synchronous and blocking — call via asyncio.to_thread.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_SR            = 16_000
_THRESHOLD     = 0.5     # speech probability to count as speech
_MIN_SPEECH_MS = 250     # discard segments shorter than this
_PADDING_MS    = 100     # silence kept before/after each speech segment
_CHUNK         = 512     # samples per inference step (32 ms at 16 kHz — package default)

_PKG_NAME = "silero_vad"
_PKG_ONNX = "data/silero_vad.onnx"


class VoiceActivityDetector:
    """
    Detects speech segments in a PCM buffer using SileroVAD ONNX.

    Returns (speech_bytes, confidence) where:
    - speech_bytes  — concatenated speech segments (16-bit LE PCM, 16 kHz mono)
    - confidence    — fraction of audio classified as speech (0.0–1.0)

    On failure: returns the original bytes with confidence 0.5 (passthrough).
    If no speech is found: returns (b"", 0.0).
    """

    def __init__(self, model_dir: str) -> None:
        self._model    = None
        model_path     = Path(model_dir) / "silero_vad" / "silero_vad.onnx"

        try:
            # Remove stale files from previous download attempts
            for stale_name in ("model_q4f16.onnx",):
                stale = model_path.parent / stale_name
                if stale.exists():
                    stale.unlink()
                    logger.info("Removed stale VAD model: %s", stale)

            if not model_path.exists():
                model_path.parent.mkdir(parents=True, exist_ok=True)
                import importlib.util, shutil
                spec = importlib.util.find_spec(_PKG_NAME)
                if spec is None or spec.origin is None:
                    raise ImportError(
                        "silero-vad package is not installed — add it to requirements.txt"
                    )
                pkg_onnx = Path(spec.origin).parent / _PKG_ONNX
                if not pkg_onnx.exists():
                    raise FileNotFoundError(
                        f"ONNX not found inside silero-vad package at {pkg_onnx}"
                    )
                shutil.copy2(pkg_onnx, model_path)
                logger.info("SileroVAD ONNX copied from package → %s", model_path)

            from silero_vad.model import OnnxWrapper
            self._model = OnnxWrapper(str(model_path))
            logger.info("SileroVAD loaded via OnnxWrapper")

        except Exception as exc:
            logger.warning(
                "VoiceActivityDetector init failed (%s) — passthrough mode", exc
            )

    @property
    def available(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------

    def process(self, pcm_bytes: bytes) -> tuple[bytes, float]:
        """
        Returns (speech_pcm_bytes, vad_confidence).
        speech_pcm_bytes is empty bytes if no speech detected.
        """
        if not pcm_bytes:
            return b"", 0.0

        if self._model is None:
            return pcm_bytes, 1.0

        try:
            audio = (
                np.frombuffer(pcm_bytes, dtype=np.int16)
                .astype(np.float32) / 32768.0
            )
            n = len(audio)

            # Pad to a multiple of _CHUNK so every chunk is full-length
            remainder = n % _CHUNK
            if remainder:
                audio = np.concatenate(
                    [audio, np.zeros(_CHUNK - remainder, dtype=np.float32)]
                )

            # OnnxWrapper manages its own LSTM state — reset for each utterance
            self._model.reset_states()
            probs: list[float] = []

            for i in range(0, len(audio), _CHUNK):
                chunk = audio[i : i + _CHUNK][np.newaxis, :]   # (1, chunk)
                prob  = self._model(chunk, sr=_SR)
                probs.append(float(prob))

            # Collect speech segments from per-chunk probabilities
            segments = _find_segments(
                probs, _CHUNK, n, _THRESHOLD,
                _MIN_SPEECH_MS, _PADDING_MS, _SR,
            )

            if not segments:
                logger.debug("VAD: no speech detected")
                return b"", 0.0

            # Trim audio back to original length before slicing
            audio = audio[:n]

            parts: list[np.ndarray] = []
            speech_samples = 0
            for start, end in segments:
                parts.append(audio[start:end])
                speech_samples += end - start

            speech   = np.concatenate(parts)
            conf     = min(float(speech_samples) / max(n, 1), 1.0)
            out_i16  = (speech * 32767.0).clip(-32768, 32767).astype(np.int16)

            logger.debug("VAD: %d segment(s), confidence=%.2f", len(segments), conf)
            return out_i16.tobytes(), conf

        except Exception as exc:
            logger.warning("VAD error (%s) — passthrough", exc)
            return pcm_bytes, 0.5


# ---------------------------------------------------------------------------
# Helpers


def _find_segments(
    probs: list[float],
    chunk_size: int,
    n_original: int,
    threshold: float,
    min_speech_ms: int,
    padding_ms: int,
    sr: int,
) -> list[tuple[int, int]]:
    """
    Convert per-chunk speech probabilities to (start, end) sample pairs.

    Applies minimum speech duration and padding, and clamps to [0, n_original].
    """
    min_samples = int(min_speech_ms / 1000.0 * sr)
    pad_samples = int(padding_ms    / 1000.0 * sr)

    segments: list[tuple[int, int]] = []
    in_speech = False
    seg_start = 0

    for i, prob in enumerate(probs):
        sample = i * chunk_size
        if prob >= threshold and not in_speech:
            in_speech = True
            seg_start = sample
        elif prob < threshold and in_speech:
            seg_end = sample
            if seg_end - seg_start >= min_samples:
                start = max(0, seg_start - pad_samples)
                end   = min(n_original, seg_end + pad_samples)
                segments.append((start, end))
            in_speech = False

    # Trailing speech that never dips below threshold
    if in_speech:
        seg_end = len(probs) * chunk_size
        if seg_end - seg_start >= min_samples:
            start = max(0, seg_start - pad_samples)
            end   = min(n_original, seg_end + pad_samples)
            segments.append((start, end))

    return segments
