"""
vad.py — SileroVAD ONNX wrapper.

Uses the SileroVAD ONNX model downloaded from HuggingFace on first run.
Pure onnxruntime inference — no torch dependency.

Model: onnx-community/silero-vad  (onnx/model_q4f16.onnx, ~1.15 MB)

process() is synchronous and blocking — call via asyncio.to_thread.
"""

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

_SR            = 16_000
_THRESHOLD     = 0.5     # speech probability to count as speech
_MIN_SPEECH_MS = 250     # discard segments shorter than this
_PADDING_MS    = 100     # silence kept before/after each speech segment
_CHUNK         = 1536    # samples per inference step (96 ms at 16 kHz)

_HF_REPO  = "onnx-community/silero-vad"
_HF_FILE  = "onnx/model_q4f16.onnx"


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
        self._sess       = None
        self._in_names   = []
        self._has_sr_inp = False
        model_path       = Path(model_dir) / "silero_vad" / "model_q4f16.onnx"

        try:
            if not model_path.exists():
                model_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(
                    "Downloading SileroVAD ONNX from HuggingFace (%s) …",
                    _HF_REPO,
                )
                from huggingface_hub import hf_hub_download
                tmp = hf_hub_download(
                    repo_id=_HF_REPO,
                    filename=_HF_FILE,
                )
                import shutil
                shutil.copy2(tmp, model_path)
                logger.info("SileroVAD ONNX saved to %s", model_path)

            self._sess = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )
            self._in_names   = [inp.name for inp in self._sess.get_inputs()]
            self._has_sr_inp = "sr" in self._in_names
            logger.info(
                "SileroVAD ONNX loaded — inputs: %s", self._in_names
            )
        except Exception as exc:
            logger.warning(
                "VoiceActivityDetector init failed (%s) — passthrough mode", exc
            )

    @property
    def available(self) -> bool:
        return self._sess is not None

    # ------------------------------------------------------------------

    def process(self, pcm_bytes: bytes) -> tuple[bytes, float]:
        """
        Returns (speech_pcm_bytes, vad_confidence).
        speech_pcm_bytes is empty bytes if no speech detected.
        """
        if not pcm_bytes:
            return b"", 0.0

        if self._sess is None:
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

            # Stateful LSTM: h and c carry context between chunks
            h = np.zeros((2, 1, 64), dtype=np.float32)
            c = np.zeros((2, 1, 64), dtype=np.float32)

            probs: list[float] = []

            for i in range(0, len(audio), _CHUNK):
                chunk = audio[i : i + _CHUNK].reshape(1, -1)
                inp   = self._make_inputs(chunk, h, c)
                outs  = self._sess.run(None, inp)
                probs.append(float(outs[0][0, 0]))
                h, c = outs[1], outs[2]

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

    # ------------------------------------------------------------------

    def _make_inputs(
        self,
        chunk: np.ndarray,
        h: np.ndarray,
        c: np.ndarray,
    ) -> dict:
        """Build the ONNX input dict, handling models with and without sr."""
        names = self._in_names
        if self._has_sr_inp:
            # [audio, sr, h, c]  — sr is a scalar int64
            return {
                names[0]: chunk,
                "sr":      np.array(_SR, dtype=np.int64),
                "h":       h,
                "c":       c,
            }
        else:
            # [audio, h, c]  — sample rate baked in
            return {names[0]: chunk, names[1]: h, names[2]: c}


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
