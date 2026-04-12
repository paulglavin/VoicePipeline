"""
vad.py — SileroVAD wrapper.

Uses the silero-vad PyPI package (v5+).
Weights are cached in MODEL_DIR/silero_vad/ via torch.hub.

process() is synchronous and blocking — call via asyncio.to_thread.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_SR             = 16_000
_THRESHOLD      = 0.5    # speech probability to count as speech
_MIN_SPEECH_MS  = 250    # discard segments shorter than this
_PADDING_MS     = 100    # silence kept before/after each speech segment


class VoiceActivityDetector:
    """
    Detects speech segments in a PCM buffer.

    Returns (speech_bytes, confidence) where:
    - speech_bytes  — concatenated speech segments (16-bit LE PCM, 16 kHz mono)
    - confidence    — fraction of audio classified as speech (0.0–1.0)

    On failure: returns the original bytes with confidence 0.5 (passthrough).
    If no speech is found: returns (b"", 0.0).
    """

    def __init__(self, model_dir: str) -> None:
        self._model    = None
        self._get_ts   = None
        self._hub_dir  = Path(model_dir) / "silero_vad"

        try:
            import torch
            torch.hub.set_dir(str(self._hub_dir))
            from silero_vad import load_silero_vad, get_speech_timestamps

            self._model  = load_silero_vad().to("cpu")
            self._get_ts = get_speech_timestamps
            logger.info("SileroVAD loaded on cpu (hub cache: %s)", self._hub_dir)
        except Exception as exc:
            logger.warning("VoiceActivityDetector init failed (%s) — passthrough mode", exc)

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
            # Passthrough — treat entire clip as speech
            return pcm_bytes, 1.0

        try:
            import torch

            audio = (
                np.frombuffer(pcm_bytes, dtype=np.int16)
                .astype(np.float32) / 32768.0
            )
            tensor = torch.from_numpy(audio).to("cpu")

            timestamps = self._get_ts(
                tensor,
                self._model,
                threshold=_THRESHOLD,
                sampling_rate=_SR,
                min_speech_duration_ms=_MIN_SPEECH_MS,
                min_silence_duration_ms=_PADDING_MS,
                return_seconds=False,
            )

            if not timestamps:
                logger.debug("VAD: no speech detected")
                return b"", 0.0

            pad_samples = int(_PADDING_MS / 1000.0 * _SR)
            n           = len(audio)
            segments: list[np.ndarray] = []
            speech_samples = 0

            for seg in timestamps:
                start = max(0, seg["start"] - pad_samples)
                end   = min(n, seg["end"]   + pad_samples)
                segments.append(audio[start:end])
                speech_samples += seg["end"] - seg["start"]

            speech  = np.concatenate(segments)
            conf    = min(float(speech_samples) / max(n, 1), 1.0)
            out_i16 = (speech * 32767.0).clip(-32768, 32767).astype(np.int16)

            logger.debug("VAD: %d segments, confidence=%.2f", len(timestamps), conf)
            return out_i16.tobytes(), conf

        except Exception as exc:
            logger.warning("VAD error (%s) — passthrough", exc)
            return pcm_bytes, 0.5
