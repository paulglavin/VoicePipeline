"""
stt.py — IBM Granite 4.0 1B Speech STT wrapper.

Model: ibm-granite/granite-4.0-1b-speech (HuggingFace)
Downloaded on first run; cached in MODEL_DIR/granite_stt/.

The model follows the Whisper encoder + Granite decoder architecture.
We use the HuggingFace `pipeline` API so the correct model class is
selected automatically, even if the implementation uses trust_remote_code.

transcribe() is synchronous and blocking — call via asyncio.to_thread.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_SR       = 16_000
_MODEL_ID = "ibm-granite/granite-4.0-1b-speech"


class SpeechToText:
    """
    Transcribes 16-bit LE PCM audio to text using Granite 4.0 1B Speech.

    Falls back to an empty string on any error — the orchestrator treats
    an empty transcript as "no utterance" and discards the interaction.
    """

    def __init__(self, model_dir: str) -> None:
        self._pipe   = None
        self._device = "cpu"
        cache_dir    = str(Path(model_dir) / "granite_stt")

        try:
            import torch
            from transformers import pipeline as hf_pipeline

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype        = torch.float16 if self._device == "cuda" else torch.float32

            logger.info(
                "Loading %s on %s — first run downloads ~2 GB", _MODEL_ID, self._device
            )

            self._pipe = hf_pipeline(
                "automatic-speech-recognition",
                model=_MODEL_ID,
                device=self._device,
                torch_dtype=dtype,
                model_kwargs={
                    "cache_dir":        cache_dir,
                    "low_cpu_mem_usage": True,
                    "trust_remote_code": True,
                },
            )
            logger.info("Granite STT ready on %s", self._device)

        except Exception as exc:
            logger.warning(
                "SpeechToText init failed (%s) — transcription unavailable", exc
            )

    @property
    def available(self) -> bool:
        return self._pipe is not None

    # ------------------------------------------------------------------

    def transcribe(self, pcm_bytes: bytes) -> str:
        """
        Transcribe raw 16-bit LE PCM at 16 kHz mono.
        Returns a stripped transcript string, or "" on failure.
        """
        if self._pipe is None or not pcm_bytes:
            return ""

        try:
            audio = (
                np.frombuffer(pcm_bytes, dtype=np.int16)
                .astype(np.float32) / 32768.0
            )
            result = self._pipe(
                {"array": audio, "sampling_rate": _SR},
                generate_kwargs={"task": "transcribe", "language": "english"},
            )
            text = result.get("text", "").strip()
            logger.debug("STT: %r", text)
            return text

        except Exception as exc:
            logger.warning("Transcription error: %s", exc)
            return ""
