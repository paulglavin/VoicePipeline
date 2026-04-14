"""
stt.py — IBM Granite 4.0 1B Speech STT wrapper.

Model: ibm-granite/granite-4.0-1b-speech (HuggingFace)
Downloaded on first run; cached in MODEL_DIR/granite_stt/.

Uses AutoProcessor + AutoModelForSpeechSeq2Seq directly rather than the
HuggingFace pipeline API. The pipeline API passes `sampling_rate` as a
keyword argument to the feature extractor, but GraniteSpeechFeatureExtractor
does not accept that parameter — using the model directly avoids the issue.

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
        self._model     = None
        self._processor = None
        self._device    = "cpu"
        cache_dir       = str(Path(model_dir) / "granite_stt")

        try:
            import torch
            from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

            cuda          = torch.cuda.is_available()
            self._device  = "cuda" if cuda else "cpu"
            self._dtype   = torch.float16 if cuda else torch.float32

            logger.info(
                "Loading %s on %s — first run downloads ~2 GB",
                _MODEL_ID, self._device,
            )

            self._processor = AutoProcessor.from_pretrained(
                _MODEL_ID,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                _MODEL_ID,
                cache_dir=cache_dir,
                trust_remote_code=True,
                torch_dtype=self._dtype,
                low_cpu_mem_usage=True,
            ).to(self._device)
            self._model.eval()
            logger.info("Granite STT ready on %s", self._device)

        except Exception as exc:
            logger.warning(
                "SpeechToText init failed (%s) — transcription unavailable", exc
            )

    @property
    def available(self) -> bool:
        return self._model is not None and self._processor is not None

    # ------------------------------------------------------------------

    def transcribe(self, pcm_bytes: bytes) -> str:
        """
        Transcribe raw 16-bit LE PCM at 16 kHz mono.
        Returns a stripped transcript string, or "" on failure.
        """
        if not self.available or not pcm_bytes:
            return ""

        try:
            import torch

            audio = (
                np.frombuffer(pcm_bytes, dtype=np.int16)
                .astype(np.float32) / 32768.0
            )
            inputs = self._processor(audio, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                generated = self._model.generate(
                    **inputs,
                    task="transcribe",
                    language="english",
                )

            text = self._processor.decode(
                generated[0], skip_special_tokens=True
            ).strip()
            logger.debug("STT: %r", text)
            return text

        except Exception as exc:
            logger.warning("Transcription error: %s", exc)
            return ""
