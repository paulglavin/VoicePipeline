"""
stt.py — IBM Granite 4.0 1B Speech STT wrapper.

Model: ibm-granite/granite-4.0-1b-speech (HuggingFace)
Downloaded on first run; cached in MODEL_DIR/granite_stt/.

Granite 4.0 Speech is a multimodal LLM — the processor requires a chat-
templated text prompt alongside the audio tensor. The <|audio|> token in
the prompt is where audio features are injected. Output decoding strips the
input tokens so we only decode the newly generated text.

transcribe() is synchronous and blocking — call via asyncio.to_thread.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_SR       = 16_000
_MODEL_ID = "ibm-granite/granite-4.0-1b-speech"
_PROMPT   = "<|audio|>can you transcribe the speech into a written format?"


class SpeechToText:
    """
    Transcribes 16-bit LE PCM audio to text using Granite 4.0 1B Speech.

    Falls back to an empty string on any error — the orchestrator treats
    an empty transcript as "no utterance" and discards the interaction.
    """

    def __init__(self, model_dir: str) -> None:
        self._model     = None
        self._processor = None
        self._tokenizer = None
        self._device    = "cpu"
        self._prompt    = None
        cache_dir       = str(Path(model_dir) / "granite_stt")

        try:
            import torch
            from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

            cuda         = torch.cuda.is_available()
            self._device = "cuda" if cuda else "cpu"

            logger.info(
                "Loading %s on %s — first run downloads ~2 GB",
                _MODEL_ID, self._device,
            )

            self._processor = AutoProcessor.from_pretrained(
                _MODEL_ID,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
            self._tokenizer = self._processor.tokenizer
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                _MODEL_ID,
                cache_dir=cache_dir,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                device_map=self._device,
            )
            self._model.eval()

            # Pre-build the prompt once — apply_chat_template is pure text processing
            chat = [{"role": "user", "content": _PROMPT}]
            self._prompt = self._tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )

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

            # Convert PCM bytes → float32 tensor, shape [1, samples] (mono)
            wav = torch.from_numpy(
                np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            ).unsqueeze(0)

            inputs = self._processor(
                self._prompt,
                wav,
                device=self._device,
                return_tensors="pt",
            ).to(self._device)

            num_input_tokens = inputs["input_ids"].shape[-1]

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    num_beams=1,
                )

            new_tokens = outputs[0, num_input_tokens:].unsqueeze(0)
            text = self._tokenizer.batch_decode(
                new_tokens,
                add_special_tokens=False,
                skip_special_tokens=True,
            )[0].strip()

            logger.debug("STT: %r", text)
            return text

        except Exception as exc:
            logger.warning("Transcription error: %s", exc)
            return ""
