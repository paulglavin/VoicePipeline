"""
stt.py — Speech-to-text: local HuggingFace or remote OpenAI-compatible.

Local provider: loads the configured HuggingFace model on-device.
Remote provider: calls /v1/audio/transcriptions on an OpenAI-compatible endpoint,
                 e.g. Ollama running on a separate GPU machine.

transcribe() is synchronous and blocking — call via asyncio.to_thread.
"""

import io
import logging
import wave
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_SR                  = 16_000
_LOCAL_MODEL_DEFAULT = "ibm-granite/granite-4.0-1b-speech"
_PROMPT_DEFAULT      = "<|audio|>can you transcribe the speech into a written format?"


class SpeechToText:
    """
    Speech-to-text wrapper. Delegates to _LocalSTT or _RemoteSTT based on provider.

    Falls back to an empty string on any error — the orchestrator treats
    an empty transcript as "no utterance" and discards the interaction.
    """

    def __init__(
        self,
        model_dir: str,
        provider: str = "local",
        model_id: str = _LOCAL_MODEL_DEFAULT,
        base_url: str = "",
        api_key: str = "",
        prompt: str = _PROMPT_DEFAULT,
    ) -> None:
        if provider == "openai_compatible" and base_url:
            self._impl = _RemoteSTT(model_id, base_url, api_key, prompt)
        else:
            self._impl = _LocalSTT(model_dir, model_id, prompt)

    @property
    def available(self) -> bool:
        return self._impl.available

    def transcribe(self, pcm_bytes: bytes) -> str:
        return self._impl.transcribe(pcm_bytes)


class _LocalSTT:
    """Local HuggingFace STT. Model ID is configurable — defaults to Granite 4.0 1B Speech."""

    def __init__(self, model_dir: str, model_id: str, prompt: str) -> None:
        self._model     = None
        self._processor = None
        self._tokenizer = None
        self._device    = "cpu"
        self._prompt_t  = None
        cache_dir       = str(Path(model_dir) / "granite_stt")

        try:
            import torch
            from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

            cuda         = torch.cuda.is_available()
            self._device = "cuda" if cuda else "cpu"

            logger.info(
                "Loading %s on %s — first run downloads model weights",
                model_id, self._device,
            )

            self._processor = AutoProcessor.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
            self._tokenizer = self._processor.tokenizer
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                device_map=self._device,
            )
            self._model.eval()

            chat = [{"role": "user", "content": prompt}]
            self._prompt_t = self._tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )

            logger.info("Local STT ready: %s on %s", model_id, self._device)

        except Exception as exc:
            logger.warning("Local STT init failed (%s) — transcription unavailable", exc)

    @property
    def available(self) -> bool:
        return self._model is not None and self._processor is not None

    def transcribe(self, pcm_bytes: bytes) -> str:
        if not self.available or not pcm_bytes:
            return ""

        try:
            import torch

            wav = torch.from_numpy(
                np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            ).unsqueeze(0)

            inputs = self._processor(
                self._prompt_t,
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


class _RemoteSTT:
    """Remote STT via OpenAI-compatible /v1/audio/transcriptions endpoint."""

    def __init__(self, model_id: str, base_url: str, api_key: str, prompt: str) -> None:
        self._model_id = model_id
        self._base_url = base_url.rstrip("/")
        self._api_key  = api_key
        # Granite-specific tokens aren't meaningful to a remote model
        self._prompt   = prompt if not prompt.startswith("<|") else ""
        self._available = False

        try:
            import httpx  # noqa: F401
            self._available = True
            logger.info("Remote STT configured: %s at %s", model_id, self._base_url)
        except ImportError:
            logger.warning("Remote STT requires httpx — pip install httpx")

    @property
    def available(self) -> bool:
        return self._available

    def transcribe(self, pcm_bytes: bytes) -> str:
        if not self._available or not pcm_bytes:
            return ""

        try:
            import httpx

            wav_buf = io.BytesIO()
            with wave.open(wav_buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(_SR)
                wf.writeframes(pcm_bytes)
            wav_buf.seek(0)

            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            data = {"model": self._model_id}
            if self._prompt:
                data["prompt"] = self._prompt

            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"{self._base_url}/audio/transcriptions",
                    headers=headers,
                    data=data,
                    files={"file": ("audio.wav", wav_buf, "audio/wav")},
                )
                response.raise_for_status()

            text = response.json().get("text", "").strip()
            logger.debug("Remote STT: %r", text)
            return text

        except Exception as exc:
            logger.warning("Remote STT error: %s", exc)
            return ""
