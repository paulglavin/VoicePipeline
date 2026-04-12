"""
orchestrator.py — wires the four-model pipeline together.

GTCRN → SileroVAD → ERes2Net → Granite STT → InteractionWriter

Each model stage runs in asyncio.to_thread so the event loop stays clean.
Failures in individual stages degrade gracefully:
  - Noise suppressor failure  → pass raw audio downstream
  - VAD failure               → pass full audio downstream (conf = 0.5)
  - VAD silent result         → discard (return None)
  - Speaker ID failure        → matched_speaker = None
  - STT failure / empty text  → discard (return None)

InteractionWriter.write() is fire-and-forget (asyncio.create_task).
"""

import asyncio
import logging
from dataclasses import dataclass

import numpy as np

from pipeline.models import (
    NoiseSuppressor,
    VoiceActivityDetector,
    SpeakerIdentifier,
    SpeechToText,
)
from pipeline.writer import InteractionWriter

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16_000


@dataclass
class PipelineResult:
    transcript:      str
    duration_ms:     int
    vad_confidence:  float
    matched_speaker: str | None
    match_confidence: float | None


class VoicePipelineOrchestrator:
    def __init__(
        self,
        noise_suppressor:  NoiseSuppressor,
        vad:               VoiceActivityDetector,
        speaker_id:        SpeakerIdentifier,
        stt:               SpeechToText,
        writer:            InteractionWriter,
    ) -> None:
        self._ns     = noise_suppressor
        self._vad    = vad
        self._sid    = speaker_id
        self._stt    = stt
        self._writer = writer

    async def process(self, raw_pcm: bytes) -> PipelineResult | None:
        """
        Run the full pipeline on a single utterance.

        raw_pcm   — 16-bit LE PCM, 16 kHz, mono (from Wyoming AudioChunk stream)

        Returns PipelineResult on success, None if the utterance should be
        discarded (silence, empty transcript, or upstream error).
        """
        if not raw_pcm:
            return None

        # ── 1. Noise suppression ────────────────────────────────────────
        try:
            clean_pcm = await asyncio.to_thread(self._ns.process, raw_pcm)
        except Exception as exc:
            logger.warning("Noise suppression error: %s", exc)
            clean_pcm = raw_pcm

        # ── 2. Voice activity detection ─────────────────────────────────
        try:
            speech_pcm, vad_conf = await asyncio.to_thread(self._vad.process, clean_pcm)
        except Exception as exc:
            logger.warning("VAD error: %s", exc)
            speech_pcm, vad_conf = clean_pcm, 0.5

        if not speech_pcm:
            logger.debug("Orchestrator: VAD found no speech — discarding utterance")
            return None

        # ── 3. Speaker identification ───────────────────────────────────
        try:
            speaker, match_conf, embedding = await asyncio.to_thread(
                self._sid.identify, speech_pcm
            )
        except Exception as exc:
            logger.warning("Speaker ID error: %s", exc)
            speaker, match_conf, embedding = None, None, None

        # ── 4. Speech-to-text ───────────────────────────────────────────
        try:
            transcript = await asyncio.to_thread(self._stt.transcribe, speech_pcm)
        except Exception as exc:
            logger.warning("STT error: %s", exc)
            transcript = ""

        if not transcript.strip():
            logger.debug("Orchestrator: empty transcript — discarding utterance")
            return None

        # ── 5. Compute duration ─────────────────────────────────────────
        num_samples  = len(speech_pcm) // 2          # 16-bit = 2 bytes/sample
        duration_ms  = max(1, int(num_samples / _SAMPLE_RATE * 1000))

        # ── 6. Write (fire-and-forget) ──────────────────────────────────
        asyncio.create_task(
            self._writer.write(
                audio_bytes=speech_pcm,
                transcript=transcript,
                duration_ms=duration_ms,
                vad_confidence=vad_conf,
                matched_speaker=speaker,
                match_confidence=match_conf,
                embedding=np.array(embedding, dtype=np.float32)
                          if embedding is not None else None,
            )
        )

        result = PipelineResult(
            transcript=transcript,
            duration_ms=duration_ms,
            vad_confidence=vad_conf,
            matched_speaker=speaker,
            match_confidence=match_conf,
        )
        logger.info(
            "Pipeline: %r | speaker=%s conf=%.2f | dur=%dms",
            transcript,
            speaker or "unknown",
            match_conf or 0.0,
            duration_ms,
        )
        return result
