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
import time
import uuid
from dataclasses import dataclass, field

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
    interaction_id:   str
    transcript:       str
    duration_ms:      int
    vad_confidence:   float
    matched_speaker:  str | None
    match_confidence: float | None
    timings:          dict = field(default_factory=dict)


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

        t_start = time.monotonic()

        # ── 1. Noise suppression ────────────────────────────────────────
        t0 = time.monotonic()
        try:
            clean_pcm = await asyncio.to_thread(self._ns.process, raw_pcm)
        except Exception as exc:
            logger.warning("Noise suppression error: %s", exc)
            clean_pcm = raw_pcm
        ns_ms = round((time.monotonic() - t0) * 1000)
        logger.info(
            "NS: active=%s  in=%d bytes  out=%d bytes  %dms",
            self._ns.available, len(raw_pcm), len(clean_pcm), ns_ms,
        )

        # ── 2. Voice activity detection ─────────────────────────────────
        t0 = time.monotonic()
        try:
            speech_pcm, vad_conf = await asyncio.to_thread(self._vad.process, clean_pcm)
        except Exception as exc:
            logger.warning("VAD error: %s", exc)
            speech_pcm, vad_conf = clean_pcm, 0.5
        vad_ms = round((time.monotonic() - t0) * 1000)
        logger.info(
            "VAD: speech=%d bytes  conf=%.3f  %dms",
            len(speech_pcm) if speech_pcm else 0, vad_conf, vad_ms,
        )

        if not speech_pcm:
            logger.info("Orchestrator: VAD found no speech — discarding utterance")
            return None

        # ── 3. Speaker identification ───────────────────────────────────
        t0 = time.monotonic()
        try:
            speaker, match_conf, embedding = await asyncio.to_thread(
                self._sid.identify, speech_pcm
            )
        except Exception as exc:
            logger.warning("Speaker ID error: %s", exc)
            speaker, match_conf, embedding = None, None, None
        sid_ms = round((time.monotonic() - t0) * 1000)

        # ── 4. Speech-to-text ───────────────────────────────────────────
        t0 = time.monotonic()
        try:
            transcript = await asyncio.to_thread(self._stt.transcribe, speech_pcm)
        except Exception as exc:
            logger.warning("STT error: %s", exc)
            transcript = ""
        stt_ms = round((time.monotonic() - t0) * 1000)

        if not transcript.strip():
            logger.info("Orchestrator: STT returned empty transcript — discarding utterance")
            return None

        # ── 5. Compute duration ─────────────────────────────────────────
        num_samples  = len(speech_pcm) // 2          # 16-bit = 2 bytes/sample
        duration_ms  = max(1, int(num_samples / _SAMPLE_RATE * 1000))
        total_ms     = round((time.monotonic() - t_start) * 1000)

        timings = {"ns": ns_ms, "vad": vad_ms, "sid": sid_ms, "stt": stt_ms, "total": total_ms}

        # ── 6. Write (fire-and-forget) ──────────────────────────────────
        interaction_id = str(uuid.uuid4())
        asyncio.create_task(
            self._writer.write(
                interaction_id=interaction_id,
                audio_bytes=speech_pcm,
                transcript=transcript,
                duration_ms=duration_ms,
                vad_confidence=vad_conf,
                matched_speaker=speaker,
                match_confidence=match_conf,
                timings=timings,
                embedding=np.array(embedding, dtype=np.float32)
                          if embedding is not None else None,
            )
        )

        result = PipelineResult(
            interaction_id=interaction_id,
            transcript=transcript,
            duration_ms=duration_ms,
            vad_confidence=vad_conf,
            matched_speaker=speaker,
            match_confidence=match_conf,
            timings=timings,
        )
        logger.info(
            "Pipeline: %r | speaker=%s conf=%.2f | dur=%dms | ns=%dms vad=%dms sid=%dms stt=%dms total=%dms",
            transcript,
            speaker or "unknown",
            match_conf or 0.0,
            duration_ms,
            ns_ms, vad_ms, sid_ms, stt_ms, total_ms,
        )
        return result
