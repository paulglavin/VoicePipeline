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

    def swap_stt(self, stt: SpeechToText) -> None:
        self._stt = stt

    def swap_sid(self, sid: SpeakerIdentifier) -> None:
        self._sid = sid

    async def process(self, raw_pcm: bytes) -> PipelineResult | None:
        """
        Run the full pipeline on a single utterance.

        raw_pcm — 16-bit LE PCM, 16 kHz, mono (from Wyoming AudioChunk stream)

        Returns PipelineResult on success, None if the utterance should be
        discarded (silence, empty transcript, or upstream error).

        Stage 3 (speaker ID) runs on raw audio to preserve voice characteristics.
        Stage 4 (STT) runs on DTLN-cleaned audio for transcription quality.
        Stages 3 and 4 run in parallel — SID on CPU, STT on GPU.
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
        # VAD runs on DTLN-cleaned audio for best probability estimates.
        # The segment boundaries are then applied to the raw audio as well,
        # so speaker ID receives unprocessed voice characteristics.
        t0 = time.monotonic()
        try:
            segments, vad_conf = await asyncio.to_thread(
                self._vad.extract_segments, clean_pcm
            )
        except Exception as exc:
            logger.warning("VAD error: %s", exc)
            n_samples = len(clean_pcm) // 2
            segments, vad_conf = [(0, n_samples)], 0.5

        if not segments:
            logger.info("Orchestrator: VAD found no speech — discarding utterance")
            return None

        clean_speech = self._vad.apply_segments(clean_pcm, segments)
        raw_speech   = self._vad.apply_segments(raw_pcm,   segments)
        vad_ms = round((time.monotonic() - t0) * 1000)
        logger.info(
            "VAD: speech=%d bytes  conf=%.3f  %dms",
            len(clean_speech), vad_conf, vad_ms,
        )

        # ── 3+4. Speaker ID and STT — run in parallel ───────────────────
        # SID receives raw (unprocessed) audio; STT receives DTLN-cleaned audio.
        # SID runs on CPU (ONNX); STT runs on GPU (CUDA) — no hardware contention.
        (speaker, match_conf, embedding), sid_ms, transcript, stt_ms = (
            await self._run_sid_stt_parallel(raw_speech, clean_speech)
        )

        if not transcript.strip():
            logger.info("Orchestrator: STT returned empty transcript — discarding utterance")
            return None

        # ── 5. Compute duration ─────────────────────────────────────────
        num_samples = len(clean_speech) // 2
        duration_ms = max(1, int(num_samples / _SAMPLE_RATE * 1000))
        total_ms    = round((time.monotonic() - t_start) * 1000)

        timings = {"ns": ns_ms, "vad": vad_ms, "sid": sid_ms, "stt": stt_ms, "total": total_ms}

        # ── 6. Write (fire-and-forget) ──────────────────────────────────
        interaction_id = str(uuid.uuid4())
        asyncio.create_task(
            self._writer.write(
                interaction_id=interaction_id,
                audio_bytes=clean_speech,
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

    async def _run_sid_stt_parallel(
        self, raw_speech: bytes, clean_speech: bytes
    ) -> tuple[tuple, int, str, int]:
        """
        Run speaker ID and STT concurrently and return their results with timings.
        Returns (sid_result_tuple, sid_ms, transcript, stt_ms).
        """
        async def _sid() -> tuple[tuple, int]:
            t = time.monotonic()
            try:
                result = await asyncio.to_thread(self._sid.identify, raw_speech)
            except Exception as exc:
                logger.warning("Speaker ID error: %s", exc)
                result = (None, None, None)
            return result, round((time.monotonic() - t) * 1000)

        async def _stt() -> tuple[str, int]:
            t = time.monotonic()
            try:
                result = await asyncio.to_thread(self._stt.transcribe, clean_speech)
            except Exception as exc:
                logger.warning("STT error: %s", exc)
                result = ""
            return result, round((time.monotonic() - t) * 1000)

        (sid_result, sid_ms), (transcript, stt_ms) = await asyncio.gather(
            _sid(), _stt()
        )
        return sid_result, sid_ms, transcript, stt_ms
