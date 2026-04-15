"""
wyoming_server.py — Wyoming-compatible STT server.

Home Assistant connects to this TCP server (default port 10300) and sends
audio via the Wyoming protocol.  Each utterance is processed by the
VoicePipelineOrchestrator and the resulting transcript is returned.

Wyoming protocol flow per utterance:
  HA → Describe           server → Info (service description)
  HA → AudioStart         (sample rate / format metadata)
  HA → AudioChunk(s)      (raw PCM chunks)
  HA → AudioStop          server → Transcript

After speaker identification and before the Transcript event is written,
a webhook is POSTed to the HA Personality Engine (if configured).  This
gives HA the speaker's identity before it invokes the conversation agent.

The server accepts multiple simultaneous connections; each runs its own
handler coroutine.

Usage:
    notifier = WebhookNotifier(db_path)
    server = WyomingServer(orchestrator, db_path, notifier, host="0.0.0.0", port=10300)
    await server.start()
    task = asyncio.create_task(server.serve_forever())
    ...
    task.cancel()
    await server.stop()
    await notifier.close()
"""

import asyncio
import logging
from functools import partial

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.asr import Transcript
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncServer, AsyncEventHandler

from pipeline.orchestrator import VoicePipelineOrchestrator
from pipeline.webhook import WebhookNotifier

logger = logging.getLogger(__name__)

_INFO = Info(
    asr=[
        AsrProgram(
            name="voice-pipeline",
            description=(
                "GTCRN noise suppression → SileroVAD → ERes2Net speaker ID "
                "→ Granite 4.0 1B Speech STT"
            ),
            attribution=Attribution(name="VoicePipeline", url=""),
            installed=True,
            version="1.0.0",
            models=[
                AsrModel(
                    name="granite-4.0-1b-speech",
                    description="IBM Granite 4.0 1B Speech",
                    attribution=Attribution(
                        name="IBM Research",
                        url="https://huggingface.co/ibm-granite",
                    ),
                    installed=True,
                    languages=["en"],
                    version="4.0.1b",
                )
            ],
        )
    ]
)


class _PipelineEventHandler(AsyncEventHandler):
    """Handles one Wyoming client connection (one utterance per connection)."""

    def __init__(
        self,
        wyoming_info: Info,
        orchestrator: VoicePipelineOrchestrator,
        notifier: WebhookNotifier,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        **kwargs,
    ) -> None:
        super().__init__(reader, writer, **kwargs)
        self._wyoming_info = wyoming_info
        self._orchestrator = orchestrator
        self._notifier     = notifier
        self._chunks:      list[bytes] = []
        self._sample_rate: int  = 16_000
        self._channels:    int  = 1
        self._width:       int  = 2      # bytes per sample (16-bit)

    async def handle_event(self, event: Event) -> bool:
        # ── Service description ─────────────────────────────────────────
        if Describe.is_type(event.type):
            await self.write_event(self._wyoming_info.event())

        # ── Audio stream start ──────────────────────────────────────────
        elif AudioStart.is_type(event.type):
            start = AudioStart.from_event(event)
            self._sample_rate = start.rate
            self._channels    = start.channels
            self._width       = start.width
            self._chunks      = []
            logger.debug(
                "Wyoming: AudioStart rate=%d ch=%d width=%d",
                self._sample_rate, self._channels, self._width,
            )

        # ── Audio chunk ─────────────────────────────────────────────────
        elif AudioChunk.is_type(event.type):
            self._chunks.append(AudioChunk.from_event(event).audio)

        # ── End of utterance — process and respond ──────────────────────
        elif AudioStop.is_type(event.type):
            raw_pcm = b"".join(self._chunks)
            self._chunks = []
            logger.debug("Wyoming: AudioStop — %d bytes received", len(raw_pcm))

            # Down-mix to mono 16-bit 16 kHz if HA sent something different
            raw_pcm = _normalise_audio(
                raw_pcm, self._sample_rate, self._channels, self._width
            )

            result = await self._orchestrator.process(raw_pcm)
            text   = result.transcript if result else ""

            # POST speaker metadata to HA Personality Engine BEFORE sending
            # the transcript so the conversation agent can correlate them.
            if result and result.matched_speaker:
                await self._notifier.notify(
                    speaker_id=result.matched_speaker.lower(),
                    confidence=result.match_confidence or 0.0,
                )

            # Always send clean transcript — no speaker prefix.
            # Speaker identity is delivered out-of-band via the webhook above.
            await self.write_event(Transcript(text=text).event())
            logger.info("Wyoming: Transcript sent: %r", text)

            # Return False to close the connection after each utterance.
            # Home Assistant re-connects for the next one.
            return False

        return True


class WyomingServer:
    """Manages the Wyoming TCP server lifecycle."""

    def __init__(
        self,
        orchestrator: VoicePipelineOrchestrator,
        notifier: WebhookNotifier,
        host: str = "0.0.0.0",
        port: int = 10_300,
    ) -> None:
        self._orchestrator = orchestrator
        self._notifier     = notifier
        self._uri          = f"tcp://{host}:{port}"
        self._server:  AsyncServer | None = None

    async def start(self) -> None:
        self._server = AsyncServer.from_uri(self._uri)
        logger.info("Wyoming STT server ready on %s", self._uri)

    async def serve_forever(self) -> None:
        if self._server is None:
            raise RuntimeError("Call start() before serve_forever()")
        await self._server.run(
            partial(_PipelineEventHandler, _INFO, self._orchestrator, self._notifier)
        )

    async def stop(self) -> None:
        # Shutdown is handled by task cancellation from the lifespan handler
        pass


# ---------------------------------------------------------------------------
# Audio normalisation helper
# ---------------------------------------------------------------------------

def _normalise_audio(
    pcm: bytes,
    sample_rate: int,
    channels: int,
    width: int,
) -> bytes:
    """
    Convert multi-channel or non-16 kHz PCM to mono 16-bit 16 kHz.
    No-op if the audio is already in the expected format.
    """
    if sample_rate == 16_000 and channels == 1 and width == 2:
        return pcm

    try:
        import numpy as np
        import torch
        import torchaudio

        dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(width, np.int16)
        audio = np.frombuffer(pcm, dtype=dtype).astype(np.float32)

        # Scale to [-1, 1]
        audio /= float(np.iinfo(dtype).max) if width < 4 else 1.0

        # Reshape to (channels, N)
        if channels > 1:
            audio = audio.reshape(-1, channels).T
        else:
            audio = audio[np.newaxis, :]

        tensor = torch.from_numpy(audio)   # (C, N)

        # Resample if needed
        if sample_rate != 16_000:
            tensor = torchaudio.functional.resample(tensor, sample_rate, 16_000)

        # Down-mix to mono
        if tensor.shape[0] > 1:
            tensor = tensor.mean(dim=0, keepdim=True)

        out = (tensor.squeeze(0).numpy() * 32767.0).clip(-32768, 32767).astype(np.int16)
        return out.tobytes()

    except Exception as exc:
        logger.warning("Audio normalisation failed (%s) — using raw bytes", exc)
        return pcm
