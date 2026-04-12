"""
writer.py — Pipeline write path for the speaker management system.

Called by the orchestrator at the end of the four-model chain.
Writes a completed interaction record to SQLite and filesystem,
then returns immediately (fire and forget via asyncio).

Usage:
    from pipeline.writer import InteractionWriter

    writer = InteractionWriter(db_path="/data/speaker.db", audio_dir="/data/audio")

    # At the end of your pipeline:
    asyncio.create_task(
        writer.write(
            audio_bytes=wav_bytes,
            transcript="Turn the kitchen lights off",
            duration_ms=2400,
            vad_confidence=0.97,
            matched_speaker="Haydn",
            match_confidence=0.62,
            embedding=numpy_array,
        )
    )
"""

import asyncio
import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class InteractionWriter:
    def __init__(self, db_path: str, audio_dir: str, embedding_dir: str | None = None):
        """
        db_path       — path to the SQLite database file
        audio_dir     — directory where raw WAV clips are stored
        embedding_dir — directory where .npy embedding files are stored
                        defaults to audio_dir/embeddings if not supplied
        """
        self.db_path = db_path
        self.audio_dir = Path(audio_dir)
        self.embedding_dir = Path(embedding_dir) if embedding_dir else self.audio_dir / "embeddings"

        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def write(
        self,
        audio_bytes: bytes,
        transcript: str,
        duration_ms: int,
        vad_confidence: float,
        matched_speaker: str | None,
        match_confidence: float | None,
        embedding: np.ndarray | None,
    ) -> str:
        """
        Write a completed interaction record.

        Returns the interaction UUID. Any exception is caught and logged —
        the pipeline should never be blocked or crashed by a write failure.
        """
        interaction_id = str(uuid.uuid4())
        recorded_at = datetime.now(timezone.utc).isoformat()

        try:
            # Determine initial status
            # Unknown speaker or low confidence (<0.75) → pending
            # Confident match (≥0.75) → resolved immediately as auto-confirmed
            if matched_speaker is None or (match_confidence is not None and match_confidence < 0.75):
                status = "pending"
            else:
                status = "resolved"

            # Write audio file
            audio_path = await asyncio.to_thread(
                self._write_audio, interaction_id, audio_bytes
            )

            # Write embedding file + get blob
            embedding_path, embedding_blob = await asyncio.to_thread(
                self._write_embedding, interaction_id, embedding
            )

            # Write to database
            await asyncio.to_thread(
                self._write_db,
                interaction_id,
                recorded_at,
                str(audio_path),
                transcript,
                duration_ms,
                vad_confidence,
                matched_speaker,
                match_confidence,
                embedding_blob,
                str(embedding_path) if embedding_path else None,
                status,
            )

            # If auto-resolved (high confidence), write a resolution record
            # and update the enrolled speaker profile
            if status == "resolved" and matched_speaker:
                await asyncio.to_thread(
                    self._write_auto_resolution,
                    interaction_id,
                    matched_speaker,
                    match_confidence,
                    str(audio_path),
                )

            logger.info(
                "Interaction written: id=%s speaker=%s confidence=%s status=%s",
                interaction_id,
                matched_speaker,
                match_confidence,
                status,
            )

        except Exception:
            logger.exception("Failed to write interaction %s", interaction_id)

        return interaction_id

    # ------------------------------------------------------------------
    # Private — filesystem writes (run in thread pool via asyncio.to_thread)
    # ------------------------------------------------------------------

    def _write_audio(self, interaction_id: str, audio_bytes: bytes) -> Path:
        path = self.audio_dir / f"{interaction_id}.wav"
        path.write_bytes(audio_bytes)
        return path

    def _write_embedding(
        self, interaction_id: str, embedding: np.ndarray | None
    ) -> tuple[Path | None, bytes | None]:
        if embedding is None:
            return None, None

        path = self.embedding_dir / f"{interaction_id}.npy"
        np.save(path, embedding)

        # Inline blob: float32, little-endian bytes
        blob = embedding.astype(np.float32).tobytes()
        return path, blob

    # ------------------------------------------------------------------
    # Private — database writes
    # ------------------------------------------------------------------

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")   # safe for concurrent readers
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _write_db(
        self,
        interaction_id: str,
        recorded_at: str,
        audio_path: str,
        transcript: str,
        duration_ms: int,
        vad_confidence: float,
        matched_speaker: str | None,
        match_confidence: float | None,
        embedding_blob: bytes | None,
        embedding_path: str | None,
        status: str,
    ) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO interactions (
                    id, recorded_at, audio_path, transcript, duration_ms,
                    vad_confidence, matched_speaker, match_confidence,
                    embedding_blob, embedding_path, status, created_at
                ) VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?
                )
                """,
                (
                    interaction_id,
                    recorded_at,
                    audio_path,
                    transcript,
                    duration_ms,
                    vad_confidence,
                    matched_speaker,
                    match_confidence,
                    embedding_blob,
                    embedding_path,
                    status,
                    recorded_at,
                ),
            )

    def _write_auto_resolution(
        self,
        interaction_id: str,
        matched_speaker: str,
        match_confidence: float,
        audio_path: str,
    ) -> None:
        """
        For high-confidence matches the pipeline auto-confirms.
        Writes a resolution record and updates the enrolled speaker profile.
        """
        resolution_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        with self._get_connection() as conn:
            # Write resolution record
            conn.execute(
                """
                INSERT INTO resolutions (id, interaction_id, resolved_at, action, assigned_to)
                VALUES (?, ?, ?, 'confirm', ?)
                """,
                (resolution_id, interaction_id, now, matched_speaker),
            )

            # Update enrolled speaker stats
            conn.execute(
                """
                UPDATE enrolled_speakers
                SET
                    interaction_count = interaction_count + 1,
                    last_active_at    = ?,
                    avg_confidence    = (
                        COALESCE(avg_confidence, 0.0) * interaction_count + ?
                    ) / (interaction_count + 1)
                WHERE name = ?
                """,
                (now, match_confidence, matched_speaker),
            )

            # Rotate reference clips — keep the N most recent high-confidence ones
            # N is read from settings
            row = conn.execute(
                "SELECT value FROM settings WHERE key = 'reference_clips_per_speaker'"
            ).fetchone()
            max_clips = int(row["value"]) if row else 5

            speaker_row = conn.execute(
                "SELECT reference_clips FROM enrolled_speakers WHERE name = ?",
                (matched_speaker,),
            ).fetchone()

            if speaker_row:
                clips: list[str] = json.loads(speaker_row["reference_clips"])

                # Only rotate in clips that are high confidence (≥0.85)
                if match_confidence >= 0.85:
                    clips.append(audio_path)
                    if max_clips > 0:
                        clips = clips[-max_clips:]   # keep most recent N

                    conn.execute(
                        "UPDATE enrolled_speakers SET reference_clips = ? WHERE name = ?",
                        (json.dumps(clips), matched_speaker),
                    )
