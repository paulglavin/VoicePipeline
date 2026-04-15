"""
speaker_id.py — Speaker identification wrapper.

Uses WeSpeaker ECAPA-TDNN512 via the wespeakerruntime package (ONNX backend).
No torch or SpeechBrain dependency — inference is pure onnxruntime.

Model downloaded on first use by wespeakerruntime; cached at the path
supplied by MODEL_DIR so it persists across container restarts.

identify() compares the input audio's embedding against the mean embeddings
of enrolled speakers (built from their reference_clips). The cache refreshes
every CACHE_TTL seconds or when explicitly called.

All public methods are synchronous and blocking — call via asyncio.to_thread.

NOTE: wespeakerruntime uses a different embedding model from the previous
SpeechBrain ECAPA-TDNN. Embeddings are not cross-compatible — any enrolled
speakers must be re-enrolled after this migration.
"""

import json
import logging
import os
import sqlite3
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

_SR           = 16_000
_MATCH_THRESH_DEFAULT = 0.50   # fallback if settings row is missing
_CACHE_TTL    = 300    # seconds between speaker cache refreshes


class SpeakerIdentifier:
    """
    WeSpeaker ECAPA-TDNN512 speaker identification.

    identify() returns (speaker_name | None, cosine_similarity | None, embedding | None).
    """

    def __init__(self, model_dir: str, db_path: str) -> None:
        self._db_path  = db_path
        self._speaker  = None
        self._cache:   dict[str, np.ndarray] = {}
        self._cache_t  = 0.0
        self._lock     = threading.Lock()

        try:
            import wespeakerruntime as wespeaker

            # Store the downloaded model inside MODEL_DIR so it is persisted
            # on the /models volume and not re-downloaded on every rebuild.
            model_dir_path = Path(model_dir) / "wespeaker"
            model_dir_path.mkdir(parents=True, exist_ok=True)

            # wespeakerruntime downloads its model on the first call if the
            # onnx_path does not yet exist; subsequent starts load from disk.
            onnx_path = model_dir_path / "voxceleb_resnet34_LM.onnx"
            if onnx_path.exists():
                self._speaker = wespeaker.Speaker(
                    lang="en", onnx_path=str(onnx_path)
                )
            else:
                # First run — let the library download, then copy to MODEL_DIR
                logger.info(
                    "WeSpeaker: model not cached, downloading (English / VoxCeleb) …"
                )
                tmp = wespeaker.Speaker(lang="en")
                # Locate the file wespeakerruntime just downloaded
                src = _find_wespeaker_onnx(tmp)
                if src and Path(src).exists():
                    import shutil
                    shutil.copy2(src, onnx_path)
                    logger.info("WeSpeaker model cached at %s", onnx_path)
                self._speaker = tmp

            logger.info("WeSpeaker ECAPA-TDNN loaded (ONNX / CPU)")
        except Exception as exc:
            logger.warning(
                "SpeakerIdentifier init failed (%s) — unknown-speaker mode", exc
            )

    @property
    def available(self) -> bool:
        return self._speaker is not None

    # ------------------------------------------------------------------
    # Speaker embedding cache

    def refresh_cache(self) -> None:
        """
        Rebuild the enrolled-speaker embedding cache from reference clips.
        Thread-safe; skips refresh if the cache is still fresh.
        """
        with self._lock:
            if time.monotonic() - self._cache_t < _CACHE_TTL:
                return
            self._rebuild_cache()

    def force_refresh(self) -> None:
        """Invalidate the cache immediately, bypassing the TTL guard."""
        with self._lock:
            self._cache_t = 0.0

    def _rebuild_cache(self) -> None:
        """Must be called with self._lock held."""
        new_cache: dict[str, np.ndarray] = {}
        try:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT name, reference_clips FROM enrolled_speakers"
            ).fetchall()
            conn.close()

            for row in rows:
                clips: list[str] = json.loads(row["reference_clips"])
                embeddings = [
                    emb for path in clips
                    if (emb := self._embed_file(path)) is not None
                ]
                if embeddings:
                    mean = np.mean(embeddings, axis=0).astype(np.float32)
                    new_cache[row["name"]] = _l2(mean)

        except Exception as exc:
            logger.warning("Speaker cache rebuild failed: %s", exc)

        self._cache   = new_cache
        self._cache_t = time.monotonic()
        logger.info("Speaker cache refreshed: %d enrolled speaker(s)", len(new_cache))

    # ------------------------------------------------------------------
    # Embedding helpers

    def _embed_file(self, path: str) -> np.ndarray | None:
        """Extract a normalised embedding from a WAV file path."""
        if not Path(path).exists():
            logger.debug("Reference clip missing (retention purged?): %s", path)
            return None
        try:
            emb = self._speaker.extract_embedding(path)
            if emb is None:
                return None
            return _l2(np.asarray(emb, dtype=np.float32).flatten())
        except Exception as exc:
            logger.warning("Embedding failed for %s: %s", path, exc)
            return None

    def extract_embedding(self, pcm_bytes: bytes) -> np.ndarray | None:
        """
        Extract a normalised speaker embedding from raw 16-bit LE PCM at 16 kHz.
        Returns None if the model is unavailable or the audio is too short.
        """
        if self._speaker is None or len(pcm_bytes) < _SR * 2 * 0.2:   # < 200 ms
            return None

        try:
            audio = (
                np.frombuffer(pcm_bytes, dtype=np.int16)
                .astype(np.float32) / 32768.0
            )
            # wespeakerruntime expects a file path — write to a temp WAV
            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                sf.write(tmp_path, audio, _SR, subtype="PCM_16")
                emb = self._speaker.extract_embedding(tmp_path)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

            if emb is None:
                return None
            return _l2(np.asarray(emb, dtype=np.float32).flatten())

        except Exception as exc:
            logger.warning("extract_embedding error: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Identification

    def identify(
        self, pcm_bytes: bytes
    ) -> tuple[str | None, float | None, np.ndarray | None]:
        """
        Returns (speaker_name, cosine_similarity, embedding).
        speaker_name is None if no enrolled speaker exceeds MATCH_THRESH.
        """
        self.refresh_cache()
        embedding = self.extract_embedding(pcm_bytes)

        if embedding is None:
            return None, None, None

        if not self._cache:
            return None, None, embedding

        best_name:  str | None = None
        best_score: float      = -1.0

        for name, ref_emb in self._cache.items():
            score = float(np.dot(embedding, ref_emb))   # cosine — both L2-normed
            if score > best_score:
                best_score = score
                best_name  = name

        match_thresh = self._read_threshold("match_threshold", _MATCH_THRESH_DEFAULT)

        if best_score >= match_thresh:
            logger.debug("Speaker identified: %s (score=%.3f)", best_name, best_score)
            return best_name, best_score, embedding

        logger.debug("No match above threshold (best=%s score=%.3f)", best_name, best_score)
        return None, best_score if best_score >= 0 else None, embedding

    def _read_threshold(self, key: str, default: float) -> float:
        try:
            conn = sqlite3.connect(self._db_path)
            row  = conn.execute(
                "SELECT value FROM settings WHERE key = ?", (key,)
            ).fetchone()
            conn.close()
            return float(row["value"]) if row else default
        except Exception:
            return default


# ---------------------------------------------------------------------------
# Helpers


def _l2(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / (norm + 1e-8)


def _find_wespeaker_onnx(speaker_obj) -> str | None:
    """
    Try to locate the ONNX file that wespeakerruntime just downloaded.
    Different versions store it in different places; we check a few.
    Returns the path string or None if not found.
    """
    try:
        # Some versions expose the path directly
        if hasattr(speaker_obj, "onnx_path"):
            return speaker_obj.onnx_path
        if hasattr(speaker_obj, "model_path"):
            return speaker_obj.model_path
        # Inspect the ONNX session for its model path (onnxruntime >=1.14)
        if hasattr(speaker_obj, "session") and hasattr(speaker_obj.session, "_model_path"):
            return speaker_obj.session._model_path
    except Exception:
        pass
    return None
