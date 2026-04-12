"""
speaker_id.py — ERes2Net speaker identification wrapper.

Uses the wespeaker package (https://github.com/wenet-e2e/wespeaker).
The pretrained ERes2Net model is downloaded via wespeaker's model hub on
first run and cached in MODEL_DIR/wespeaker/.

identify() compares the input audio's embedding against the mean embeddings
of enrolled speakers (built from their reference_clips). The cache refreshes
every CACHE_TTL seconds or when explicitly called.

All public methods are synchronous and blocking — call via asyncio.to_thread.
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

logger = logging.getLogger(__name__)

_SR            = 16_000
_MATCH_THRESH  = 0.75   # cosine similarity threshold — matches writer.py
_CACHE_TTL     = 300    # seconds between speaker cache refreshes

# wespeaker pretrained model name.  'english' resolves to the VoxCeleb
# ResNet34-LM model which uses the ERes2Net architecture.
_WESPEAKER_MODEL = "english"


class SpeakerIdentifier:
    """
    ERes2Net-based speaker identification.

    identify() returns (speaker_name | None, cosine_similarity | None, embedding | None).
    """

    def __init__(self, model_dir: str, db_path: str) -> None:
        self._db_path     = db_path
        self._model       = None
        self._cache:  dict[str, np.ndarray] = {}
        self._cache_t     = 0.0
        self._lock        = threading.Lock()

        cache_dir = str(Path(model_dir) / "wespeaker")
        os.environ.setdefault("WESPEAKER_HOME", cache_dir)

        try:
            import wespeaker
            self._model = wespeaker.load_model(_WESPEAKER_MODEL)
            logger.info("ERes2Net (wespeaker/%s) loaded", _WESPEAKER_MODEL)
        except Exception as exc:
            logger.warning(
                "SpeakerIdentifier init failed (%s) — unknown-speaker mode", exc
            )

    @property
    def available(self) -> bool:
        return self._model is not None

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
        """Extract L2-normalised embedding from a WAV file path."""
        if not Path(path).exists():
            logger.debug("Reference clip missing (retention purged?): %s", path)
            return None
        try:
            raw = self._model.extract_embedding(path)
            if raw is None:
                return None
            arr = np.array(raw, dtype=np.float32)
            return _l2(arr)
        except Exception as exc:
            logger.debug("Embedding failed for %s: %s", path, exc)
            return None

    def extract_embedding(self, pcm_bytes: bytes) -> np.ndarray | None:
        """
        Extract an L2-normalised speaker embedding from raw 16-bit PCM.
        Returns None if the model is unavailable or the audio is too short.
        """
        if self._model is None or len(pcm_bytes) < _SR * 2 * 0.2:   # < 200 ms
            return None

        try:
            # wespeaker works on WAV files — write a temp file
            audio = (
                np.frombuffer(pcm_bytes, dtype=np.int16)
                .astype(np.float32) / 32768.0
            )

            # Use extract_embedding_from_pcm if available (wespeaker ≥ 1.2)
            if hasattr(self._model, "extract_embedding_from_pcm"):
                raw = self._model.extract_embedding_from_pcm(audio, sample_rate=_SR)
            else:
                import soundfile as sf
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
                    tmp = fh.name
                try:
                    sf.write(tmp, audio, _SR, subtype="PCM_16")
                    raw = self._model.extract_embedding(tmp)
                finally:
                    Path(tmp).unlink(missing_ok=True)

            if raw is None:
                return None
            return _l2(np.array(raw, dtype=np.float32))

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

        if best_score >= _MATCH_THRESH:
            logger.debug(
                "Speaker identified: %s (score=%.3f)", best_name, best_score
            )
            return best_name, best_score, embedding

        logger.debug(
            "No match above threshold (best=%s score=%.3f)", best_name, best_score
        )
        return None, best_score if best_score >= 0 else None, embedding


# ---------------------------------------------------------------------------

def _l2(v: np.ndarray) -> np.ndarray:
    """L2-normalise a vector in place and return it."""
    norm = np.linalg.norm(v)
    return v / (norm + 1e-8)
