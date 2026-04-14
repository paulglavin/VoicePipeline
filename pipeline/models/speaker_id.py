"""
speaker_id.py — Speaker identification wrapper.

Uses SpeechBrain's ECAPA-TDNN model (speechbrain/spkrec-ecapa-voxceleb),
which provides equivalent quality to ERes2Net for speaker verification.
Model is downloaded from HuggingFace on first run and cached in
MODEL_DIR/speechbrain/.

identify() compares the input audio's embedding against the mean embeddings
of enrolled speakers (built from their reference_clips). The cache refreshes
every CACHE_TTL seconds or when explicitly called.

All public methods are synchronous and blocking — call via asyncio.to_thread.
"""

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_SR           = 16_000
_MATCH_THRESH_DEFAULT = 0.50   # fallback if settings row is missing
_CACHE_TTL    = 300    # seconds between speaker cache refreshes
_HF_MODEL     = "speechbrain/spkrec-ecapa-voxceleb"


class SpeakerIdentifier:
    """
    ECAPA-TDNN speaker identification.

    identify() returns (speaker_name | None, cosine_similarity | None, embedding | None).
    """

    def __init__(self, model_dir: str, db_path: str) -> None:
        self._db_path  = db_path
        self._model    = None
        self._device   = "cpu"
        self._cache:   dict[str, np.ndarray] = {}
        self._cache_t  = 0.0
        self._lock     = threading.Lock()
        self._save_dir = str(Path(model_dir) / "speechbrain")

        try:
            import torch
            from speechbrain.inference.speaker import EncoderClassifier
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = EncoderClassifier.from_hparams(
                source=_HF_MODEL,
                savedir=self._save_dir,
                run_opts={"device": self._device},
            )
            logger.info("SpeechBrain ECAPA-TDNN loaded on %s from %s", self._device, _HF_MODEL)
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
            import wave
            import torch

            with wave.open(path, "rb") as wf:
                raw = wf.readframes(wf.getnframes())
                sr  = wf.getframerate()

            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

            if sr != _SR:
                # Simple resample via torch if rate differs
                signal = torch.from_numpy(audio).unsqueeze(0)
                import torchaudio
                signal = torchaudio.functional.resample(signal, sr, _SR)
            else:
                signal = torch.from_numpy(audio).unsqueeze(0)  # (1, N)

            signal = signal.to(self._device)
            with torch.no_grad():
                emb = self._model.encode_batch(signal)   # (1, 1, dim)

            arr = emb.squeeze().cpu().numpy().astype(np.float32)
            return _l2(arr)
        except Exception as exc:
            logger.warning("Embedding failed for %s: %s", path, exc)
            return None

    def extract_embedding(self, pcm_bytes: bytes) -> np.ndarray | None:
        """
        Extract a normalised speaker embedding from raw 16-bit LE PCM at 16 kHz.
        Returns None if the model is unavailable or the audio is too short.
        """
        if self._model is None or len(pcm_bytes) < _SR * 2 * 0.2:   # < 200 ms
            return None

        try:
            import torch

            audio = (
                np.frombuffer(pcm_bytes, dtype=np.int16)
                .astype(np.float32) / 32768.0
            )
            tensor = torch.from_numpy(audio).unsqueeze(0).to(self._device)   # (1, N)

            with torch.no_grad():
                emb = self._model.encode_batch(tensor)      # (1, 1, dim)

            arr = emb.squeeze().cpu().numpy().astype(np.float32)
            return _l2(arr)

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

def _l2(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / (norm + 1e-8)
