"""
retention.py — Scheduled retention job.

Purges audio files and interaction records according to the
retention settings stored in the settings table.

Designed to run as an APScheduler job inside the FastAPI process:

    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from pipeline.retention import RetentionJob

    scheduler = AsyncIOScheduler()
    job = RetentionJob(db_path="/data/speaker.db")
    scheduler.add_job(job.run, "interval", hours=6)
    scheduler.start()

The job runs every 6 hours by default. Adjust the interval to taste.
"""

import asyncio
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class RetentionJob:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _load_settings(self, conn: sqlite3.Connection) -> dict:
        rows = conn.execute("SELECT key, value FROM settings").fetchall()
        return {r["key"]: r["value"] for r in rows}

    async def run(self) -> None:
        """Entry point called by the scheduler."""
        try:
            await asyncio.to_thread(self._run_sync)
        except Exception:
            logger.exception("Retention job failed")

    def _run_sync(self) -> None:
        with self._get_connection() as conn:
            settings = self._load_settings(conn)

            pending_days  = int(settings.get("pending_retention_days", 7))
            resolved_days = int(settings.get("resolved_retention_days", 3))

            now = datetime.now(timezone.utc)

            # -- Pending clips ------------------------------------------------
            # Delete interactions (and their files) that are still pending
            # and older than pending_retention_days.
            # 0 = never purge.
            if pending_days > 0:
                cutoff = (now - timedelta(days=pending_days)).isoformat()
                old_pending = conn.execute(
                    """
                    SELECT id, audio_path, embedding_path
                    FROM interactions
                    WHERE status = 'pending'
                    AND recorded_at < ?
                    """,
                    (cutoff,),
                ).fetchall()

                for row in old_pending:
                    self._delete_files(row["audio_path"], row["embedding_path"])

                if old_pending:
                    ids = [r["id"] for r in old_pending]
                    placeholders = ",".join("?" * len(ids))
                    # Resolutions cascade — but these are pending so there
                    # shouldn't be any. Delete explicitly to be safe.
                    conn.execute(
                        f"DELETE FROM resolutions WHERE interaction_id IN ({placeholders})",
                        ids,
                    )
                    conn.execute(
                        f"DELETE FROM interactions WHERE id IN ({placeholders})",
                        ids,
                    )
                    logger.info("Retention: purged %d pending interactions", len(old_pending))

            # -- Resolved audio files ----------------------------------------
            # Delete only the WAV file for resolved interactions older than
            # resolved_retention_days. Metadata row is kept permanently.
            # -1 = keep indefinitely.
            if resolved_days >= 0:
                cutoff = (now - timedelta(days=resolved_days)).isoformat()
                old_resolved = conn.execute(
                    """
                    SELECT id, audio_path, embedding_path
                    FROM interactions
                    WHERE status = 'resolved'
                    AND recorded_at < ?
                    AND audio_path IS NOT NULL
                    """,
                    (cutoff,),
                ).fetchall()

                for row in old_resolved:
                    # Delete audio file only — keep embedding for future
                    # re-analysis if needed
                    self._delete_file(row["audio_path"])
                    conn.execute(
                        "UPDATE interactions SET audio_path = NULL WHERE id = ?",
                        (row["id"],),
                    )

                if old_resolved:
                    logger.info(
                        "Retention: purged audio for %d resolved interactions",
                        len(old_resolved),
                    )

    @staticmethod
    def _delete_files(*paths: str | None) -> None:
        for path in paths:
            RetentionJob._delete_file(path)

    @staticmethod
    def _delete_file(path: str | None) -> None:
        if not path:
            return
        p = Path(path)
        try:
            if p.exists():
                p.unlink()
        except OSError:
            logger.warning("Could not delete file: %s", path)
