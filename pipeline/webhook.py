"""
webhook.py — Notify Home Assistant of speaker identification results.

Posts a JSON payload to the Personality Engine webhook endpoint in HA
immediately after speaker identification, before the Wyoming transcript is
sent.  This gives the HA conversation agent the speaker's identity before
it processes the utterance.

The POST is awaited with a 1-second timeout.  If HA is unreachable or slow,
a warning is logged and the pipeline continues — the Wyoming transcript is
sent regardless.

Configuration (via environment variables or DB settings):
    HA_BASE_URL          e.g. http://homeassistant.local:8123
    HA_WEBHOOK_ENABLED   "true" to enable (default: "false")

Usage:
    notifier = WebhookNotifier.from_env()
    await notifier.notify(speaker_id="paul", confidence=0.92, interaction_id="uuid")
"""

import asyncio
import logging
import os
import sqlite3
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_WEBHOOK_PATH = "/api/webhook/personality_engine_input"
_TIMEOUT_SECONDS = 1.0


class WebhookNotifier:
    """Posts speaker metadata to the HA Personality Engine webhook.

    Instantiate once and reuse across utterances.  No state is held between
    calls — the DB settings are re-read on each notify() call so live
    changes in the management UI take effect immediately.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._session = None  # aiohttp.ClientSession, created lazily

    # ------------------------------------------------------------------

    async def notify(
        self,
        speaker_id: str,
        confidence: float,
        interaction_id: str | None = None,
    ) -> None:
        """POST speaker metadata to HA.  Fire-and-forget safe to await.

        Reads ha_base_url and ha_webhook_enabled from the DB settings table
        on every call so management-UI changes are picked up without restart.

        Args:
            speaker_id:     Lowercase speaker name (e.g. "paul").
            confidence:     Match confidence 0.0–1.0.
            interaction_id: Optional VoicePipeline interaction UUID.
        """
        ha_base_url, enabled = self._read_settings()
        if not enabled or not ha_base_url:
            return

        url = ha_base_url.rstrip("/") + _WEBHOOK_PATH
        payload: dict = {
            "speaker_id": speaker_id,
            "confidence": round(float(confidence), 4),
            "timestamp":  datetime.now(timezone.utc).isoformat(),
        }
        if interaction_id:
            payload["interaction_id"] = interaction_id

        try:
            await self._post(url, payload)
            logger.info(
                "Webhook: POST → HA succeeded  speaker=%s conf=%.2f",
                speaker_id, confidence,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Webhook: POST to %s timed out (%.1f s) — continuing without HA speaker context",
                url, _TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.warning(
                "Webhook: POST to %s failed (%s) — continuing without HA speaker context",
                url, exc,
            )

    # ------------------------------------------------------------------

    async def _post(self, url: str, payload: dict) -> None:
        """HTTP POST with timeout.  Raises on non-2xx or timeout."""
        import aiohttp

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

        async with asyncio.timeout(_TIMEOUT_SECONDS):
            async with self._session.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    raise RuntimeError(
                        f"HTTP {resp.status}: {body[:200]}"
                    )

    async def close(self) -> None:
        """Close the underlying HTTP session (call on app shutdown)."""
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------

    def _read_settings(self) -> tuple[str, bool]:
        """Read ha_base_url and ha_webhook_enabled from DB settings.

        Falls back to environment variables if the DB values are absent
        (supports both config sources for flexibility).

        Returns:
            Tuple of (ha_base_url, enabled).
        """
        # Environment variable fallbacks
        env_url     = os.getenv("HA_BASE_URL", "").strip()
        env_enabled = os.getenv("HA_WEBHOOK_ENABLED", "false").lower() == "true"

        try:
            conn = sqlite3.connect(self._db_path)
            rows = conn.execute(
                "SELECT key, value FROM settings WHERE key IN ('ha_base_url', 'ha_webhook_enabled')"
            ).fetchall()
            conn.close()

            settings = {r[0]: r[1] for r in rows}
            # Env vars take priority when set; DB provides UI-controlled overrides.
            url = env_url or settings.get("ha_base_url", "").strip()
            if env_enabled:
                enabled = True
            else:
                enabled = settings.get("ha_webhook_enabled", "false").lower() == "true"
            return url, enabled

        except Exception as exc:
            logger.debug("Webhook: DB settings read failed (%s) — using env vars", exc)
            return env_url, env_enabled
