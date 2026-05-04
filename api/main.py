"""
main.py — FastAPI speaker management application.

Start with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Or via environment variables:
    DB_PATH=/data/speaker.db AUDIO_DIR=/data/audio uvicorn api.main:app
"""

import asyncio
import json
import logging
import os
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from pipeline.init_db import init as init_db
from pipeline.models import NoiseSuppressor, VoiceActivityDetector, SpeakerIdentifier, SpeechToText
from pipeline.orchestrator import VoicePipelineOrchestrator
from pipeline.retention import RetentionJob
from pipeline.writer import InteractionWriter
from pipeline.webhook import WebhookNotifier
from pipeline.wyoming_server import WyomingServer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH        = os.getenv("DB_PATH",        "/data/speaker.db")
AUDIO_DIR      = os.getenv("AUDIO_DIR",      "/data/audio")
EMBEDDING_DIR  = os.getenv("EMBEDDING_DIR",  "/data/embeddings")
MODEL_DIR      = os.getenv("MODEL_DIR",      "/models")
WYOMING_HOST   = os.getenv("WYOMING_HOST",   "0.0.0.0")
WYOMING_PORT   = int(os.getenv("WYOMING_PORT", "10300"))
HA_BASE_URL    = os.getenv("HA_BASE_URL",    "")
HA_WEBHOOK_ENABLED = os.getenv("HA_WEBHOOK_ENABLED", "false")

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class InteractionBase(BaseModel):
    id: str
    recorded_at: str
    transcript: str
    duration_ms: int
    vad_confidence: float
    matched_speaker: str | None
    match_confidence: float | None
    status: str
    audio_url: str | None


class PendingInteraction(InteractionBase):
    # Present only for low-confidence matches (matched_speaker is not None)
    reference_clip_url: str | None


class HistoryInteraction(InteractionBase):
    current_action: str | None
    current_assigned_to: str | None
    last_resolved_at: str | None


class EnrolledSpeakerOut(BaseModel):
    id: str
    name: str
    created_at: str
    last_active_at: str | None
    interaction_count: int
    avg_confidence: float | None
    reference_clip_count: int


class ResolveRequest(BaseModel):
    action: str
    assigned_to: str | None = None

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        allowed = {"confirm", "reject", "assign", "enrol", "dismiss"}
        if v not in allowed:
            raise ValueError(f"action must be one of {sorted(allowed)}")
        return v


class ResolveResponse(BaseModel):
    interaction_id: str
    action: str
    new_status: str


class SettingsResponse(BaseModel):
    pending_retention_days: int
    resolved_retention_days: int
    reference_clips_per_speaker: int
    match_threshold: float
    confirm_threshold: float
    personality_processing: bool
    ha_base_url: str
    ha_webhook_enabled: bool


class SettingsUpdate(BaseModel):
    pending_retention_days: int | None = None
    resolved_retention_days: int | None = None
    reference_clips_per_speaker: int | None = None
    match_threshold: float | None = None
    confirm_threshold: float | None = None
    personality_processing: bool | None = None
    ha_base_url: str | None = None
    ha_webhook_enabled: bool | None = None


class ModelSettingsResponse(BaseModel):
    stt_provider: str
    stt_model: str
    stt_base_url: str
    stt_api_key: str
    stt_prompt: str
    speaker_id_model: str


class ModelSettingsUpdate(BaseModel):
    stt_provider: str | None = None
    stt_model: str | None = None
    stt_base_url: str | None = None
    stt_api_key: str | None = None
    stt_prompt: str | None = None
    speaker_id_model: str | None = None


class ModelReloadStatus(BaseModel):
    status: str   # "reloading" | "ready" | "error"
    detail: str = ""


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _audio_url(interaction_id: str, audio_path: str | None) -> str | None:
    if not audio_path:
        return None
    return f"/api/v1/audio/{interaction_id}"


# ---------------------------------------------------------------------------
# Query helpers (synchronous — called via asyncio.to_thread)
# ---------------------------------------------------------------------------


def _fetch_pending() -> list[dict]:
    with _get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, recorded_at, transcript, duration_ms, vad_confidence,
                   matched_speaker, match_confidence, status, audio_path
            FROM interactions
            WHERE status = 'pending'
            ORDER BY recorded_at DESC
            """
        ).fetchall()

        result = []
        for row in rows:
            d = dict(row)

            # Build reference_clip_url for low-confidence matches
            ref_clip_url = None
            if d["matched_speaker"]:
                speaker_row = conn.execute(
                    "SELECT reference_clips FROM enrolled_speakers WHERE name = ?",
                    (d["matched_speaker"],),
                ).fetchone()
                if speaker_row:
                    clips: list[str] = json.loads(speaker_row["reference_clips"])
                    if clips:
                        # Most recent clip — derive interaction_id from filename stem
                        clip_stem = Path(clips[-1]).stem
                        ref_clip_url = f"/api/v1/audio/{clip_stem}"

            result.append(
                {
                    "id": d["id"],
                    "recorded_at": d["recorded_at"],
                    "transcript": d["transcript"],
                    "duration_ms": d["duration_ms"],
                    "vad_confidence": d["vad_confidence"],
                    "matched_speaker": d["matched_speaker"],
                    "match_confidence": d["match_confidence"],
                    "status": d["status"],
                    "audio_url": _audio_url(d["id"], d["audio_path"]),
                    "reference_clip_url": ref_clip_url,
                }
            )
        return result


def _fetch_history(
    speaker: str | None,
    date_from: str | None,
    date_to: str | None,
) -> list[dict]:
    filters: list[str] = []
    params: list[str] = []

    if speaker:
        filters.append("i.matched_speaker = ?")
        params.append(speaker)
    if date_from:
        filters.append("i.recorded_at >= ?")
        params.append(date_from)
    if date_to:
        filters.append("i.recorded_at <= ?")
        params.append(date_to)

    where = ("WHERE " + " AND ".join(filters)) if filters else ""

    with _get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT i.id, i.recorded_at, i.transcript, i.duration_ms, i.vad_confidence,
                   i.matched_speaker, i.match_confidence, i.status, i.audio_path,
                   i.timings,
                   r.action      AS current_action,
                   r.assigned_to AS current_assigned_to,
                   r.resolved_at AS last_resolved_at
            FROM interactions i
            LEFT JOIN resolutions r ON r.id = (
                SELECT id FROM resolutions
                WHERE interaction_id = i.id
                ORDER BY resolved_at DESC
                LIMIT 1
            )
            {where}
            ORDER BY i.recorded_at DESC
            """,
            params,
        ).fetchall()

    return [
        {
            "id": row["id"],
            "recorded_at": row["recorded_at"],
            "transcript": row["transcript"],
            "duration_ms": row["duration_ms"],
            "vad_confidence": row["vad_confidence"],
            "matched_speaker": row["matched_speaker"],
            "match_confidence": row["match_confidence"],
            "status": row["status"],
            "audio_url": _audio_url(row["id"], row["audio_path"]),
            "current_action": row["current_action"],
            "current_assigned_to": row["current_assigned_to"],
            "last_resolved_at": row["last_resolved_at"],
            "timings": json.loads(row["timings"]) if row["timings"] else None,
        }
        for row in rows
    ]


def _do_resolve(interaction_id: str, action: str, assigned_to: str | None) -> dict:
    """
    Atomic resolve: writes resolution, updates interaction status, and applies
    any enrolled_speakers side-effects — all within a single transaction.
    """
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM interactions WHERE id = ?", (interaction_id,)
        ).fetchone()
        if not row:
            raise LookupError(f"Interaction {interaction_id!r} not found")

        new_status = "resolved" if action in ("confirm", "assign", "enrol") else "dismissed"
        now = datetime.now(timezone.utc).isoformat()
        resolution_id = str(uuid.uuid4())

        conn.execute(
            """
            INSERT INTO resolutions (id, interaction_id, resolved_at, action, assigned_to)
            VALUES (?, ?, ?, ?, ?)
            """,
            (resolution_id, interaction_id, now, action, assigned_to),
        )
        conn.execute(
            "UPDATE interactions SET status = ? WHERE id = ?",
            (new_status, interaction_id),
        )

        if action == "enrol" and assigned_to:
            _enrol_speaker(conn, assigned_to, row["audio_path"], now)
        elif action in ("confirm", "assign") and assigned_to:
            _update_speaker_stats(conn, assigned_to, row["match_confidence"], row["audio_path"], now)

    return {"interaction_id": interaction_id, "action": action, "new_status": new_status}


def _enrol_speaker(
    conn: sqlite3.Connection, name: str, audio_path: str | None, now: str
) -> None:
    existing = conn.execute(
        "SELECT id, reference_clips, interaction_count FROM enrolled_speakers WHERE name = ?",
        (name,),
    ).fetchone()

    if existing:
        clips: list[str] = json.loads(existing["reference_clips"])
        if audio_path and audio_path not in clips:
            clips.append(audio_path)
            row = conn.execute(
                "SELECT value FROM settings WHERE key = 'reference_clips_per_speaker'"
            ).fetchone()
            max_clips = int(row["value"]) if row else 5
            if max_clips > 0:
                clips = clips[-max_clips:]
        conn.execute(
            """
            UPDATE enrolled_speakers
            SET reference_clips = ?, last_active_at = ?, interaction_count = interaction_count + 1
            WHERE name = ?
            """,
            (json.dumps(clips), now, name),
        )
    else:
        clips = [audio_path] if audio_path else []
        conn.execute(
            """
            INSERT INTO enrolled_speakers
                (id, name, created_at, last_active_at, interaction_count, avg_confidence, reference_clips)
            VALUES (?, ?, ?, ?, 1, NULL, ?)
            """,
            (str(uuid.uuid4()), name, now, now, json.dumps(clips)),
        )


def _update_speaker_stats(
    conn: sqlite3.Connection,
    name: str,
    match_confidence: float | None,
    audio_path: str | None,
    now: str,
) -> None:
    existing = conn.execute(
        "SELECT interaction_count, avg_confidence, reference_clips FROM enrolled_speakers WHERE name = ?",
        (name,),
    ).fetchone()
    if not existing:
        return

    count = existing["interaction_count"]
    if match_confidence is not None:
        old_avg = existing["avg_confidence"] or 0.0
        new_avg = (old_avg * count + match_confidence) / (count + 1)
    else:
        new_avg = existing["avg_confidence"]   # preserve existing, don't factor in manual confirms

    conn.execute(
        """
        UPDATE enrolled_speakers
        SET interaction_count = interaction_count + 1, last_active_at = ?, avg_confidence = ?
        WHERE name = ?
        """,
        (now, new_avg, name),
    )

    # Manual confirm/assign always adds to reference clips — the user is explicitly
    # saying "this is me", so we trust it regardless of pipeline confidence.
    # (The 0.85 threshold applies only to auto-confirmation in writer.py.)
    if audio_path:
        clips: list[str] = json.loads(existing["reference_clips"])
        clips.append(audio_path)
        row = conn.execute(
            "SELECT value FROM settings WHERE key = 'reference_clips_per_speaker'"
        ).fetchone()
        max_clips = int(row["value"]) if row else 5
        if max_clips > 0:
            clips = clips[-max_clips:]
        conn.execute(
            "UPDATE enrolled_speakers SET reference_clips = ? WHERE name = ?",
            (json.dumps(clips), name),
        )


def _fetch_settings() -> dict[str, str]:
    with _get_connection() as conn:
        rows = conn.execute("SELECT key, value FROM settings").fetchall()
    return {r["key"]: r["value"] for r in rows}


def _save_settings(updates: dict[str, int]) -> dict[str, str]:
    with _get_connection() as conn:
        for key, value in updates.items():
            conn.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                (key, str(value)),
            )
        rows = conn.execute("SELECT key, value FROM settings").fetchall()
    return {r["key"]: r["value"] for r in rows}


def _fetch_speakers() -> list[dict]:
    with _get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, name, created_at, last_active_at, interaction_count,
                   avg_confidence, reference_clips
            FROM enrolled_speakers
            ORDER BY name
            """
        ).fetchall()
    return [
        {
            "id": row["id"],
            "name": row["name"],
            "created_at": row["created_at"],
            "last_active_at": row["last_active_at"],
            "interaction_count": row["interaction_count"],
            "avg_confidence": row["avg_confidence"],
            "reference_clip_count": len(json.loads(row["reference_clips"])),
        }
        for row in rows
    ]


def _delete_speaker(speaker_id: str) -> None:
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT name, reference_clips FROM enrolled_speakers WHERE id = ?",
            (speaker_id,),
        ).fetchone()
        if not row:
            raise LookupError(f"Speaker {speaker_id!r} not found")
        clips: list[str] = json.loads(row["reference_clips"])
        for path in clips:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass
        conn.execute("DELETE FROM enrolled_speakers WHERE id = ?", (speaker_id,))


def _clear_speaker_clips(speaker_id: str) -> None:
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT reference_clips FROM enrolled_speakers WHERE id = ?",
            (speaker_id,),
        ).fetchone()
        if not row:
            raise LookupError(f"Speaker {speaker_id!r} not found")
        clips: list[str] = json.loads(row["reference_clips"])
        for path in clips:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass
        conn.execute(
            "UPDATE enrolled_speakers SET reference_clips = '[]', avg_confidence = NULL WHERE id = ?",
            (speaker_id,),
        )


def _model_settings_response(raw: dict[str, str]) -> ModelSettingsResponse:
    return ModelSettingsResponse(
        stt_provider=raw.get("stt_provider", "local"),
        stt_model=raw.get("stt_model", "ibm-granite/granite-4.0-1b-speech"),
        stt_base_url=raw.get("stt_base_url", ""),
        stt_api_key=raw.get("stt_api_key", ""),
        stt_prompt=raw.get("stt_prompt", "<|audio|>can you transcribe the speech into a written format?"),
        speaker_id_model=raw.get("speaker_id_model", "voxceleb_resnet34_LM.onnx"),
    )


def _settings_response(raw: dict[str, str]) -> SettingsResponse:
    return SettingsResponse(
        pending_retention_days=int(raw.get("pending_retention_days", 7)),
        resolved_retention_days=int(raw.get("resolved_retention_days", 3)),
        reference_clips_per_speaker=int(raw.get("reference_clips_per_speaker", 5)),
        match_threshold=float(raw.get("match_threshold", 0.50)),
        confirm_threshold=float(raw.get("confirm_threshold", 0.75)),
        personality_processing=raw.get("personality_processing", "false").lower() == "true",
        ha_base_url=raw.get("ha_base_url", HA_BASE_URL),
        ha_webhook_enabled=raw.get("ha_webhook_enabled", HA_WEBHOOK_ENABLED).lower() == "true",
    )


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── DB init (idempotent) ────────────────────────────────────────────
    await asyncio.to_thread(init_db, DB_PATH)

    # ── Read model settings from DB ─────────────────────────────────────
    raw_settings = await asyncio.to_thread(_fetch_settings)
    ms = _model_settings_response(raw_settings)

    # ── Load ML models in parallel (each handles its own download/cache) ─
    logger.info("Loading pipeline models — first run may download several GB…")
    ns, vad, sid, stt = await asyncio.gather(
        asyncio.to_thread(NoiseSuppressor,       MODEL_DIR),
        asyncio.to_thread(VoiceActivityDetector, MODEL_DIR),
        asyncio.to_thread(SpeakerIdentifier,     MODEL_DIR, DB_PATH, ms.speaker_id_model),
        asyncio.to_thread(SpeechToText,          MODEL_DIR, ms.stt_provider,
                          ms.stt_model, ms.stt_base_url, ms.stt_api_key, ms.stt_prompt),
    )
    logger.info(
        "Models ready — noise_suppressor=%s  vad=%s  speaker_id=%s  stt=%s",
        ns.available, vad.available, sid.available, stt.available,
    )

    # ── Build orchestrator ──────────────────────────────────────────────
    writer       = InteractionWriter(db_path=DB_PATH, audio_dir=AUDIO_DIR,
                                     embedding_dir=EMBEDDING_DIR)
    orchestrator = VoicePipelineOrchestrator(ns, vad, sid, stt, writer)
    app.state.sid         = sid
    app.state.orchestrator = orchestrator
    app.state.model_status = {"status": "ready"}

    # ── HA webhook notifier ─────────────────────────────────────────────
    notifier = WebhookNotifier(db_path=DB_PATH)

    # ── Wyoming STT server ──────────────────────────────────────────────
    wyoming      = WyomingServer(orchestrator, notifier, host=WYOMING_HOST, port=WYOMING_PORT)
    await wyoming.start()
    wyoming_task = asyncio.create_task(wyoming.serve_forever())

    # ── Retention + speaker-cache scheduler ────────────────────────────
    scheduler = AsyncIOScheduler()
    scheduler.add_job(RetentionJob(db_path=DB_PATH).run, "interval", hours=6)
    scheduler.add_job(sid.refresh_cache, "interval", minutes=5)
    scheduler.start()
    logger.info("Retention scheduler started")

    yield

    # ── Shutdown ────────────────────────────────────────────────────────
    wyoming_task.cancel()
    try:
        await wyoming_task
    except asyncio.CancelledError:
        pass
    await wyoming.stop()
    await notifier.close()
    scheduler.shutdown(wait=False)
    logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(title="VoicePipeline Speaker Management API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:8000",
        "http://127.0.0.1",
        "http://127.0.0.1:8000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static audio files — direct browser access at /audio/<filename>.wav
Path(AUDIO_DIR).mkdir(parents=True, exist_ok=True)
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio_static")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

api = APIRouter(prefix="/api/v1")


@api.get("/speakers", response_model=list[EnrolledSpeakerOut])
async def get_speakers():
    return await asyncio.to_thread(_fetch_speakers)


@api.delete("/speakers/{speaker_id}", status_code=204)
async def delete_speaker(speaker_id: str, request: Request):
    try:
        await asyncio.to_thread(_delete_speaker, speaker_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    request.app.state.sid.force_refresh()


@api.post("/speakers/{speaker_id}/clear-clips", status_code=204)
async def clear_speaker_clips(speaker_id: str, request: Request):
    try:
        await asyncio.to_thread(_clear_speaker_clips, speaker_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    request.app.state.sid.force_refresh()


@api.get("/interactions/pending", response_model=list[PendingInteraction])
async def get_pending():
    return await asyncio.to_thread(_fetch_pending)


@api.get("/interactions/history", response_model=list[HistoryInteraction])
async def get_history(
    speaker: str | None = Query(default=None),
    date_from: str | None = Query(default=None, description="ISO-8601 lower bound"),
    date_to: str | None = Query(default=None, description="ISO-8601 upper bound"),
):
    return await asyncio.to_thread(_fetch_history, speaker, date_from, date_to)


@api.get("/audio/{interaction_id}")
async def get_audio(interaction_id: str):
    def _fetch_path() -> str | None:
        with _get_connection() as conn:
            row = conn.execute(
                "SELECT audio_path FROM interactions WHERE id = ?", (interaction_id,)
            ).fetchone()
            if row is None:
                raise LookupError("not_found")
            return row["audio_path"]

    try:
        audio_path = await asyncio.to_thread(_fetch_path)
    except LookupError:
        raise HTTPException(status_code=404, detail="Interaction not found")

    if not audio_path:
        raise HTTPException(
            status_code=404, detail="Audio file has been deleted by retention policy"
        )

    path = Path(audio_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found on disk")

    return FileResponse(path, media_type="audio/wav")


@api.post("/interactions/{interaction_id}/resolve", response_model=ResolveResponse)
async def resolve_interaction(interaction_id: str, body: ResolveRequest, request: Request):
    if body.action in ("confirm", "assign") and not body.assigned_to:
        raise HTTPException(
            status_code=422,
            detail=f"assigned_to is required for action '{body.action}'",
        )

    try:
        result = await asyncio.to_thread(_do_resolve, interaction_id, body.action, body.assigned_to)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if body.action in ("enrol", "confirm", "assign"):
        request.app.state.sid.force_refresh()

    return result


@api.get("/settings", response_model=SettingsResponse)
async def get_settings():
    raw = await asyncio.to_thread(_fetch_settings)
    return _settings_response(raw)


@api.put("/settings", response_model=SettingsResponse)
async def update_settings(body: SettingsUpdate):
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raw = await asyncio.to_thread(_fetch_settings)
    else:
        raw = await asyncio.to_thread(_save_settings, updates)
    return _settings_response(raw)


@api.get("/settings/models", response_model=ModelSettingsResponse)
async def get_model_settings():
    raw = await asyncio.to_thread(_fetch_settings)
    return _model_settings_response(raw)


@api.put("/settings/models", response_model=ModelSettingsResponse)
async def update_model_settings(body: ModelSettingsUpdate):
    updates = body.model_dump(exclude_none=True)
    if updates:
        await asyncio.to_thread(_save_settings, updates)
    raw = await asyncio.to_thread(_fetch_settings)
    return _model_settings_response(raw)


@api.post("/models/reload", response_model=ModelReloadStatus)
async def reload_models(request: Request):
    """Re-initialise STT and speaker ID models from current settings. Runs in background."""
    if request.app.state.model_status.get("status") == "reloading":
        return ModelReloadStatus(status="reloading", detail="Reload already in progress")

    request.app.state.model_status = {"status": "reloading"}

    async def _do_reload():
        try:
            raw = await asyncio.to_thread(_fetch_settings)
            ms  = _model_settings_response(raw)
            new_stt, new_sid = await asyncio.gather(
                asyncio.to_thread(SpeechToText, MODEL_DIR, ms.stt_provider,
                                  ms.stt_model, ms.stt_base_url, ms.stt_api_key, ms.stt_prompt),
                asyncio.to_thread(SpeakerIdentifier, MODEL_DIR, DB_PATH, ms.speaker_id_model),
            )
            request.app.state.orchestrator.swap_stt(new_stt)
            request.app.state.orchestrator.swap_sid(new_sid)
            request.app.state.sid = new_sid
            request.app.state.model_status = {"status": "ready"}
            logger.info("Model reload complete — stt=%s  sid=%s", new_stt.available, new_sid.available)
        except Exception as exc:
            logger.error("Model reload failed: %s", exc)
            request.app.state.model_status = {"status": "error", "detail": str(exc)}

    asyncio.create_task(_do_reload())
    return ModelReloadStatus(status="reloading", detail="Reload started")


@api.get("/models/status", response_model=ModelReloadStatus)
async def get_model_status(request: Request):
    s = request.app.state.model_status
    return ModelReloadStatus(status=s.get("status", "ready"), detail=s.get("detail", ""))


app.include_router(api)

# Serve the static UI — must be mounted after all API routes so the router
# takes precedence. html=True serves index.html for directory requests.
_UI_DIR = Path(__file__).parent.parent / "ui"
if _UI_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_UI_DIR), html=True), name="ui")
