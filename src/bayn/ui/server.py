"""Dashboard server.

Subscribes to the bus and forwards messages to every connected browser over one
WebSocket. Static assets are served from ./static (no CDN, works offline).
Browser actions (trigger, calibrate, gesture choice, simulator) become bus commands.

    python -m bayn.ui.server
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from bayn import __version__
from bayn.common.bus import CommandSender, ThreadedSubscriber
from bayn.common.config import AppConfig, load_config
from bayn.common.log import setup_logging
from bayn.common.schemas import (GESTURES, TOPIC_AUDIO_LEVEL, TOPIC_AUDIO_STATUS, TOPIC_CALIBRATION,
                                 TOPIC_EEG_BANDS, TOPIC_EEG_FILT, TOPIC_EEG_INJECTED, TOPIC_EEG_STATUS, TOPIC_EVENT,
                                 TOPIC_EVENT_DONE, TOPIC_EVENTS_STATUS, TOPIC_SUMMARY, TOPIC_SUMMARY_STATUS,
                                 TOPIC_SUMMARY_TOKEN, TOPIC_TRAINING, TOPIC_TRANSCRIPT, TOPIC_TRANSCRIPT_PARTIAL,
                                 TOPIC_TRANSCRIPT_STATUS, TOPIC_TRANSCRIPT_WINDOW)
from bayn.events.detectors import GESTURE_CHOICES
from bayn.events.store import DOWNLOADABLE, EventStore
from bayn.events.training import CONDITIONS, delete_recording, session_summaries
from bayn.transcription.service import LANGUAGES
from bayn.transcription.store import TranscriptStore

log = logging.getLogger("bayn.ui")
STATIC = Path(__file__).parent / "static"


class Hub:
    """Fan-out of bus messages to WebSocket clients."""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self.last_eeg_t = 0.0          # wall time we last received EEG
        self.last_status: dict | None = None
        self.last_events_status: dict | None = None
        self.last_calibration: dict | None = None
        self.last_by_type: dict[str, dict] = {}      # latest status messages, replayed on connect

    async def broadcast(self, msg: dict) -> None:
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json(msg)
            except Exception:  # client went away mid-send
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


def _eeg_to_json(m: dict) -> dict:
    data = np.asarray(m["data"], dtype=np.float32)
    return {"type": "eeg", "t0": m["t0"], "fs": m["fs"], "seq": m.get("seq", 0),
            "data": np.round(data, 1).tolist()}


async def _relay_eeg(cfg: AppConfig, hub: Hub) -> None:
    sub = ThreadedSubscriber(cfg.bus.eeg_host, cfg.bus.eeg_pub_port,
                             [TOPIC_EEG_FILT, TOPIC_EEG_STATUS, TOPIC_EEG_INJECTED, TOPIC_EEG_BANDS],
                             asyncio.get_running_loop())
    try:
        while True:
            topic, m = await sub.queue.get()
            if topic == TOPIC_EEG_FILT:
                hub.last_eeg_t = time.time()
                if hub.clients:
                    await hub.broadcast(_eeg_to_json(m))
            elif topic == TOPIC_EEG_STATUS:
                hub.last_status = {"type": "eeg_status", **m}
                await hub.broadcast(hub.last_status)
            elif topic == TOPIC_EEG_INJECTED:
                await hub.broadcast({"type": "injected", **m})
            elif topic == TOPIC_EEG_BANDS:
                await hub.broadcast({"type": "bands", "t": m["t"], "channels": m["channels"],
                                     "bands": m["bands"],
                                     "relative": np.round(np.asarray(m["relative"]), 4).tolist()})
    finally:
        sub.close()


async def _relay_events(cfg: AppConfig, hub: Hub) -> None:
    sub = ThreadedSubscriber(cfg.bus.events_host, cfg.bus.events_pub_port,
                             [TOPIC_EVENTS_STATUS, TOPIC_EVENT, TOPIC_EVENT_DONE, TOPIC_CALIBRATION, TOPIC_TRAINING],
                             asyncio.get_running_loop())
    kinds = {TOPIC_EVENTS_STATUS: "trigger_status", TOPIC_EVENT: "event",
             TOPIC_EVENT_DONE: "event_done", TOPIC_CALIBRATION: "calibration", TOPIC_TRAINING: "training"}
    try:
        while True:
            topic, m = await sub.queue.get()
            msg = {"type": kinds[topic], **m}
            if topic == TOPIC_EVENTS_STATUS:
                hub.last_events_status = msg
            elif topic == TOPIC_CALIBRATION:
                hub.last_calibration = msg
            elif topic == TOPIC_TRAINING and m.get("state") in ("trained", "train_failed"):
                hub.last_by_type["training"] = msg          # a reloaded page still shows the last result
            await hub.broadcast(msg)
    finally:
        sub.close()


async def _relay(host: str, port: int, kinds: dict[str, str], hub: Hub, keep: set[str] = frozenset()) -> None:
    """Generic relay: topic -> {"type": kinds[topic], **payload}; topics in `keep` are replayed to new clients."""
    sub = ThreadedSubscriber(host, port, list(kinds), asyncio.get_running_loop())
    try:
        while True:
            topic, m = await sub.queue.get()
            msg = {"type": kinds[topic], **m}
            if topic in keep:
                hub.last_by_type[kinds[topic]] = msg
            await hub.broadcast(msg)
    finally:
        sub.close()


async def _heartbeat(hub: Hub) -> None:
    """Lets the UI show 'no signal' when the bridge is down."""
    while True:
        await asyncio.sleep(1.0)
        age = time.time() - hub.last_eeg_t if hub.last_eeg_t else None
        await hub.broadcast({"type": "heartbeat", "eeg_age_s": age, "t": time.time()})


def _log_task_crash(task: asyncio.Task) -> None:
    if not task.cancelled() and task.exception() is not None:
        log.error("background task %s crashed", task.get_name(), exc_info=task.exception())


class InjectRequest(BaseModel):
    gesture: str


class TriggerRequest(BaseModel):
    source: str = "button"          # button | hotkey


class CalibrationRequest(BaseModel):
    action: str = "start"           # start | cancel


class GestureRequest(BaseModel):
    gesture: str


class TrainingRequest(BaseModel):
    action: str = "start"           # start | cancel | retrain
    protocol: str = "standard"      # standard | noisy | boundaries
    condition: str = "seated"       # seated | standing | walking


class SessionDeleteRequest(BaseModel):
    id: str


class BlinkMethodRequest(BaseModel):
    method: str                     # auto | rule | ml


class LanguageRequest(BaseModel):
    language: str


class RegenerateRequest(BaseModel):
    event_id: str
    lang: str | None = None          # en | ar | None (= configured language)


def create_app(cfg: AppConfig | None = None, store: EventStore | None = None) -> FastAPI:
    cfg = cfg or load_config()
    hub = Hub()
    state: dict[str, Any] = {}
    b = cfg.bus

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["eeg_cmd"] = CommandSender(b.eeg_host, b.eeg_cmd_port)
        state["events_cmd"] = CommandSender(b.events_host, b.events_cmd_port)
        state["stt_cmd"] = CommandSender(b.stt_host, b.stt_cmd_port)
        state["sum_cmd"] = CommandSender(b.sum_host, b.sum_cmd_port)
        state["store"] = store or EventStore()
        state["transcripts"] = TranscriptStore(state["store"].root)
        audio_kinds = {TOPIC_AUDIO_LEVEL: "audio_level", TOPIC_AUDIO_STATUS: "audio_status"}
        stt_kinds = {TOPIC_TRANSCRIPT: "transcript", TOPIC_TRANSCRIPT_PARTIAL: "transcript_partial",
                     TOPIC_TRANSCRIPT_STATUS: "transcript_status", TOPIC_TRANSCRIPT_WINDOW: "transcript_window"}
        tasks = [asyncio.create_task(_relay_eeg(cfg, hub)), asyncio.create_task(_relay_events(cfg, hub)),
                 asyncio.create_task(_relay(b.audio_host, b.audio_pub_port, audio_kinds, hub, {TOPIC_AUDIO_STATUS})),
                 asyncio.create_task(_relay(b.stt_host, b.stt_pub_port, stt_kinds, hub, {TOPIC_TRANSCRIPT_STATUS})),
                 asyncio.create_task(_relay(b.sum_host, b.sum_pub_port,
                                            {TOPIC_SUMMARY_TOKEN: "summary_token", TOPIC_SUMMARY: "summary",
                                             TOPIC_SUMMARY_STATUS: "summary_status"}, hub, {TOPIC_SUMMARY_STATUS})),
                 asyncio.create_task(_heartbeat(hub))]
        for t in tasks:
            t.add_done_callback(_log_task_crash)
        log.info("dashboard on http://%s:%d", cfg.ui.host, cfg.ui.port)
        yield
        for t in tasks:
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t
        for k in ("eeg_cmd", "events_cmd", "stt_cmd", "sum_cmd"):
            state[k].close()

    app = FastAPI(title="Bayn", version=__version__, lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def _hello() -> dict:
        return {"type": "hello", "version": __version__, "source": cfg.eeg.source,
                "channels": cfg.eeg.channels, "fs": cfg.eeg.fs, "plot_seconds": cfg.ui.plot_seconds,
                "notch": cfg.eeg.notch_hz, "bandpass": list(cfg.eeg.bandpass_hz), "gestures": list(GESTURES),
                "trigger_gestures": list(GESTURE_CHOICES), "pre_s": cfg.events.pre_s, "post_s": cfg.events.post_s,
                "languages": list(LANGUAGES), "audio_source": cfg.audio.source}

    def _send_events(payload: dict) -> None:
        if not state["events_cmd"].send(payload):
            raise HTTPException(503, "events service not reachable")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    async def health() -> dict:
        age = time.time() - hub.last_eeg_t if hub.last_eeg_t else None
        return {"ok": True, "eeg_age_s": age, "clients": len(hub.clients)}

    @app.get("/api/config")
    async def config() -> dict:
        return _hello()

    @app.post("/api/sim/inject", status_code=202)
    async def inject(req: InjectRequest) -> dict:
        if req.gesture not in GESTURES:
            raise HTTPException(400, f"gesture must be one of {GESTURES}")
        if not state["eeg_cmd"].send({"cmd": "inject", "gesture": req.gesture}):
            raise HTTPException(503, "EEG bridge not reachable")
        return {"queued": req.gesture}

    # ---------------------------------------------------------------- events
    @app.post("/api/events/trigger", status_code=202)
    async def trigger(req: TriggerRequest) -> dict:
        source = req.source if req.source in ("button", "hotkey") else "button"
        _send_events({"cmd": "trigger", "source": source})
        return {"queued": source}

    @app.post("/api/events/calibration", status_code=202)
    async def calibration(req: CalibrationRequest) -> dict:
        if req.action not in ("start", "cancel"):
            raise HTTPException(400, "action must be start or cancel")
        _send_events({"cmd": "calibrate", "action": req.action})
        return {"queued": req.action}

    @app.post("/api/events/training", status_code=202)
    async def training(req: TrainingRequest) -> dict:
        if req.action not in ("start", "cancel", "retrain") or req.protocol not in ("standard", "noisy", "boundaries") \
                or req.condition not in CONDITIONS:
            raise HTTPException(400, "action: start | cancel | retrain; protocol: standard | noisy | boundaries; "
                                     "condition: seated | standing | walking")
        _send_events({"cmd": "train", "action": req.action, "protocol": req.protocol, "condition": req.condition})
        return {"queued": req.action}

    def _eeg_source() -> str:
        return (hub.last_status or {}).get("source") or cfg.eeg.source

    @app.get("/api/events/training/sessions")
    async def training_sessions() -> dict:
        """Saved recording sessions for this profile + headset (re-record: delete one, record it again)."""
        return {"profile": cfg.events.profile, "source": _eeg_source(),
                "sessions": session_summaries(cfg.events.profile, _eeg_source())}

    @app.post("/api/events/training/delete")
    async def training_delete(req: SessionDeleteRequest) -> dict:
        if not delete_recording(cfg.events.profile, _eeg_source(), req.id):
            raise HTTPException(404, "no such session")
        return {"deleted": req.id}

    @app.post("/api/events/blink_method", status_code=202)
    async def blink_method(req: BlinkMethodRequest) -> dict:
        if req.method not in ("auto", "rule", "ml"):
            raise HTTPException(400, "method must be auto, rule or ml")
        _send_events({"cmd": "set_blink_method", "method": req.method})
        return {"queued": req.method}

    @app.post("/api/events/gesture", status_code=202)
    async def gesture(req: GestureRequest) -> dict:
        if req.gesture not in GESTURE_CHOICES:
            raise HTTPException(400, f"gesture must be one of {GESTURE_CHOICES}")
        _send_events({"cmd": "set_gesture", "gesture": req.gesture})
        return {"queued": req.gesture}

    @app.post("/api/transcription/language", status_code=202)
    async def language(req: LanguageRequest) -> dict:
        if req.language not in LANGUAGES:
            raise HTTPException(400, f"language must be one of {LANGUAGES}")
        if not state["stt_cmd"].send({"cmd": "set_language", "language": req.language}):
            raise HTTPException(503, "transcription service not reachable")
        return {"queued": req.language}

    @app.post("/api/summary/regenerate", status_code=202)
    async def regenerate(req: RegenerateRequest) -> dict:
        try:
            state["store"].dir_for(req.event_id)
        except ValueError:
            raise HTTPException(400, "bad event id") from None
        if not state["sum_cmd"].send({"cmd": "regenerate", "event_id": req.event_id, "lang": req.lang}):
            raise HTTPException(503, "summarizer not reachable")
        return {"queued": req.event_id}

    @app.post("/api/summary/language", status_code=202)
    async def summary_language(req: LanguageRequest) -> dict:
        if req.language not in ("auto", "en", "ar", "both"):
            raise HTTPException(400, "language must be auto, en, ar or both")
        if not state["sum_cmd"].send({"cmd": "set_language", "language": req.language}):
            raise HTTPException(503, "summarizer not reachable")
        return {"queued": req.language}

    @app.get("/api/transcript")
    async def transcript(limit: int = 50) -> list[dict]:
        return state["transcripts"].recent(min(max(limit, 1), 500))

    @app.get("/api/events")
    async def list_events(limit: int = 50) -> list[dict]:
        return state["store"].list(min(max(limit, 1), 500))

    @app.get("/api/events/{event_id}/{name}")
    async def event_file(event_id: str, name: str) -> FileResponse:
        if name not in DOWNLOADABLE:
            raise HTTPException(404)
        try:
            path = state["store"].dir_for(event_id) / name
        except ValueError:
            raise HTTPException(404) from None
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, filename=f"{event_id}-{name}")

    # ---------------------------------------------------------------- websocket
    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        hub.clients.add(ws)
        await ws.send_json(_hello())
        for last in (hub.last_status, hub.last_events_status, *hub.last_by_type.values()):
            if last:
                await ws.send_json(last)
        await ws.send_json({"type": "event_history", "events": state["store"].list(30)})
        await ws.send_json({"type": "transcript_history", "segments": state["transcripts"].recent(40)})
        try:
            while True:
                await ws.receive_text()   # client messages unused; keeps socket open
        except WebSocketDisconnect:
            pass
        finally:
            hub.clients.discard(ws)

    return app


def main() -> None:
    cfg = load_config()
    setup_logging(cfg.logging.level)
    uvicorn.run(create_app(cfg), host=cfg.ui.host, port=cfg.ui.port,
                log_level=cfg.logging.level.lower(), access_log=False)


if __name__ == "__main__":
    main()
