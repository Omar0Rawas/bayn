"""Dashboard server.

Subscribes to the bus and forwards messages to every connected browser over one
WebSocket. Static assets are served from ./static (no CDN, works offline).

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
from bayn.common.schemas import GESTURES, TOPIC_EEG_FILT, TOPIC_EEG_INJECTED, TOPIC_EEG_STATUS

log = logging.getLogger("bayn.ui")
STATIC = Path(__file__).parent / "static"


class Hub:
    """Fan-out of bus messages to WebSocket clients."""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self.last_eeg_t = 0.0          # wall time we last received EEG
        self.last_status: dict | None = None

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
                             [TOPIC_EEG_FILT, TOPIC_EEG_STATUS, TOPIC_EEG_INJECTED],
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


def create_app(cfg: AppConfig | None = None) -> FastAPI:
    cfg = cfg or load_config()
    hub = Hub()
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["cmd"] = CommandSender(cfg.bus.eeg_host, cfg.bus.eeg_cmd_port)
        tasks = [asyncio.create_task(_relay_eeg(cfg, hub)), asyncio.create_task(_heartbeat(hub))]
        for t in tasks:
            t.add_done_callback(_log_task_crash)
        log.info("dashboard on http://%s:%d", cfg.ui.host, cfg.ui.port)
        yield
        for t in tasks:
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t
        state["cmd"].close()

    app = FastAPI(title="Bayn", version=__version__, lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def _hello() -> dict:
        return {"type": "hello", "version": __version__, "source": cfg.eeg.source,
                "channels": cfg.eeg.channels, "fs": cfg.eeg.fs, "plot_seconds": cfg.ui.plot_seconds,
                "notch": cfg.eeg.notch_hz, "bandpass": list(cfg.eeg.bandpass_hz), "gestures": list(GESTURES)}

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
        ok = state["cmd"].send({"cmd": "inject", "gesture": req.gesture})
        if not ok:
            raise HTTPException(503, "EEG bridge not reachable")
        return {"queued": req.gesture}

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        hub.clients.add(ws)
        await ws.send_json(_hello())
        if hub.last_status:
            await ws.send_json(hub.last_status)
        try:
            while True:
                await ws.receive_text()   # client messages unused for now; keeps socket open
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
