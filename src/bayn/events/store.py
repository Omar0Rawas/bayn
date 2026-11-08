"""Event storage: SQLite index (data/bayn.db) + one folder per event (data/events/<id>/).

Per event folder:
  eeg.npz    raw + filtered EEG (µV), timestamps, channels, fs, event time
  eeg.csv    filtered EEG as CSV (time relative to the event, one column per sensor)
  meta.json  detector, timing, EEG stats
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path

import numpy as np

from bayn.common.config import DATA_DIR

ID_RE = re.compile(r"^evt-\d{8}-\d{6}-\d{3}$")
FILES = ("eeg.npz", "eeg.csv", "meta.json")


class EventStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or DATA_DIR)
        self.events_dir = self.root / "events"
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "bayn.db"
        self._lock = threading.Lock()
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY, t REAL, detector TEXT, kind TEXT, state TEXT, meta TEXT)""")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=5)

    def dir_for(self, event_id: str) -> Path:
        if not ID_RE.match(event_id):
            raise ValueError(f"bad event id {event_id!r}")
        return self.events_dir / event_id

    def upsert(self, meta: dict) -> None:
        with self._lock, self._connect() as db:
            db.execute("INSERT OR REPLACE INTO events (id, t, detector, kind, state, meta) VALUES (?,?,?,?,?,?)",
                       (meta["id"], meta["t"], meta["detector"], meta["kind"], meta["state"], json.dumps(meta)))

    def save_window(self, meta: dict, raw: np.ndarray, filt: np.ndarray, ts: np.ndarray,
                    channels: list[str], fs: float) -> dict:
        """Write the EEG window + metadata; returns the file map added to meta."""
        d = self.dir_for(meta["id"])
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "eeg.npz", raw=raw.astype(np.float32), filt=filt.astype(np.float32),
                            ts=ts, channels=np.array(channels), fs=fs, t_event=meta["t"])
        rel = ts - meta["t"]
        header = "t_rel_s," + ",".join(f"{c}_uv" for c in channels)
        np.savetxt(d / "eeg.csv", np.column_stack([rel, filt.T]), delimiter=",", header=header,
                   comments="", fmt=["%.4f"] + ["%.2f"] * len(channels))
        files = {name: f"events/{meta['id']}/{name}" for name in FILES}
        meta = {**meta, "files": files}
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self.upsert(meta)
        return files

    def list(self, limit: int = 50) -> list[dict]:
        with self._lock, self._connect() as db:
            rows = db.execute("SELECT meta FROM events ORDER BY t DESC LIMIT ?", (limit,)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def get(self, event_id: str) -> dict | None:
        with self._lock, self._connect() as db:
            row = db.execute("SELECT meta FROM events WHERE id = ?", (event_id,)).fetchone()
        return json.loads(row[0]) if row else None
