"""Transcript segments in SQLite (same data/bayn.db as events), queryable by time window."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from bayn.common.config import DATA_DIR

COLUMNS = ("id", "utterance_id", "t_start", "t_end", "text", "lang", "lang_prob", "latency_s")


class TranscriptStore:
    def __init__(self, root: Path | None = None) -> None:
        root = Path(root or DATA_DIR)
        root.mkdir(parents=True, exist_ok=True)
        self.db_path = root / "bayn.db"
        self._lock = threading.Lock()
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS segments (
                id TEXT PRIMARY KEY, utterance_id TEXT, t_start REAL, t_end REAL, text TEXT,
                lang TEXT, lang_prob REAL, latency_s REAL)""")
            db.execute("CREATE INDEX IF NOT EXISTS segments_t ON segments (t_start)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=5)

    def add(self, seg: dict) -> None:
        with self._lock, self._connect() as db:
            db.execute(f"INSERT OR REPLACE INTO segments ({','.join(COLUMNS)}) VALUES ({','.join('?' * len(COLUMNS))})",
                       tuple(seg[c] for c in COLUMNS))

    def window(self, t0: float, t1: float) -> list[dict]:
        """Segments overlapping [t0, t1], oldest first."""
        with self._lock, self._connect() as db:
            rows = db.execute(f"SELECT {','.join(COLUMNS)} FROM segments WHERE t_end > ? AND t_start < ? "
                              "ORDER BY t_start", (t0, t1)).fetchall()
        return [dict(zip(COLUMNS, r)) for r in rows]

    def recent(self, limit: int = 50) -> list[dict]:
        with self._lock, self._connect() as db:
            rows = db.execute(f"SELECT {','.join(COLUMNS)} FROM segments ORDER BY t_start DESC LIMIT ?",
                              (limit,)).fetchall()
        return [dict(zip(COLUMNS, r)) for r in reversed(rows)]


def format_transcript(segments: list[dict], t_ref: float) -> str:
    """Plain text with times relative to the event, e.g. '[-08.2 s] (en) text'."""
    return "\n".join(f"[{s['t_start'] - t_ref:+06.1f} s] ({s['lang']}) {s['text']}" for s in segments)
