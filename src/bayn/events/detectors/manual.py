"""Manual trigger (dashboard button or hotkey) as a Detector, so every event source
goes through the same path: detection -> cooldown/debounce -> event -> EEG window."""

from __future__ import annotations

import time

from bayn.common.schemas import EEGChunk
from bayn.events.detectors.base import Detection, Detector


class ManualDetector(Detector):
    name = "manual"
    label = "Manual"
    kind = "manual"

    def __init__(self, fs: float = 256.0, channels: list[str] | None = None) -> None:
        super().__init__(fs, channels or [])
        self._pending: list[Detection] = []

    def fire(self, source: str = "button", t: float | None = None) -> None:
        """Queue a trigger; it is returned by the next update() (or drain())."""
        self._pending.append(Detection(self.name, t if t is not None else time.time(), 1.0,
                                       kind="manual", info={"source": source}))

    def drain(self) -> list[Detection]:
        out, self._pending = self._pending, []
        return out

    def update(self, raw: EEGChunk, filt: EEGChunk) -> list[Detection]:
        return self.drain()

    def status(self) -> dict:
        return {"name": self.name, "label": self.label, "level": None, "threshold": None, "unit": ""}
