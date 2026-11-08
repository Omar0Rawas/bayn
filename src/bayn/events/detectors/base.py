"""The Detector interface: anything that can decide "an event happened" from EEG.

Today: deliberate gestures (jaw clench, double blink) and a manual button.
Later: a real seizure detector implements the same interface and is dropped in
via config, without touching the event manager, storage, UI or pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from bayn.common.schemas import EEGChunk


@dataclass(slots=True)
class Detection:
    detector: str              # detector name, e.g. "clench"
    t: float                   # onset, unix seconds (data time, not wall clock)
    confidence: float = 1.0    # 0..1 (gestures: how far above threshold, capped)
    kind: str = "gesture"      # gesture | manual | seizure
    info: dict = field(default_factory=dict)


@dataclass(slots=True)
class CalibrationResult:
    threshold: float
    rest_level: float
    gesture_level: float
    separation: float          # gesture_level / rest_level
    quality: str               # good | weak | failed
    message: str = ""

    def to_dict(self) -> dict:
        return {"threshold": self.threshold, "rest_level": self.rest_level,
                "gesture_level": self.gesture_level, "separation": self.separation,
                "quality": self.quality, "message": self.message}


class Detector(ABC):
    """Consumes paired raw + filtered chunks; emits Detections.

    Contract (checked by tests/test_detectors.py for every implementation):
      - update() is called with consecutive chunks and never blocks
      - update() returns a (possibly empty) list of Detection with unix onset times
      - reset() clears internal state (e.g. after a signal gap)
      - status() returns a JSON-able dict with at least {"name", "level", "threshold", "unit"}
    """

    name: str = "base"
    label: str = "Detector"       # human-readable, shown in the UI
    kind: str = "gesture"
    calibratable: bool = False

    def __init__(self, fs: float, channels: list[str]) -> None:
        self.fs = float(fs)
        self.channels = list(channels)

    @abstractmethod
    def update(self, raw: EEGChunk, filt: EEGChunk) -> list[Detection]: ...

    def reset(self) -> None:
        pass

    def status(self) -> dict:
        return {"name": self.name, "label": self.label, "level": None, "threshold": None, "unit": ""}

    # -- calibration (optional)
    def calibrate(self, rest_raw: np.ndarray, gesture_raws: list[np.ndarray]) -> CalibrationResult:
        raise NotImplementedError(f"{self.name} cannot be calibrated")

    def apply_profile(self, params: dict) -> None:
        """Load calibrated parameters (from a saved profile)."""

    def _ch(self, name: str) -> int:
        return self.channels.index(name)
