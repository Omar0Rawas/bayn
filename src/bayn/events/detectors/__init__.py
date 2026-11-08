"""Detector implementations and a factory driven by `events.gesture`."""

from __future__ import annotations

from bayn.common.config import EventsConfig
from bayn.events.detectors.base import CalibrationResult, Detection, Detector
from bayn.events.detectors.clench import ClenchDetector
from bayn.events.detectors.double_blink import DoubleBlinkDetector
from bayn.events.detectors.manual import ManualDetector
from bayn.events.detectors.seizure_stub import SeizureDetectorStub

GESTURE_CHOICES = ("clench", "double_blink", "both", "manual")


def make_gesture_detectors(gesture: str, cfg: EventsConfig, fs: float, channels: list[str]) -> list[Detector]:
    """EEG-driven detectors for the chosen gesture ("manual" = none, button/hotkey only)."""
    if gesture not in GESTURE_CHOICES:
        raise ValueError(f"gesture must be one of {GESTURE_CHOICES}")
    out: list[Detector] = []
    if gesture in ("clench", "both"):
        out.append(ClenchDetector(fs, channels, cfg.clench))
    if gesture in ("double_blink", "both"):
        out.append(DoubleBlinkDetector(fs, channels, cfg.double_blink))
    return out


__all__ = ["CalibrationResult", "ClenchDetector", "Detection", "Detector", "DoubleBlinkDetector",
           "GESTURE_CHOICES", "ManualDetector", "SeizureDetectorStub", "make_gesture_detectors"]
