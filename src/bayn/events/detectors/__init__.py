"""Detector implementations and a factory driven by `events.gesture`."""

from __future__ import annotations

from bayn.common.config import EventsConfig
from bayn.events.detectors.base import CalibrationResult, Detection, Detector
from bayn.events.detectors.clench import ClenchDetector
from bayn.events.detectors.blinks import BlinkCountDetector
from bayn.events.detectors.manual import ManualDetector
from bayn.events.detectors.seizure_model import SeizureModelDetector

GESTURE_CHOICES = ("clench", "triple_blink", "both", "manual")


def make_gesture_detectors(gesture: str, cfg: EventsConfig, fs: float, channels: list[str]) -> list[Detector]:
    """EEG-driven detectors for the chosen gesture ("manual" = none, button/hotkey only)."""
    if gesture not in GESTURE_CHOICES:
        raise ValueError(f"gesture must be one of {GESTURE_CHOICES}")
    out: list[Detector] = []
    if gesture in ("clench", "both"):
        out.append(ClenchDetector(fs, channels, cfg.clench))
    if gesture in ("triple_blink", "both"):
        if cfg.triple_blink.method == "rule":
            out.append(BlinkCountDetector(fs, channels, cfg.triple_blink))
        else:   # the ML detector (falls back to the rule until a model is trained); lazy: avoids a cycle
            from bayn.events.blink_ml import MLBlinkDetector

            out.append(MLBlinkDetector(fs, channels, cfg.triple_blink))
    if cfg.seizure.enabled:              # the seizure socket runs alongside any gesture (or none)
        out.append(SeizureModelDetector(fs, channels, cfg.seizure.threshold))
    return out


__all__ = ["CalibrationResult", "ClenchDetector", "Detection", "Detector", "BlinkCountDetector",
           "GESTURE_CHOICES", "ManualDetector", "SeizureModelDetector",
           "make_gesture_detectors"]
