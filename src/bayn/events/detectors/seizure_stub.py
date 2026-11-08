"""Placeholder for a real seizure detector. NOT IMPLEMENTED - never fires.

How to plug in a real model later:
  1. Subclass Detector (as here); keep a rolling window of `window_s` seconds of
     `filt.data` (or `raw.data`) in update().
  2. Every `step_s`, run the model on the window -> probability p.
  3. If p >= threshold for long enough, return [Detection("seizure", onset, p, kind="seizure")].
  4. Register it in bayn.events.detectors.make_detectors and select it in config.
The event manager, EEG window storage, transcript and summary pipeline then work unchanged.

Any real seizure detection requires clinical validation and regulatory clearance.
This project is a prototype and is not a medical device.
"""

from __future__ import annotations

from bayn.common.schemas import EEGChunk
from bayn.events.detectors.base import Detection, Detector


class SeizureDetectorStub(Detector):
    name = "seizure_stub"
    label = "Seizure detector (not implemented)"
    kind = "seizure"
    window_s = 10.0
    step_s = 1.0

    def update(self, raw: EEGChunk, filt: EEGChunk) -> list[Detection]:
        return []
