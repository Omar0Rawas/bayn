"""The seizure socket, filled with a MOCK model: shows how a real seizure detector plugs into Bayn.

The model (src/bayn/seizure/assets/model.json, card: docs/SEIZURE_MODEL.md) is a logistic regression trained on
the public CHB-MIT scalp EEG database (children, clinical electrodes, Muse-like channels approximated). It is NOT
validated on Muse data and NOT for clinical use. Its alarm goes through exactly the same path as a gesture:
Detection -> event -> saved EEG window -> transcript -> summary.

Live: 4 s windows every 1 s on TP9/AF7/AF8/TP10 (features identical to training, bayn.seizure.features);
alarm when P >= threshold for `consecutive` windows in a row, then quiet for `refractory_s`.
Off by default (`events.seizure.enabled`); `bayn demo` and `bayn showcase` switch it on.

Replacing it with a real model (e.g. one for absence seizures, trained and validated on the target headset):
  1. Subclass Detector like this class: keep the signal it needs (raw or filtered EEG) in update().
  2. Run the model every step on the latest window -> probability p.
  3. When p stays above the threshold long enough, return [Detection(name, onset, p, kind="seizure")].
  4. Create it in bayn.events.detectors.make_gesture_detectors (here: when events.seizure.enabled).
The event manager, EEG window storage, transcript and summary pipeline then work unchanged; seizure-kind
detections bypass the gesture cooldown and are labelled "automatic". Any real seizure detection needs clinical
validation and regulatory clearance.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from bayn.common.schemas import EEGChunk
from bayn.events.detectors.base import Detection, Detector

log = logging.getLogger("bayn.events")
MODEL_PATH = Path(__file__).resolve().parents[2] / "seizure" / "assets" / "model.json"
DISCLAIMER = "mock model trained on CHB-MIT, not validated on Muse data, not for clinical use"


def load_model(path: Path | None = None) -> dict | None:
    try:
        return json.loads(Path(path or MODEL_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def probability(model: dict, rows: np.ndarray) -> np.ndarray:
    z = ((rows - np.array(model["mean"])) / np.array(model["std"])) @ np.array(model["coef"]) + model["intercept"]
    return 1 / (1 + np.exp(-z))


class SeizureModelDetector(Detector):
    name = "seizure_model"
    label = "Seizure-like pattern (mock model)"
    kind = "seizure"

    def __init__(self, fs: float, channels: list[str], threshold: float | None = None,
                 model_path: Path | None = None) -> None:
        super().__init__(fs, channels)
        self.model = load_model(model_path)
        need = (self.model or {}).get("channels", ["TP9", "AF7", "AF8", "TP10"])
        self.idx = [channels.index(c) for c in need if c in channels]
        self.usable = self.model is not None and len(self.idx) == len(need) and abs(fs - self.model.get("fs", fs)) < 1
        if self.model is None:
            log.warning("seizure model not found (%s) - the seizure detector will never fire", model_path or MODEL_PATH)
        self.threshold = float(threshold if threshold is not None else (self.model or {}).get("threshold", 0.9))
        self.consecutive = int((self.model or {}).get("consecutive", 5))
        self.refractory_s = float((self.model or {}).get("refractory_s", 120.0))
        self.reset()

    def reset(self) -> None:
        from bayn.eeg.filters import StreamingFilter
        from bayn.seizure.features import BAND, FeatureStream

        self._filter = StreamingFilter(self.fs, len(self.idx) or 1, BAND, notch="auto", order=4)
        self._stream = FeatureStream(self.fs, len(self.idx) or 1)
        self._t_start: float | None = None
        self._run = 0
        self._run_start = 0.0
        self._last_alarm = -1e18
        self.p = 0.0

    def update(self, raw: EEGChunk, filt: EEGChunk) -> list[Detection]:
        if not self.usable:
            return []
        if self._t_start is None:
            self._t_start = raw.t0
        first_end = self._stream.next_end                     # absolute sample index of the next window end
        rows = self._stream.push(self._filter.process(raw.data[self.idx]))
        if not rows:
            return []
        probs = probability(self.model, np.array(rows))
        out = []
        for k, p in enumerate(probs):
            t_end = self._t_start + (first_end + k * self._stream.step) / self.fs
            self.p = float(p)
            if not self._stream.has_baseline:                  # needs ~40 s of signal to know "normal"
                self._run = 0
                continue
            if p >= self.threshold:
                if self._run == 0:
                    self._run_start = t_end - self._stream.win / self.fs / 2
                self._run += 1
            else:
                self._run = 0
            if self._run >= self.consecutive and t_end - self._last_alarm >= self.refractory_s:
                self._last_alarm = t_end
                out.append(Detection(self.name, self._run_start, float(p), kind="seizure", info={
                    "p": round(float(p), 3), "threshold": self.threshold, "model": "CHB-MIT mock (logistic)",
                    "disclaimer": DISCLAIMER}))
        return out

    def status(self) -> dict:
        return {"name": self.name, "label": self.label, "level": round(100 * self.p, 1),
                "threshold": round(100 * self.threshold, 1), "unit": "% P(seizure-like)",
                "active": self._run > 0, "usable": self.usable,
                "baseline_ready": self._stream.has_baseline, "disclaimer": DISCLAIMER}
