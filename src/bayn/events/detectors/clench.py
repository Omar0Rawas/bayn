"""Jaw clench: a sustained burst of muscle activity (EMG) on TP9/TP10.

Feature: RMS envelope of the raw signal band-passed to 55-100 Hz (above brain
activity, with 50/60 Hz mains and harmonics notched out), averaged over TP9 and TP10.
Fires when the envelope stays above threshold for `hold_s` (talking or a quick
swallow is shorter). Needs to drop back below threshold before it can fire again.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from bayn.common.config import ClenchConfig
from bayn.common.schemas import EEGChunk
from bayn.eeg.filters import StreamingFilter
from bayn.events.detectors.base import CalibrationResult, Detection, Detector

TEMPORAL = ("TP9", "TP10")
HYSTERESIS = 0.7           # must fall below 70 % of threshold to re-arm
BASELINE_S = 30.0          # adaptive baseline memory


class ClenchDetector(Detector):
    name = "clench"
    label = "Jaw clench"
    calibratable = True

    def __init__(self, fs: float, channels: list[str], cfg: ClenchConfig | None = None) -> None:
        super().__init__(fs, channels)
        self.cfg = cfg or ClenchConfig()
        self.idx = [self._ch(c) for c in TEMPORAL if c in channels] or [0]
        self.calibrated_threshold: float | None = self.cfg.threshold_uv
        self.reset()

    # ------------------------------------------------------------ feature
    def _new_filter(self) -> StreamingFilter:
        return StreamingFilter(self.fs, len(self.idx), tuple(self.cfg.band_hz), notch="auto",
                               extra_notches=(100.0, 120.0))

    def reset(self) -> None:
        self._filter = self._new_filter()
        self._win = max(1, int(self.cfg.window_s * self.fs))
        self._tail = np.zeros(0)                  # last win-1 squared samples (for the moving average)
        self._baseline: deque[float] = deque(maxlen=int(BASELINE_S * self.fs / 32) + 1)
        self._above_since: float | None = None
        self._armed = True
        self.level = 0.0

    def _envelope(self, raw: np.ndarray) -> np.ndarray:
        """Per-sample RMS envelope (µV) for a chunk, continuous across chunks."""
        y = self._filter.process(raw[self.idx])
        sq = np.mean(y.astype(np.float64) ** 2, axis=0)
        ext = np.concatenate([self._tail, sq])
        csum = np.cumsum(np.concatenate([[0.0], ext]))
        n_out = len(sq)
        start = len(ext) - n_out
        lo = np.maximum(np.arange(start, start + n_out) - self._win + 1, 0)
        hi = np.arange(start, start + n_out) + 1
        ma = (csum[hi] - csum[lo]) / (hi - lo)
        self._tail = ext[-(self._win - 1):] if self._win > 1 else np.zeros(0)
        return np.sqrt(ma)

    @property
    def threshold(self) -> float:
        if self.calibrated_threshold:
            return float(self.calibrated_threshold)
        base = float(np.median(self._baseline)) if self._baseline else 0.0
        return max(self.cfg.min_threshold_uv, self.cfg.baseline_k * base)

    # ------------------------------------------------------------ detection
    def update(self, raw: EEGChunk, filt: EEGChunk) -> list[Detection]:
        env = self._envelope(raw.data)
        ts = raw.timestamps()
        thr = self.threshold
        out: list[Detection] = []
        for t, e in zip(ts, env):
            if e >= thr:
                if self._above_since is None:
                    self._above_since = float(t)
                if self._armed and t - self._above_since >= self.cfg.hold_s:
                    out.append(Detection(self.name, self._above_since, min(1.0, e / (2 * thr)),
                                         info={"level_uv": float(e), "threshold_uv": thr}))
                    self._armed = False
            else:
                self._above_since = None
                if e < HYSTERESIS * thr:
                    self._armed = True
        med = float(np.median(env))
        if self._above_since is None:           # learn the resting level only while relaxed
            self._baseline.append(med)
        self.level = float(env.max())
        return out

    def status(self) -> dict:
        return {"name": self.name, "label": self.label, "level": round(self.level, 2),
                "threshold": round(self.threshold, 2), "unit": "µV",
                "calibrated": self.calibrated_threshold is not None,
                "active": self._above_since is not None}

    # ------------------------------------------------------------ calibration
    def feature_trace(self, raw: np.ndarray) -> np.ndarray:
        """Envelope over a stored segment (fresh filter state)."""
        saved = (self._filter, self._tail)
        self._filter, self._tail = self._new_filter(), np.zeros(0)
        try:
            return self._envelope(raw)
        finally:
            self._filter, self._tail = saved

    def calibrate(self, rest_raw: np.ndarray, gesture_raws: list[np.ndarray]) -> CalibrationResult:
        # segments arrive with 1 s of extra signal in front for filter warm-up
        warm = int(self.fs)
        rest = self.feature_trace(rest_raw)[warm:]
        rest_level = float(np.percentile(rest, 95))
        # per repetition: the level reached during the prompt (the user may start a bit late)
        holds = []
        for g in gesture_raws:
            env = self.feature_trace(g)[warm:]
            if len(env) == 0:
                continue
            holds.append(float(np.percentile(env, 75)))
        if not holds or rest_level <= 0:
            return CalibrationResult(self.threshold, rest_level, 0.0, 0.0, "failed", "no gesture data")
        gesture_level = float(np.median(holds))
        sep = gesture_level / rest_level
        thr = float(np.sqrt(rest_level * gesture_level))     # geometric mean: between the two
        thr = max(thr, self.cfg.min_threshold_uv)
        quality = "good" if sep >= 3 else "weak" if sep >= 1.5 else "failed"
        msg = {"good": "clear difference between rest and clench",
               "weak": "small difference - clench harder or relax more between prompts",
               "failed": "clench not distinguishable from rest - check TP9/TP10 contact"}[quality]
        if quality != "failed":
            self.calibrated_threshold = thr
        return CalibrationResult(thr, rest_level, gesture_level, sep, quality, msg)

    def apply_profile(self, params: dict) -> None:
        if params.get("threshold"):
            self.calibrated_threshold = float(params["threshold"])
