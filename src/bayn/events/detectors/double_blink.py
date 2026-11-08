"""Double blink: exactly two blink-shaped peaks 0.2-0.8 s apart on AF7/AF8.

Feature: the average of AF7 and AF8 band-passed to 0.3-10 Hz. Blinks appear with
the same polarity on both forehead sensors and add up; sideways eye movements have
opposite polarity and cancel. A peak counts as a blink when it is short
(< max_width_s); two of them with the right gap, and no third within settle_s, fire.

The signal is kept signed: the band-pass makes every blink "rebound" briefly in the
opposite direction, which must not count as another blink. The polarity of a pair
is set by its first peak, so it works whichever way the headset's blinks point.
"""

from __future__ import annotations

import numpy as np

from bayn.common.config import DoubleBlinkConfig
from bayn.common.schemas import EEGChunk
from bayn.eeg.filters import StreamingFilter
from bayn.events.detectors.base import CalibrationResult, Detection, Detector

FRONTAL = ("AF7", "AF8")
RELEASE = 0.5              # a peak ends when the signal falls below 50 % of threshold


class DoubleBlinkDetector(Detector):
    name = "double_blink"
    label = "Double blink"
    calibratable = True

    def __init__(self, fs: float, channels: list[str], cfg: DoubleBlinkConfig | None = None) -> None:
        super().__init__(fs, channels)
        self.cfg = cfg or DoubleBlinkConfig()
        self.idx = [self._ch(c) for c in FRONTAL if c in channels] or [0]
        self.calibrated_threshold: float | None = self.cfg.threshold_uv
        self.reset()

    def _new_filter(self) -> StreamingFilter:
        return StreamingFilter(self.fs, 1, tuple(self.cfg.band_hz), notch="off", order=2)

    def reset(self) -> None:
        self._filter = self._new_filter()
        self._in_peak = False
        self._sign = 1.0                               # polarity of the peak in progress
        self._peak_start = 0.0
        self._peak_t = 0.0
        self._peak_amp = 0.0
        self._peaks: list[tuple[float, float, float]] = []   # (time, amplitude, sign) of blinks
        self.level = 0.0

    @property
    def threshold(self) -> float:
        return float(self.calibrated_threshold or self.cfg.default_threshold_uv)

    def _signal(self, raw: np.ndarray) -> np.ndarray:
        return self._filter.process(raw[self.idx].mean(axis=0, keepdims=True))[0]

    # ------------------------------------------------------------ detection
    def update(self, raw: EEGChunk, filt: EEGChunk) -> list[Detection]:
        x = self._signal(raw.data)
        ts = raw.timestamps()
        thr = self.threshold
        out: list[Detection] = []
        for t, v in zip(ts, x):
            t, v = float(t), float(v)
            if not self._in_peak:
                if abs(v) >= thr:
                    self._in_peak, self._sign = True, (1.0 if v > 0 else -1.0)
                    self._peak_start, self._peak_t, self._peak_amp = t, t, abs(v)
                else:
                    out += self._evaluate(t)
            else:
                s = self._sign * v                       # signal in this peak's direction
                if s > self._peak_amp:
                    self._peak_t, self._peak_amp = t, s
                if s < RELEASE * thr:                    # back down (or swung the other way)
                    self._in_peak = False
                    if t - self._peak_start <= self.cfg.max_width_s:
                        self._add_peak(self._peak_t, self._peak_amp, self._sign)
                    else:                                # long excursion = movement, not a blink
                        self._peaks.clear()
        self.level = float(np.abs(x).max()) if len(x) else 0.0
        return out

    def _add_peak(self, t: float, amp: float, sign: float) -> None:
        if self._peaks:
            pt, pa, ps = self._peaks[-1]
            if sign != self._peaks[0][2]:
                return                                   # rebound of the previous blink
            if t - pt < self.cfg.min_gap_s:              # same blink split in two: keep the larger
                self._peaks[-1] = (pt, pa, ps) if pa >= amp else (t, amp, sign)
                return
        self._peaks.append((t, amp, sign))

    def _evaluate(self, now: float) -> list[Detection]:
        if not self._peaks:
            return []
        since_last = now - self._peaks[-1][0]
        # one blink: wait for a possible second one; two: wait to rule out a third
        wait = self.cfg.max_gap_s + 0.05 if len(self._peaks) == 1 else self.cfg.settle_s
        if since_last < wait:
            return []
        peaks, self._peaks = self._peaks, []
        if len(peaks) != 2:
            return []                                     # single or triple blink: ignore
        (t1, a1, _), (t2, a2, _) = peaks
        gap = t2 - t1
        if not (self.cfg.min_gap_s <= gap <= self.cfg.max_gap_s):
            return []
        conf = min(1.0, min(a1, a2) / (2 * self.threshold))
        return [Detection(self.name, t1, conf, info={"gap_s": round(gap, 3),
                                                     "amplitudes_uv": [round(a1, 1), round(a2, 1)],
                                                     "threshold_uv": self.threshold})]

    def status(self) -> dict:
        return {"name": self.name, "label": self.label, "level": round(self.level, 1),
                "threshold": round(self.threshold, 1), "unit": "µV",
                "calibrated": self.calibrated_threshold is not None,
                "active": self._in_peak or bool(self._peaks)}

    # ------------------------------------------------------------ calibration
    def feature_trace(self, raw: np.ndarray) -> np.ndarray:
        """|filtered AF7/AF8 average| over a stored segment (fresh filter state)."""
        f = self._new_filter()
        return np.abs(f.process(raw[self.idx].mean(axis=0, keepdims=True))[0])

    def calibrate(self, rest_raw: np.ndarray, gesture_raws: list[np.ndarray]) -> CalibrationResult:
        warm = int(self.fs)
        rest = self.feature_trace(rest_raw)[warm:]
        rest_level = float(np.percentile(rest, 99))       # includes natural blinks during rest
        peaks = [float(self.feature_trace(g)[warm:].max()) for g in gesture_raws if g.shape[1] > warm]
        if not peaks or rest_level <= 0:
            return CalibrationResult(self.threshold, rest_level, 0.0, 0.0, "failed", "no gesture data")
        gesture_level = float(np.median(peaks))
        sep = gesture_level / rest_level
        # between rest and the deliberate blinks, but low enough that both blinks of a pair pass
        thr = min(float(np.sqrt(rest_level * gesture_level)), 0.7 * gesture_level)
        quality = "good" if sep >= 1.8 else "weak" if sep >= 1.2 else "failed"
        msg = {"good": "deliberate blinks clearly stronger than resting signal",
               "weak": "blink a bit harder, and keep your eyes still between prompts",
               "failed": "blinks not distinguishable - check AF7/AF8 forehead contact"}[quality]
        if quality != "failed":
            self.calibrated_threshold = thr
        return CalibrationResult(thr, rest_level, gesture_level, sep, quality, msg)

    def apply_profile(self, params: dict) -> None:
        if params.get("threshold"):
            self.calibrated_threshold = float(params["threshold"])
