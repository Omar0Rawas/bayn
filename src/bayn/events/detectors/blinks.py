"""Blink-count trigger (threshold rule): exactly N blink-shaped peaks, 0.2-0.8 s apart.

N = `events.triple_blink.count` (3: the user's choice, 2026-11-10; a double blink, a single blink and
four blinks must NOT fire).

Blink signal: weights . channels, band-passed to 0.3-10 Hz. On the user's Muse blinks are largest on
TP9/TP10, because the Muse reference sits on the forehead (docs/TESTING.md, D25). Which pattern is used
(`pattern` in the status), most personal first:
  trained     a blink model trained from the person's labelled sessions (events/blink_ml.py)
  calibrated  learned by Calibrate... from the person's three triple blinks (about a minute)
  built-in    a real headset with Muse channels: the pattern learned from one person's 31 min of sessions
              (events/assets/blink_muse.json), so triple blink works on a fresh install
  front       the AF7/AF8 average (simulator, other headsets)
Checked on the user's 6 real sessions, held out: front 34 % caught / 2.6 false per min; calibrated 66 % /
0.52; trained 67 % / 0.30 (docs/TESTING.md, "Setup without blink training").
A peak counts as a blink when it is short (< max_width_s); N of them with the right gaps, and no further one
within settle_s, fire.

Nod check (optional, `veto_ratio`, learned from the person's sessions): nods and head movement swing the
forehead sensors much more than the ear sensors, blinks don't (user's data: forehead/ear ~1 for blinks,
~4 for nods). A series whose forehead/ear swing ratio exceeds veto_ratio is not a triple blink.

The signal is kept signed: the band-pass makes every blink "rebound" briefly in the opposite direction,
which must not count as another blink. The polarity of a series is set by its first peak, so it works
whichever way the headset's blinks point.

Whether a classifier (events/blink_ml.py) is needed on top of this rule is decided by a pre-registered
test on the person's real recordings: docs/BLINK_DECISION.md.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from bayn.common.config import BlinkConfig
from bayn.common.schemas import EEGChunk
from bayn.eeg.filters import StreamingFilter
from bayn.events.detectors.base import CalibrationResult, Detection, Detector

FRONTAL = ("AF7", "AF8")
EARS = ("TP9", "TP10")
RELEASE = 0.5              # a peak ends when the signal falls below 50 % of threshold...
DIP = 0.6                  # ...or dips below 60 % of its own maximum (fast blinks overlap: no full return)
REARM = 0.25               # after a dip, the signal must rise 25 % of threshold above the valley to be a new peak
MIN_RELATIVE = 0.45        # peaks below 45 % of the series' largest are not counted (stray natural blinks;
                           # later blinks of a series measure smaller, riding on the previous one's rebound)
COUNT_LABELS = {1: "Single blink", 2: "Double blink", 3: "Triple blink"}
VETO_PRE_S, VETO_POST_S = 0.3, 0.4    # window around the series for the forehead/ear ratio
BUILTIN_PATH = Path(__file__).resolve().parents[1] / "assets" / "blink_muse.json"
log = logging.getLogger("bayn.events")


def weights_id(w) -> str:
    """Names a blink signal, so a calibrated threshold is only reused on the signal it was measured on."""
    return "w:" + ",".join(f"{v:+.2f}" for v in np.asarray(w, dtype=np.float64))


def builtin_pattern(channels: list[str], path: Path | None = None) -> tuple[np.ndarray, float] | None:
    """The shipped Muse blink pattern (weights in `channels` order, rule threshold), or None if this
    headset doesn't have the Muse sensors."""
    try:
        b = json.loads(Path(path or BUILTIN_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if sorted(b["channels"]) != sorted(channels):
        return None
    w = dict(zip(b["channels"], b["weights"]))
    return np.array([w[c] for c in channels], dtype=np.float64), float(b["rule_threshold"])


def learn_pattern(segments: list[np.ndarray], fs: float, band: tuple[float, float], warm: int = 0) -> np.ndarray | None:
    """The person's blink pattern from a few deliberate-blink segments (all sensors): the dominant spatial
    pattern of each (first principal component of the 0.3-10 Hz signal, weighted by its strength), signs
    aligned, median, unit length. Same method as blink_ml.learn_weights, on calibration segments. None with
    fewer than 3 usable segments."""
    vecs = []
    for g in segments:
        x = StreamingFilter(fs, g.shape[0], band, notch="off", order=2).process(np.asarray(g, np.float64))
        x = np.asarray(x[:, warm:], np.float64)
        if x.shape[1] < 10:
            continue
        x = x - x.mean(axis=1, keepdims=True)
        u = np.linalg.svd(x, full_matrices=False)[0][:, 0]
        vecs.append(u * np.sqrt(np.sum((u @ x) ** 2) / x.shape[1]))
    if len(vecs) < 3:
        return None
    vecs = np.array(vecs)
    vecs *= np.sign(vecs @ vecs[np.argmax(np.linalg.norm(vecs, axis=1))])[:, None]
    w = np.median(vecs, axis=0)
    n = np.linalg.norm(w)
    return w / n if n > 0 else None


def front_ear_ratio(band: np.ndarray, channels: list[str]) -> float | None:
    """Largest forehead swing / average ear swing (peak-to-peak of 0.3-10 Hz signals). None if the
    headset has no forehead + ear sensors."""
    f = [channels.index(c) for c in FRONTAL if c in channels]
    e = [channels.index(c) for c in EARS if c in channels]
    if not f or not e or band.shape[1] < 2:
        return None
    ptp = np.ptp(band, axis=1)
    return float(ptp[f].max() / max(ptp[e].mean(), 1e-6))


class BlinkCountDetector(Detector):
    name = "triple_blink"
    label = "Triple blink"
    calibratable = True

    def __init__(self, fs: float, channels: list[str], cfg: BlinkConfig | None = None) -> None:
        super().__init__(fs, channels)
        self.cfg = cfg or BlinkConfig()
        self.count = int(self.cfg.count)
        self.label = COUNT_LABELS.get(self.count, f"{self.count} blinks")
        self.idx = [self._ch(c) for c in FRONTAL if c in channels] or [0]
        # blink signal = weights . channels; starts as the AF7/AF8 average ("front", see the module docstring)
        self.weights = np.zeros(len(channels))
        self.weights[self.idx] = 1.0 / len(self.idx)
        self.weights_id = "front"
        self.pattern = "front"
        self.calibrated_threshold: float | None = self.cfg.threshold_uv
        self.pattern_threshold: float | None = None      # comes with the built-in / trained pattern
        self.veto_ratio: float | None = None
        self.vetoed = 0
        self.reset()

    def set_weights(self, weights, weights_id: str, threshold: float | None = None,
                    veto_ratio: float | None = None, pattern: str = "trained") -> None:
        """Use a learned blink pattern (and, optionally, the threshold and nod check that go with it)."""
        self.weights = np.asarray(weights, dtype=np.float64)
        self.weights_id = weights_id
        self.pattern = pattern
        if threshold:
            self.calibrated_threshold = self.pattern_threshold = float(threshold)
        self.veto_ratio = veto_ratio
        self.reset()

    def use_builtin_pattern(self) -> bool:
        """Real headset, no trained model: the shipped Muse blink pattern (if this headset has the Muse sensors)."""
        b = builtin_pattern(self.channels)
        if b is None:
            return False
        self.set_weights(b[0], weights_id(b[0]), b[1], pattern="built-in")
        return True

    def _new_filter(self) -> StreamingFilter:
        return StreamingFilter(self.fs, 1, tuple(self.cfg.band_hz), notch="off", order=2)

    def reset(self) -> None:
        from bayn.eeg.buffer import RingBuffer

        self._filter = self._new_filter()
        self._band = StreamingFilter(self.fs, len(self.channels), tuple(self.cfg.band_hz), notch="off", order=2)
        self._buf = RingBuffer(len(self.channels), int(4 * self.fs))   # all sensors, for the nod check
        self._in_peak = False
        self._sign = 1.0                               # polarity of the peak in progress
        self._peak_start = 0.0
        self._peak_t = 0.0
        self._peak_amp = 0.0
        self._peaks: list[tuple[float, float, float]] = []   # (time, amplitude, sign) of blinks
        self._valley: float | None = None              # lowest point since the last peak ended (its direction)
        self._base = 0.0                               # level the current peak rose from (<= 0)
        self.level = 0.0

    @property
    def threshold(self) -> float:
        return float(self.calibrated_threshold or self.cfg.default_threshold_uv)

    def _signal(self, raw: np.ndarray) -> np.ndarray:
        return self._filter.process((self.weights @ np.asarray(raw, dtype=np.float64))[None, :])[0]

    # ------------------------------------------------------------ detection
    def update(self, raw: EEGChunk, filt: EEGChunk) -> list[Detection]:
        x = self._signal(raw.data)
        ts = raw.timestamps()
        if self.veto_ratio:
            self._buf.append(self._band.process(raw.data), ts)
        thr = self.threshold
        out: list[Detection] = []
        for t, v in zip(ts, x):
            t, v = float(t), float(v)
            if not self._in_peak:
                if self._valley is not None and self._peaks:
                    # inside a series: each blink's high-pass rebound pulls the baseline down, so a later
                    # blink counts by its RISE from the dip before it, not by its absolute level
                    s = self._sign * v
                    self._valley = min(self._valley, s)
                    base = min(self._valley, 0.0)
                    # ...but it must also clearly cross zero (the rebound recovering to baseline is no blink)
                    start = s - base >= thr and s >= RELEASE * thr and s >= self._valley + REARM * thr
                else:
                    self._valley, base = None, 0.0
                    start = abs(v) >= thr
                    if start:
                        self._sign = 1.0 if v > 0 else -1.0
                if start:
                    self._in_peak, self._base, self._valley = True, base, None
                    self._peak_start, self._peak_t, self._peak_amp = t, t, self._sign * v
                else:
                    out += self._evaluate(t)
            else:
                s = self._sign * v                       # signal in this peak's direction
                if s > self._peak_amp:
                    self._peak_t, self._peak_amp = t, s
                rise = self._peak_amp - self._base
                if s < RELEASE * thr or s - self._base < DIP * rise:   # back down, or a dip between fast blinks
                    self._in_peak = False
                    self._valley = s
                    if t - self._peak_start <= self.cfg.max_width_s:
                        self._add_peak(self._peak_t, rise, self._sign)
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
        # fewer than N blinks: wait for the next one; N (or more): wait to rule out one more
        wait = self.cfg.max_gap_s + 0.05 if len(self._peaks) < self.count else self.cfg.settle_s
        if since_last < wait:
            return []
        peaks, self._peaks = self._peaks, []
        # deliberate blinks of a series are similar in size; a weaker natural blink right before or after
        # it (more likely the longer the series) must not turn "3 blinks" into "4 blinks"
        top = max(p[1] for p in peaks)
        peaks = [p for p in peaks if p[1] >= MIN_RELATIVE * top]
        if len(peaks) != self.count:
            return []                                     # too few or too many blinks: ignore
        gaps = np.diff([p[0] for p in peaks])
        if not all(self.cfg.min_gap_s <= g <= self.cfg.max_gap_s for g in gaps):
            return []
        amps = [p[1] for p in peaks]
        info = {"gaps_s": [round(float(g), 3) for g in gaps], "amplitudes_uv": [round(a, 1) for a in amps],
                "threshold_uv": self.threshold, "method": "rule"}
        if self.veto_ratio:
            band, _ = self._buf.window(peaks[0][0] - VETO_PRE_S, peaks[-1][0] + VETO_POST_S)
            ratio = front_ear_ratio(band, self.channels)
            if ratio is not None:
                info["front_ear_ratio"] = round(ratio, 2)
                if ratio > self.veto_ratio:
                    self.vetoed += 1
                    return []                             # forehead-heavy: a nod / head movement
        conf = min(1.0, min(amps) / (2 * self.threshold))
        return [Detection(self.name, peaks[0][0], conf, info=info)]

    def status(self) -> dict:
        return {"name": self.name, "label": self.label, "level": round(self.level, 1),
                "threshold": round(self.threshold, 1), "unit": "µV",
                "calibrated": self.calibrated_threshold is not None and (self.pattern == "trained" or
                                                                  self.calibrated_threshold != self.pattern_threshold),
                "pattern": self.pattern, "active": self._in_peak or bool(self._peaks)}

    # ------------------------------------------------------------ calibration
    def feature_trace(self, raw: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        """|filtered blink signal| over a stored segment (fresh filter state)."""
        w = self.weights if weights is None else weights
        f = self._new_filter()
        return np.abs(f.process((w @ np.asarray(raw, dtype=np.float64))[None, :])[0])

    def _measure(self, w: np.ndarray, rest_raw: np.ndarray, gesture_raws: list[np.ndarray]) -> CalibrationResult:
        warm = int(self.fs)
        space = self.weights_id if w is self.weights else weights_id(w)
        rest = self.feature_trace(rest_raw, w)[warm:]
        rest_level = float(np.percentile(rest, 99))       # includes natural blinks during rest
        peaks = [float(self.feature_trace(g, w)[warm:].max()) for g in gesture_raws if g.shape[1] > warm]
        if not peaks or rest_level <= 0:
            return CalibrationResult(self.threshold, rest_level, 0.0, 0.0, "failed", "no gesture data",
                                     space)
        gesture_level = float(np.median(peaks))
        sep = gesture_level / rest_level
        # between rest and the deliberate blinks, but low enough that every blink of the series passes
        thr = min(float(np.sqrt(rest_level * gesture_level)), 0.7 * gesture_level)
        quality = "good" if sep >= 1.8 else "weak" if sep >= 1.2 else "failed"
        msg = {"good": "deliberate blinks clearly stronger than resting signal",
               "weak": "blink a bit harder, and keep your eyes still between prompts",
               "failed": "blinks not distinguishable - check the headset fits snugly (forehead and ears)"}[quality]
        return CalibrationResult(thr, rest_level, gesture_level, sep, quality, msg, space)

    def calibrate(self, rest_raw: np.ndarray, gesture_raws: list[np.ndarray]) -> CalibrationResult:
        """Threshold between rest and the person's deliberate blinks. Unless a trained blink model is loaded,
        the blink pattern is learned too, from these same blinks, and kept unless it can't tell them from rest
        (then the current signal stays if it can). On the user's real sessions, learning always was slightly
        better than keeping whichever signal separated better (66.2 vs 65.4 % caught): a signal that misses
        the blinks can still look "separated", because natural blinks don't show on it either."""
        cur = self._measure(self.weights, rest_raw, gesture_raws)
        best, w_best = cur, self.weights
        learned = None if self.pattern == "trained" else learn_pattern(gesture_raws, self.fs, tuple(self.cfg.band_hz),
                                                                       warm=int(self.fs))
        if learned is not None:
            learned = np.round(learned, 3)                # as stored in the profile (same weights_id on reload)
            r = self._measure(learned, rest_raw, gesture_raws)
            if r.quality != "failed" or r.separation > cur.separation:
                best, w_best = r, learned
                best.weights = dict(zip(self.channels, [float(v) for v in learned]))
        if best.quality != "failed":
            if best.weights:
                self.set_weights(w_best, best.space, pattern="calibrated")
            self.calibrated_threshold = best.threshold
        return best

    def apply_profile(self, params: dict) -> None:
        w = params.get("weights")
        if w and self.pattern != "trained" and sorted(w) == sorted(self.channels):
            arr = np.array([w[c] for c in self.channels], dtype=np.float64)    # the pattern learned by Calibrate
            self.set_weights(arr, weights_id(arr), pattern="calibrated")
        if params.get("space", "front") != self.weights_id:
            return          # calibrated on another blink signal (e.g. before a blink model was trained)
        if params.get("threshold"):
            self.calibrated_threshold = float(params["threshold"])
