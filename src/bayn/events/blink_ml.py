"""Machine-learning triple-blink trigger detector (Phase 6), EEG only.

Why: the rule-based detector (one amplitude threshold on the forehead signal) can't tell a deliberate
triple blink from a nod or a head turn, whose electrode artefacts are just as large (finding F1).

How:
  1. CandidateFinder: a deliberately sensitive streaming peak finder on the forehead signal
     (AF7+AF8, 0.3-10 Hz). Peaks close together form an *episode* (a triple blink, a nod, ...).
  2. episode_features(): ~30 numbers describing the episode on all four sensors: peak count, gaps,
     widths and rebound on the forehead; whether AF7 and AF8 agree (blinks) or oppose (sideways
     eye movements); the synchronised TP9/TP10 peak the user noticed with blinks (size, polarity,
     correlation and lag versus the forehead); slow electrode drift (head movement) and muscle
     energy (talking, neck).
  3. A random forest per user, trained on labelled recording sessions (events/training.py), with
     classes triple_blink / blinks (one or two: must not fire) / movement / other. It fires when
     P(triple_blink) >= threshold.
  4. Evaluation against the rule on held-out data (leave one session out, or time blocks within a
     single session): triple blinks caught per prompt, and false triggers per minute.

The finder and features run identically offline (training) and live (MLBlinkDetector), on the
same causal filters, so what was evaluated is what runs.

    python -m bayn.events.blink_ml train   [--profile default] [--source muse]   # evaluate + save model
    python -m bayn.events.blink_ml inspect [--profile default] [--source muse]   # TP9/TP10 vs forehead stats
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, peak_widths

from bayn.common.config import DATA_DIR, BlinkConfig, load_config
from bayn.common.schemas import EEGChunk
from bayn.eeg.buffer import RingBuffer
from bayn.eeg.filters import StreamingFilter
from bayn.events.detectors.base import CalibrationResult, Detection, Detector
from bayn.events.detectors.blinks import BlinkCountDetector, front_ear_ratio
from bayn.events.training import MOVEMENTS, NEAR_MISSES, Recording, list_recordings

log = logging.getLogger("bayn.events")

FRONT, TEMP = ("AF7", "AF8"), ("TP9", "TP10")
BAND = (0.3, 10.0)
SLOW_BAND = (0.05, 1.0)
EMG_BAND = (20.0, 100.0)
PRE_S, POST_S, WARM_S = 1.0, 0.7, 2.0          # feature window around an episode (+ filter warm-up)
TARGET = "triple_blink"
CLASSES = (TARGET, "blinks", "movement", "other")    # blinks = one or two blinks: must NOT fire
MODELS_DIR = DATA_DIR / "models"
MODEL_VERSION = 3              # 3: learned blink signal (weights) + rule threshold in the bundle
CHUNK = 32


# ---------------------------------------------------------------- the person's blink signal
def default_weights(channels: list[str]) -> np.ndarray:
    """Before training: the AF7/AF8 average (where blinks are textbook-largest)."""
    w = np.zeros(len(channels))
    idx = [channels.index(c) for c in FRONT if c in channels] or [0]
    w[idx] = 1.0 / len(idx)
    return w


def learn_weights(recordings: list[Recording]) -> np.ndarray:
    """The person's own blink pattern across the sensors, from their labelled blink prompts.

    For each deliberate-blink prompt, the dominant spatial pattern of the 0.3-10 Hz signal (first principal
    component) is taken, signs aligned, and the median normalised. On the user's Muse this is mostly TP9/TP10
    (the forehead reference carries the blink onto the ear sensors), not the AF7/AF8 average assumed before.
    """
    vecs = []
    for rec in recordings:
        x = StreamingFilter(rec.fs, rec.data.shape[0], BAND, notch="off", order=2).process(rec.data)
        for s in rec.steps:
            if s.kind != "do" or s.label not in (TARGET, "double_blink", "blink", "four_blinks"):
                continue
            i0, i1 = np.searchsorted(rec.ts, [s.start + DO_LEAD, s.end + 0.5])
            seg = np.asarray(x[:, i0:i1], np.float64)
            if seg.shape[1] < 10:
                continue
            seg = seg - seg.mean(axis=1, keepdims=True)
            u = np.linalg.svd(seg, full_matrices=False)[0][:, 0]
            vecs.append(u * np.sqrt(np.sum((u @ seg) ** 2) / seg.shape[1]))     # weighted by its strength
    if len(vecs) < 3:
        return default_weights(recordings[0].channels)
    vecs = np.array(vecs)
    ref = vecs[np.argmax(np.linalg.norm(vecs, axis=1))]
    vecs *= np.sign(vecs @ ref)[:, None]
    w = np.median(vecs, axis=0)
    w /= np.linalg.norm(w) or 1.0
    return w


def weights_id(w: np.ndarray) -> str:
    return "w:" + ",".join(f"{v:+.2f}" for v in w)


def rule_threshold(recordings: list[Recording], w: np.ndarray, cfg: BlinkConfig) -> float | None:
    """Rule threshold on the learned blink signal, like calibration: between the resting level and the
    deliberate blinks (geometric mean, at most 70 % of a typical blink so every blink of a series passes)."""
    rest, peaks = [], []
    for rec in recordings:
        x = np.abs(StreamingFilter(rec.fs, 1, tuple(cfg.band_hz), notch="off", order=2)
                   .process((w @ rec.data.astype(np.float64))[None, :])[0])
        for s in rec.steps:
            i0, i1 = np.searchsorted(rec.ts, [s.start, s.end])
            if (s.kind in ("rest", "pause") or s.label in ("natural", "rest")) and i1 > i0:   # noisy: no pauses
                rest.append(x[i0:i1])
            elif s.kind == "do" and s.label == TARGET:
                j0, j1 = np.searchsorted(rec.ts, [s.start + DO_LEAD, s.end + DO_TAIL])
                if j1 > j0:
                    peaks.append(float(x[j0:j1].max()))
    if not rest or len(peaks) < 3:
        return None
    rest_level, blink_level = float(np.percentile(np.concatenate(rest), 99)), float(np.median(peaks))
    return min(float(np.sqrt(rest_level * blink_level)), 0.7 * blink_level)


def veto_ratio(recordings: list[Recording], cfg: BlinkConfig, w: np.ndarray, threshold: float | None) -> float | None:
    """Nod check for the rule, learned from the rule's own detections on the training sessions: the
    forehead/ear swing ratio around each detected series, for its correct detections (triple-blink prompts)
    and its mistakes. The cut-off sits between them (geometric mean of the medians) but at least 1.25 x the
    90th-percentile ratio of the person's real triple blinks, so it doesn't reject them. None when the rule
    made no mistakes in training, or blinks and mistakes aren't separable."""
    good, bad = [], []
    for rec in recordings:
        band = StreamingFilter(rec.fs, rec.data.shape[0], tuple(cfg.band_hz), notch="off", order=2).process(rec.data)
        wins = target_windows(rec)
        for t_first, t_last in rule_series(rec, cfg, threshold, w):
            i0, i1 = np.searchsorted(rec.ts, [t_first - 0.3, t_last + 0.4])
            r = front_ear_ratio(band[:, i0:i1], rec.channels)
            if r is not None:
                (good if any(a <= t_first < b for a, b in wins) else bad).append(r)
    if len(good) < 3 or not bad or np.median(bad) <= np.percentile(good, 90):
        return None
    return float(max(np.sqrt(np.median(good) * np.median(bad)), 1.25 * np.percentile(good, 90)))


# ---------------------------------------------------------------- candidate episodes
@dataclass(slots=True)
class Episode:
    t_first: float                         # time of the first peak (= event onset if it fires)
    t_last: float                          # time of the last peak
    peaks: list = field(default_factory=list)   # [(t, signed amplitude µV, width s)]
    threshold: float = 0.0

    @property
    def amp(self) -> float:
        return max(abs(p[1]) for p in self.peaks)


class CandidateFinder:
    """Streaming forehead peak finder that groups nearby peaks into episodes.

    Threshold = k_sigma x robust noise of the last noise_s seconds, kept between min_uv and max_uv,
    i.e. well below a deliberate blink: it is the classifier's job to reject what isn't one. The cap
    matters on the real headset, whose forehead noise (22-87 µV robust sigma in the user's Phase 3
    recordings, vs ~10 µV in the simulator) would otherwise push the threshold above real blinks.
    """

    def __init__(self, fs: float, channels: list[str], min_uv: float = 30.0, k_sigma: float = 3.0,
                 max_uv: float = 100.0, gap_s: float = 0.6, max_len_s: float = 2.5, noise_s: float = 20.0,
                 weights: np.ndarray | None = None) -> None:
        self.fs, self.min_uv, self.k_sigma, self.max_uv = float(fs), min_uv, k_sigma, max_uv
        self.gap_s, self.max_len_s = gap_s, max_len_s
        self.weights = default_weights(channels) if weights is None else np.asarray(weights, np.float64)
        self._noise_n = int(noise_s * fs)
        self.reset()

    def reset(self) -> None:
        self._filter = StreamingFilter(self.fs, 1, BAND, notch="off", order=2)
        self._hist = np.zeros(self._noise_n, np.float32)
        self._hist_n = self._hist_i = 0
        self._since_sigma = 0
        self.sigma: float | None = None
        self._in_peak = False
        self._sign = 1.0
        self._p_start = self._p_t = self._p_amp = 0.0
        self._ep: Episode | None = None
        self._last_exit = 0.0

    @property
    def threshold(self) -> float:
        return min(self.max_uv, max(self.min_uv, self.k_sigma * self.sigma)) if self.sigma else self.min_uv

    def _update_noise(self, x: np.ndarray) -> None:
        n = len(x)
        idx = (self._hist_i + np.arange(n)) % self._noise_n
        self._hist[idx] = x
        self._hist_i = (self._hist_i + n) % self._noise_n
        self._hist_n = min(self._hist_n + n, self._noise_n)
        self._since_sigma += n
        if self._since_sigma >= self.fs and self._hist_n >= 2 * self.fs:
            h = self._hist[: self._hist_n]
            self.sigma = float(1.4826 * np.median(np.abs(h - np.median(h))))
            self._since_sigma = 0

    def feed(self, data: np.ndarray, ts: np.ndarray) -> list[Episode]:
        x = self._filter.process((self.weights @ np.asarray(data, np.float64))[None, :])[0]
        out: list[Episode] = []
        thr = self.threshold
        for t, v in zip(ts, x):
            t, v = float(t), float(v)
            if not self._in_peak:
                if abs(v) >= thr:
                    self._in_peak, self._sign = True, (1.0 if v > 0 else -1.0)
                    self._p_start, self._p_t, self._p_amp = t, t, abs(v)
                elif self._ep is not None and (t - self._last_exit >= self.gap_s
                                               or t - self._ep.t_first >= self.max_len_s):
                    out.append(self._ep)
                    self._ep = None
            else:
                s = self._sign * v
                if s > self._p_amp:
                    self._p_t, self._p_amp = t, s
                if s < 0.5 * thr:
                    self._in_peak = False
                    self._last_exit = t
                    peak = (self._p_t, self._sign * self._p_amp, t - self._p_start)
                    if self._ep is None:
                        self._ep = Episode(self._p_t, self._p_t, [peak], thr)
                    else:
                        self._ep.peaks.append(peak)
                        self._ep.t_last = self._p_t
        self._update_noise(x)
        return out


def find_episodes(data: np.ndarray, ts: np.ndarray, fs: float, channels: list[str], **kw) -> list[Episode]:
    """Offline: run the streaming finder over a whole recording (same code path as live)."""
    f = CandidateFinder(fs, channels, **kw)
    out: list[Episode] = []
    for i in range(0, data.shape[1], CHUNK):
        out += f.feed(data[:, i:i + CHUNK], ts[i:i + CHUNK])
    return out


# ---------------------------------------------------------------- features
FEATURES = (
    "log_amp", "n_peaks", "n_strong", "n_finder_peaks", "n_same_sign", "duration_s", "gap_s", "gap2_s",
    "second_ratio", "third_ratio", "width_main_s", "width_second_s", "width_third_s", "rebound", "noise_pre",
    "af_corr", "af_symmetry", "af_same_sign", "af_diff_ratio",
    "tp_corr", "tp_at_peak", "tp_ratio", "tp_lag_s", "tp9_tp10_corr",
    "slow_front", "slow_tp", "slow_tp_diff",
    "emg_front", "emg_tp", "emg_tp_vs_pre", "emg_front_vs_pre",
)


def segment_bounds(ep: Episode) -> tuple[float, float]:
    return ep.t_first - PRE_S - WARM_S, ep.t_last + POST_S


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _filtered(seg: np.ndarray, fs: float, band: tuple[float, float], order: int) -> np.ndarray:
    return StreamingFilter(fs, seg.shape[0], band, notch="off", order=order).process(seg).astype(np.float64)


def episode_features(seg: np.ndarray, seg_t0: float, fs: float, channels: list[str], ep: Episode,
                     weights: np.ndarray | None = None) -> np.ndarray:
    """Feature vector (order = FEATURES) for one episode. seg: raw µV from segment_bounds(ep).
    The "blink signal" features use weights . channels (the learned blink pattern; default AF7/AF8)."""
    ci = {c: channels.index(c) for c in FRONT + TEMP}
    n = seg.shape[1]

    def at(t: float) -> int:
        return int(np.clip(round((t - seg_t0) * fs), 0, n - 1))

    band = _filtered(seg, fs, BAND, 2)
    slow = _filtered(seg, fs, SLOW_BAND, 2)
    emg = _filtered(seg, fs, EMG_BAND, 4)
    s0, s1 = at(ep.t_first - 0.25), max(at(ep.t_last + 0.45), at(ep.t_first - 0.25) + 4)
    p0, p1 = at(ep.t_first - PRE_S), max(at(ep.t_first - 0.3), at(ep.t_first - PRE_S) + 4)

    w = default_weights(channels) if weights is None else np.asarray(weights, np.float64)
    front = w @ band                                  # the blink signal
    tp = (band[ci["TP9"]] + band[ci["TP10"]]) / 2
    f = front[s0:s1]
    ipk = int(np.argmax(np.abs(f)))
    sign = 1.0 if f[ipk] >= 0 else -1.0
    fsg = sign * f
    amp = max(float(fsg[ipk]), 1e-3)

    pk, _ = find_peaks(fsg, height=0.3 * amp, distance=max(1, int(0.15 * fs)), prominence=0.2 * amp)
    if len(pk) == 0:
        pk = np.array([ipk])
    top = sorted(pk, key=lambda i: -fsg[i])[:3]             # the three strongest peaks
    top_t = sorted(top)
    gaps = np.diff(top_t) / fs
    gap = float(gaps[0]) if len(gaps) > 0 else 0.0
    gap2 = float(gaps[1]) if len(gaps) > 1 else 0.0
    second = fsg[top[1]] / amp if len(top) > 1 else 0.0
    third = fsg[top[2]] / amp if len(top) > 2 else 0.0
    with warnings.catch_warnings():                # flat-topped peaks: width 0 is fine as a feature
        warnings.simplefilter("ignore")
        widths = peak_widths(fsg, top, rel_height=0.5)[0] / fs
    width_main = float(widths[0])
    width_second = float(widths[1]) if len(widths) > 1 else 0.0
    width_third = float(widths[2]) if len(widths) > 2 else 0.0
    n_strong = int((fsg[pk] >= 0.5 * amp).sum())

    a7, a8 = band[ci["AF7"]][s0:s1], band[ci["AF8"]][s0:s1]
    v7, v8 = a7[ipk], a8[ipk]
    af_sym = min(abs(v7), abs(v8)) / max(abs(v7), abs(v8), 1e-6)
    af_diff = np.abs(a7 - a8).max() / max(np.abs(a7).max() + np.abs(a8).max(), 1e-6)

    t_ = tp[s0:s1]
    lag_n = int(0.15 * fs)
    fz, tz = (f - f.mean()) / (f.std() + 1e-9), (t_ - t_.mean()) / (t_.std() + 1e-9)
    xc = [np.mean(fz[max(0, -k):len(fz) - max(0, k)] * tz[max(0, k):len(tz) - max(0, -k)])
          for k in range(-lag_n, lag_n + 1)]
    lag = (int(np.argmax(np.abs(xc))) - lag_n) / fs

    sf = (slow[ci["AF7"]] + slow[ci["AF8"]])[s0:s1] / 2
    st = (slow[ci["TP9"]] + slow[ci["TP10"]])[s0:s1] / 2
    sd = (slow[ci["TP9"]] - slow[ci["TP10"]])[s0:s1]

    def rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(x ** 2))) + 1e-3

    ef = (emg[ci["AF7"]] + emg[ci["AF8"]]) / 2
    et = (emg[ci["TP9"]] + emg[ci["TP10"]]) / 2
    return np.array([
        np.log10(amp), len(pk), n_strong, len(ep.peaks), sum(1 for p in ep.peaks if np.sign(p[1]) == sign),
        ep.t_last - ep.t_first, gap, gap2, second, third, width_main, width_second, width_third,
        max(0.0, float(-fsg.min())) / amp, float(np.std(sign * front[p0:p1])) / amp,
        _corr(a7, a8), af_sym, float(np.sign(v7) == np.sign(v8)), af_diff,
        _corr(f, t_), sign * t_[ipk] / amp, float(np.abs(t_).max()) / amp, lag,
        _corr(band[ci["TP9"]][s0:s1], band[ci["TP10"]][s0:s1]),
        float(np.ptp(sf)) / amp, float(np.ptp(st)) / amp, float(np.ptp(sd)) / amp,
        np.log10(rms(ef[s0:s1])), np.log10(rms(et[s0:s1])),
        np.log10(rms(et[s0:s1]) / rms(et[p0:p1])), np.log10(rms(ef[s0:s1]) / rms(ef[p0:p1])),
    ], dtype=np.float64)


def features_for(rec_data: np.ndarray, rec_ts: np.ndarray, fs: float, channels: list[str],
                 episodes: list[Episode], weights: np.ndarray | None = None) -> tuple[np.ndarray, list[int]]:
    """Features of every episode whose window lies inside the recording; returns (X, kept indices)."""
    rows, kept = [], []
    for k, ep in enumerate(episodes):
        t0, t1 = segment_bounds(ep)
        i0, i1 = np.searchsorted(rec_ts, [t0, t1])
        if i0 == 0 and rec_ts[0] > t0 or i1 >= len(rec_ts):
            continue
        rows.append(episode_features(rec_data[:, i0:i1], float(rec_ts[i0]), fs, channels, ep, weights))
        kept.append(k)
    return (np.vstack(rows) if rows else np.zeros((0, len(FEATURES)))), kept


# ---------------------------------------------------------------- labels from the prompts
DO_LEAD, DO_TAIL = 0.2, 0.8          # reaction time after a prompt / gesture may end a bit late
BLINK_PROMPTS = (TARGET,) + NEAR_MISSES


def label_episodes(rec: Recording, episodes: list[Episode]) -> list[str | None]:
    """Class of each episode from the prompt it happened in (None = before the session started)."""
    labels: list[str | None] = [None] * len(episodes)
    taken: set[int] = set()
    for s in rec.steps:
        if s.kind != "do":
            continue
        inside = [k for k, e in enumerate(episodes)
                  if s.start + DO_LEAD <= e.t_first < s.end + DO_TAIL and k not in taken]
        taken.update(inside)
        if s.label in BLINK_PROMPTS:
            main = max(inside, key=lambda k: episodes[k].amp, default=None)
            cls = TARGET if s.label == TARGET else "blinks"
            for k in inside:
                labels[k] = cls if k == main else "other"
        else:
            for k in inside:
                labels[k] = "movement" if s.label in MOVEMENTS else "other"
    for k, e in enumerate(episodes):
        if k in taken:
            continue
        step = next((s for s in rec.steps if s.start <= e.t_first < s.end), None)
        if step is not None:
            labels[k] = "movement" if step.label in MOVEMENTS else "other"
    return labels


def target_windows(rec: Recording) -> list[tuple[float, float]]:
    return [(s.start + DO_LEAD, s.end + DO_TAIL) for s in rec.steps if s.kind == "do" and s.label == TARGET]


def window_metrics(rec: Recording, detections: list[float]) -> dict:
    """Prompt-level scoring: a triple-blink prompt counts as caught if a detection falls in its window;
    every detection outside those windows is a false trigger."""
    wins = target_windows(rec)
    caught = sum(any(a <= t < b for t in detections) for a, b in wins)
    false = [t for t in detections if not any(a <= t < b for a, b in wins)]
    t0, t1 = rec.steps[0].start, rec.steps[-1].end
    neg_min = max((t1 - t0) - sum(b - a for a, b in wins), 1.0) / 60
    by_context: dict[str, int] = {}
    for t in false:
        st = next((s for s in rec.steps if s.start <= t < s.end), None)
        key = (st.label + (f"/{st.variant}" if st.variant else "")) if st else "outside"
        by_context[key] = by_context.get(key, 0) + 1
    missed = {}
    for s in rec.steps:
        if s.kind == "do" and s.label == TARGET and not any(s.start + DO_LEAD <= t < s.end + DO_TAIL for t in detections):
            key = s.variant or "normal"
            missed[key] = missed.get(key, 0) + 1
    return {"prompts": len(wins), "caught": caught, "false_triggers": len(false),
            "minutes_without_prompt": round(neg_min, 2), "false_by_context": by_context, "missed_by_variant": missed}


def _sum_metrics(ms: list[dict]) -> dict:
    p = sum(m["prompts"] for m in ms)
    c = sum(m["caught"] for m in ms)
    f = sum(m["false_triggers"] for m in ms)
    mins = sum(m["minutes_without_prompt"] for m in ms)
    ctx: dict[str, int] = {}
    missed: dict[str, int] = {}
    for m in ms:
        for k, v in m["false_by_context"].items():
            ctx[k] = ctx.get(k, 0) + v
        for k, v in m.get("missed_by_variant", {}).items():
            missed[k] = missed.get(k, 0) + v
    return {"prompts": p, "caught": c, "recall": round(c / p, 3) if p else None, "false_triggers": f,
            "false_per_min": round(f / mins, 2) if mins else None, "minutes": round(mins, 2),
            "false_by_context": ctx, "missed_by_variant": missed}


def rule_series(rec: Recording, cfg: BlinkConfig, threshold: float | None = None,
                weights: np.ndarray | None = None, veto: float | None = None) -> list[tuple[float, float]]:
    """(first blink, last blink) of every series the rule fires on."""
    d = BlinkCountDetector(rec.fs, rec.channels, cfg)
    if weights is not None:
        d.set_weights(weights, weights_id(weights), threshold, veto)
    elif threshold:
        d.calibrated_threshold = threshold
    out = []
    for i in range(0, rec.data.shape[1], CHUNK):
        ch = EEGChunk(float(rec.ts[i]), rec.fs, rec.channels, rec.data[:, i:i + CHUNK])
        out += [(x.t, x.t + sum(x.info.get("gaps_s", []))) for x in d.update(ch, ch)]
    return out


def rule_detections(rec: Recording, cfg: BlinkConfig, threshold: float | None = None,
                    weights: np.ndarray | None = None, veto: float | None = None) -> list[float]:
    return [t for t, _ in rule_series(rec, cfg, threshold, weights, veto)]


# ---------------------------------------------------------------- dataset
@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray            # class names
    groups: np.ndarray       # CV group per row (time block within a single recording)
    t: np.ndarray            # episode onset
    rec: np.ndarray          # recording id per row
    recordings: list[Recording]


def build_dataset(recordings: list[Recording], weights: np.ndarray | None = None, blocks: int = 4) -> Dataset:
    X, y, g, t, r = [], [], [], [], []
    for rec in recordings:
        eps = find_episodes(rec.data, rec.ts, rec.fs, rec.channels, weights=weights)
        feats, kept = features_for(rec.data, rec.ts, rec.fs, rec.channels, eps, weights)
        labels = label_episodes(rec, eps)
        t0, t1 = rec.steps[0].start, rec.steps[-1].end
        for row, k in zip(feats, kept):
            if labels[k] is None:
                continue
            e = eps[k]
            X.append(row); y.append(labels[k]); t.append(e.t_first); r.append(rec.id)
            g.append(f"{rec.id}#{min(blocks - 1, int(blocks * (e.t_first - t0) / max(t1 - t0, 1)))}")
    return Dataset(np.array(X).reshape(-1, len(FEATURES)), np.array(y), np.array(g), np.array(t), np.array(r),
                   recordings)


# ---------------------------------------------------------------- synthetic data from the person's own recordings
SYN_LEN_S, SYN_ONSET_S = 9.0, 3.5


def blink_templates(recordings: list[Recording], weights: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
    """The person's individual real blinks, cut out of their labelled blink prompts (all sensors, raw µV,
    linear trend removed so only the blink deflection remains), and the gaps of their real triple blinks."""
    templates, gaps = [], []
    for rec in recordings:
        fs = rec.fs
        p = StreamingFilter(fs, 1, BAND, notch="off", order=2).process((weights @ rec.data.astype(np.float64))[None, :])[0]
        l0, l1 = int(0.3 * fs), int(0.45 * fs)
        for s in rec.steps:
            if s.kind != "do" or s.label not in (TARGET, "double_blink", "blink", "four_blinks"):
                continue
            i0, i1 = np.searchsorted(rec.ts, [s.start + DO_LEAD, s.end + DO_TAIL])
            seg = p[i0:i1]
            if len(seg) < 10:
                continue
            q = seg * (1.0 if seg[np.argmax(np.abs(seg))] >= 0 else -1.0)
            pk, _ = find_peaks(q, height=0.4 * q.max(), distance=int(0.15 * fs), prominence=0.3 * q.max())
            if s.label == TARGET and len(pk) == 3:
                gaps += list(np.diff(pk) / fs)
            for k in pk + i0:
                if k - l0 < 0 or k + l1 > rec.data.shape[1]:
                    continue
                snip = rec.data[:, k - l0:k + l1].astype(np.float64)
                a, b = snip[:, :8].mean(axis=1), snip[:, -8:].mean(axis=1)
                templates.append(snip - (a[:, None] + (b - a)[:, None] * np.linspace(0, 1, snip.shape[1])[None, :]))
    return templates, gaps


def background_windows(rec: Recording) -> list[tuple[int, str]]:
    """Start indices of SYN_LEN_S windows whose middle has no blink prompt, with the class of that middle."""
    n = int(SYN_LEN_S * rec.fs)
    blink_spans = [(s.start - 0.5, s.end + DO_TAIL) for s in rec.steps if s.kind == "do" and s.label in BLINK_PROMPTS]
    out = []
    for i in range(0, rec.data.shape[1] - n, int(rec.fs)):
        mid0, mid1 = rec.ts[i] + SYN_ONSET_S - 0.5, rec.ts[i] + SYN_ONSET_S + 3.5
        if any(a < mid1 and mid0 < b for a, b in blink_spans):
            continue
        step = next((s for s in rec.steps if s.start <= rec.ts[i] + SYN_ONSET_S < s.end), None)
        if step is not None:
            out.append((i, "movement" if step.label in MOVEMENTS else "other"))
    return out


def synthesize(recordings: list[Recording], weights: np.ndarray, n: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic labelled episodes built only from the person's real data: 1-4 of their real blinks (varied
    strength, speed and gaps, centred on their real triple-blink rhythm and a bit beyond) placed on real
    stretches of their recordings, including nods, head turns and talking. Used for TRAINING only."""
    rng = np.random.default_rng(seed)
    templates, gaps = blink_templates(recordings, weights)
    bgs = [(rec, i, lab) for rec in recordings for i, lab in background_windows(rec)]
    if len(templates) < 5 or not bgs:
        return np.zeros((0, len(FEATURES))), np.array([])
    fs, channels = recordings[0].fs, recordings[0].channels
    n_len = int(SYN_LEN_S * fs)
    X, y = [], []
    for _ in range(n):
        rec, i, bg_label = bgs[rng.integers(len(bgs))]
        seg = rec.data[:, i:i + n_len].astype(np.float64).copy()
        k = int(rng.choice([3, 3, 3, 1, 2, 2, 4]))
        base = float(rng.choice(gaps)) if gaps else float(rng.uniform(0.3, 0.6))
        onsets = [SYN_ONSET_S]
        for _ in range(k - 1):
            onsets.append(onsets[-1] + float(np.clip(base * rng.uniform(0.8, 1.3), 0.2, 0.95)))
        for on in onsets:
            tpl = templates[rng.integers(len(templates))]
            stretch = rng.uniform(0.85, 1.2)
            m = int(tpl.shape[1] * stretch)
            src = np.linspace(0, tpl.shape[1] - 1, m)
            w = np.vstack([np.interp(src, np.arange(tpl.shape[1]), ch) for ch in tpl]) * rng.uniform(0.75, 1.3)
            a = int(on * fs) - int(0.3 * fs * stretch)
            seg[:, a:a + m] += w[:, :max(0, n_len - a)]
        ts = np.arange(n_len) / fs
        eps = find_episodes(seg, ts, fs, channels, weights=weights)
        feats, kept = features_for(seg, ts, fs, channels, eps, weights)
        series = [kk for kk in kept if onsets[0] - 0.3 <= eps[kk].t_first <= onsets[-1] + 0.3]
        main = max(series, key=lambda kk: eps[kk].amp, default=None)
        for row, kk in zip(feats, kept):
            if kk == main:
                y.append(TARGET if k == 3 else "blinks")
            elif kk in series:
                y.append("other")
            elif not (onsets[0] - 1.0 <= eps[kk].t_first <= onsets[-1] + 1.0):
                y.append(bg_label)
            else:
                continue                 # overlaps the series ambiguously: leave it out
            X.append(row)
    return np.array(X).reshape(-1, len(FEATURES)), np.array(y)


# ---------------------------------------------------------------- training + evaluation
def new_model():
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                  random_state=0, n_jobs=1)


def p_target(model, X: np.ndarray) -> np.ndarray:
    classes = list(model.classes_)
    return model.predict_proba(X)[:, classes.index(TARGET)] if TARGET in classes else np.zeros(len(X))


def _fit(ds: Dataset, mask: np.ndarray, syn: tuple[np.ndarray, np.ndarray] | None):
    X, y = ds.X[mask], ds.y[mask]
    if syn is not None and len(syn[1]):
        X, y = np.vstack([X, syn[0]]), np.concatenate([y, syn[1]])
    if len(set(y)) < 2 or TARGET not in set(y):
        return None
    return new_model().fit(X, y)


def evaluate(recordings: list[Recording], cfg: BlinkConfig, threshold: float = 0.5,
             front_threshold: float | None = None, n_synthetic: int = 300) -> dict:
    """Held-out comparison on the person's REAL recordings only (synthetic data is never tested on):
      rule_old   the threshold rule on the AF7/AF8 average (what ran before training)
      rule       the rule on the learned blink signal, threshold learned from the training sessions:
                 the simple contender in the decision test, and the live rule
      rule_plus  rule + nod check (forehead/ear ratio, learned from the rule's training mistakes): exploratory
      ml_real    classifier trained on real episodes only
      ml         classifier trained on real + synthetic episodes: the ML contender, and the live classifier
    Several sessions: leave one session out, and the blink signal, rule threshold and synthetic data come
    from the training sessions only. One session: 4 time blocks (signal + synthetic from the same session,
    so optimistic)."""
    res = {k: [] for k in ("rule_old", "rule", "rule_plus", "ml_real", "ml")}
    scored = correct = 0
    folds = [([r for r in recordings if r is not test], test) for test in recordings] if len(recordings) > 1 \
        else [(recordings, recordings[0])]
    conditions = [test.condition for _, test in folds]
    for train_recs, test in folds:
        w = learn_weights(train_recs)
        thr = rule_threshold(train_recs, w, cfg)
        res["rule_old"].append(window_metrics(test, rule_detections(test, cfg, front_threshold)))
        res["rule"].append(window_metrics(test, rule_detections(test, cfg, thr, w)))
        res["rule_plus"].append(window_metrics(test, rule_detections(test, cfg, thr, w,
                                                                     veto_ratio(train_recs, cfg, w, thr))))
        syn = synthesize(train_recs, w, n_synthetic, seed=1)
        if len(recordings) > 1:
            dtrain, dtest = build_dataset(train_recs, w), build_dataset([test], w)
            pairs = [(dtrain, np.ones(len(dtrain.y), bool), dtest, np.ones(len(dtest.y), bool))]
        else:
            d = build_dataset([test], w)
            pairs = [(d, d.groups != g, d, d.groups == g) for g in np.unique(d.groups)]
        det = {"ml_real": [], "ml": []}
        for dtr, mtr, dte, mte in pairs:
            for key, s in (("ml_real", None), ("ml", syn)):
                m = _fit(dtr, mtr, s)
                if m is None:
                    continue
                p = p_target(m, dte.X[mte])
                det[key] += list(dte.t[mte][p >= threshold])
                if key == "ml":
                    scored += int(mte.sum())
                    correct += int(((p >= threshold) == (dte.y[mte] == TARGET)).sum())
        for key in det:
            res[key].append(window_metrics(test, det[key]))
    out = {k: _sum_metrics(v) for k, v in res.items()}
    out["by_condition"] = {c: {k: _sum_metrics([m for m, cc in zip(v, conditions) if cc == c]) for k, v in res.items()}
                           for c in sorted(set(conditions))}
    out["episodes_scored"] = scored
    out["episode_accuracy"] = round(correct / scored, 3) if scored else None
    out["cv"] = "leave one session out" if len(recordings) > 1 else "4 time blocks within one session (optimistic)"
    return out


def tp_observation(ds: Dataset) -> dict:
    """The user's observation: a synchronised TP9/TP10 peak with blinks. Median per class."""
    out = {}
    for cls in CLASSES:
        sel = ds.y == cls
        if sel.sum() == 0:
            continue
        col = lambda name: round(float(np.median(ds.X[sel, FEATURES.index(name)])), 3)
        out[cls] = {"n": int(sel.sum()), "tp_corr": col("tp_corr"), "tp_at_peak": col("tp_at_peak"),
                    "tp_ratio": col("tp_ratio"), "tp_lag_s": col("tp_lag_s"), "width_main_s": col("width_main_s"),
                    "af_same_sign": col("af_same_sign")}
    return out


# Pre-registered decision (user, 2026-11-10: "is a machine learning model even necessary?"). Fixed BEFORE
# the deciding sessions were recorded; docs/BLINK_DECISION.md is generated from these numbers.
RECALL_MARGIN = 0.10       # ML must catch >= 10 points more triple blinks (with no more false triggers)...
FALSE_MARGIN = 0.5         # ...or make >= 0.5 fewer false triggers per minute...
RECALL_TOLERANCE = 0.05    # ...without catching more than 5 points fewer. Otherwise the simpler rule wins.
FINAL_PROTOCOLS = {"standard", "noisy", "boundaries"}
FINAL_CONDITIONS = {"seated", "standing"}    # user, 2026-11-10: also standing (the demo may be given standing);
                                             # walking in place is reported, but not required


def ml_clearly_better(report: dict) -> bool:
    """The pre-registered verdict: is the classifier clearly better than the rule (Rule+) held-out?"""
    ml, rule = report.get("ml") or {}, report.get("rule") or {}
    if ml.get("recall") is None or rule.get("recall") is None:
        return False
    ml_f, rule_f = ml.get("false_per_min") or 0.0, rule.get("false_per_min") or 0.0
    return (ml["recall"] >= rule["recall"] + RECALL_MARGIN and ml_f <= rule_f) or \
        (ml_f <= rule_f - FALSE_MARGIN and ml["recall"] >= rule["recall"] - RECALL_TOLERANCE)


ml_not_worse = ml_clearly_better      # Auto uses the classifier only when the decision rule says so

DECISION_DOC = Path(__file__).resolve().parents[3] / "docs" / "BLINK_DECISION.md"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % confidence interval of a proportion (Wilson score)."""
    if not n:
        return 0.0, 0.0
    p = k / n
    centre, half = (p + z * z / (2 * n)) / (1 + z * z / n), z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def rate_ci(k: int, minutes: float) -> tuple[float, float]:
    """95 % confidence interval of a rate per minute (exact Poisson)."""
    from scipy.stats import chi2

    if minutes <= 0:
        return 0.0, 0.0
    lo = chi2.ppf(0.025, 2 * k) / 2 if k else 0.0
    return lo / minutes, chi2.ppf(0.975, 2 * k + 2) / 2 / minutes


def decision(report: dict) -> dict:
    protocols = set(report.get("protocols", []))
    missing = [f"a {p} session" for p in sorted(FINAL_PROTOCOLS - protocols)] + \
        [f"a session done {c}" for c in sorted(FINAL_CONDITIONS - set(report.get("conditions", ["seated"])))]
    better = ml_clearly_better(report)
    return {"ml_clearly_better": better, "final": not missing, "missing_protocols": missing,
            "answer": ("Yes - the classifier is clearly better than the rule on held-out real data."
                       if better else
                       "No - the classifier is not clearly better than the simple rule on the learned blink "
                       "signal, so the simpler rule is preferred.")}


def model_path(profile: str, source: str) -> Path:
    return MODELS_DIR / f"blink-{profile}-{source}.joblib"


def train_from_sessions(profile: str, source: str, cfg: BlinkConfig | None = None,
                        front_threshold: float | None = None, save: bool = True,
                        n_synthetic: int = 300) -> dict:
    """Evaluate on held-out real data, then learn the blink signal + rule threshold and train the classifier
    on everything (real + synthetic), and save. Returns the report."""
    import joblib

    cfg = cfg or BlinkConfig()
    paths = list_recordings(profile, source)
    if not paths:
        raise ValueError(f"no training sessions for profile '{profile}' / source '{source}' yet")
    recs = [Recording.load(p) for p in paths]
    w = learn_weights(recs)
    ds = build_dataset(recs, w)
    counts = {c: int((ds.y == c).sum()) for c in CLASSES}
    if counts[TARGET] < 5 or len(set(ds.y)) < 2:
        raise ValueError(f"not enough labelled examples yet: {counts}")
    thr = rule_threshold(recs, w, cfg)
    veto = veto_ratio(recs, cfg, w, thr)
    syn = synthesize(recs, w, n_synthetic, seed=2)
    report = {"profile": profile, "source": source, "sessions": [r.id for r in recs],
              "minutes": round(sum(r.minutes() for r in recs), 1), "class_counts": counts,
              "synthetic": {c: int((syn[1] == c).sum()) for c in CLASSES},
              "weights": dict(zip(recs[0].channels, [round(float(v), 3) for v in w])),
              "rule_threshold": round(thr, 1) if thr else None,
              "veto_ratio": round(veto, 2) if veto else None,
              "protocols": sorted({r.info.get("protocol", "?") for r in recs}),
              "conditions": sorted({r.condition for r in recs}),
              "session_details": [{"id": r.id, "protocol": r.info.get("protocol"), "condition": r.condition,
                                   "minutes": round(r.minutes(), 1),
                                   "triple_prompts": len(target_windows(r))} for r in recs],
              "threshold": cfg.ml_threshold, **evaluate(recs, cfg, cfg.ml_threshold, front_threshold, n_synthetic),
              "tp_observation": tp_observation(ds), "trained": time.time()}
    report["decision"] = decision(report)
    model = _fit(ds, np.ones(len(ds.y), bool), syn)
    imp = sorted(zip(FEATURES, model.feature_importances_), key=lambda kv: -kv[1])[:8]
    report["top_features"] = [[k, round(float(v), 3)] for k, v in imp]
    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = model_path(profile, source)
        joblib.dump({"version": MODEL_VERSION, "model": model, "features": FEATURES, "channels": recs[0].channels,
                     "fs": recs[0].fs, "weights": w, "rule_threshold": thr, "veto_ratio": veto,
                     "report": report}, path)
        path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["model_path"] = str(path)
    return report


# ---------------------------------------------------------------- live detector
class MLBlinkDetector(Detector):
    """Triple blink via the trained classifier; the rule-based detector until a model exists.

    The rule detector always runs alongside: it keeps the level meter and calibration working, and
    it is what fires when method = rule or no model is trained for this profile/source.
    """

    name = TARGET
    label = "Triple blink"
    calibratable = True
    BUFFER_S = 10.0

    def __init__(self, fs: float, channels: list[str], cfg: BlinkConfig | None = None) -> None:
        super().__init__(fs, channels)
        self.cfg = cfg or BlinkConfig()
        self.rule = BlinkCountDetector(fs, channels, self.cfg)
        self.label = self.rule.label
        self.method = self.cfg.method
        self.bundle: dict | None = None
        self.model_file: str | None = None
        self.model_ok = False
        self.last_p: float | None = None
        self.usable = all(c in channels for c in FRONT + TEMP)
        self.weights = default_weights(channels)
        self.reset()

    def reset(self) -> None:
        self.rule.reset()
        self.finder = CandidateFinder(self.fs, self.channels, weights=self.weights) if self.usable else None
        self.buf = RingBuffer(len(self.channels), int(self.BUFFER_S * self.fs))

    def load_model(self, path: Path) -> bool:
        """Load the person's trained blink model: classifier + learned blink signal + rule threshold. The
        rule runs on the learned signal from then on, even while Auto keeps the classifier off."""
        self.bundle, self.model_file, self.model_ok = None, None, False
        self.rule = BlinkCountDetector(self.fs, self.channels, self.cfg)          # back to the defaults
        self.weights = default_weights(self.channels)
        self.reset()
        if not self.usable or not Path(path).exists():
            return False
        try:
            import joblib

            b = joblib.load(path)
            if b.get("version") != MODEL_VERSION or tuple(b.get("features", ())) != FEATURES \
                    or b.get("channels") != self.channels or abs(float(b.get("fs", 0)) - self.fs) > 1:
                log.warning("blink model %s doesn't match this setup - retrain it", path)
                return False
            self.bundle, self.model_file = b, Path(path).name
            self.model_ok = ml_not_worse(b.get("report", {}))
            if b.get("weights") is not None:
                self.weights = np.asarray(b["weights"], np.float64)
                self.rule.set_weights(self.weights, weights_id(self.weights), b.get("rule_threshold"))
                self.reset()
            log.info("ML blink detector loaded (%s)%s", self.model_file,
                     "" if self.model_ok else " - not clearly better than the rule held-out: auto keeps the rule")
            return True
        except Exception as e:  # corrupted / incompatible pickle
            log.warning("could not load blink model %s: %s", path, e)
            return False

    @property
    def ml_active(self) -> bool:
        """ml: always the classifier once trained; auto: only if it beat the rule on held-out data."""
        if self.bundle is None or self.method == "rule":
            return False
        return self.method == "ml" or self.model_ok

    def update(self, raw: EEGChunk, filt: EEGChunk) -> list[Detection]:
        rule_out = self.rule.update(raw, filt)
        if not self.ml_active:
            return rule_out
        ts = raw.timestamps()
        self.buf.append(raw.data, ts)
        out: list[Detection] = []
        for ep in self.finder.feed(raw.data, ts):
            t0, t1 = segment_bounds(ep)
            seg, sts = self.buf.window(t0, t1)
            if sts.size < 0.9 * (t1 - t0) * self.fs:
                continue                          # not enough history yet (just connected)
            x = episode_features(seg, float(sts[0]), self.fs, self.channels, ep, self.weights)[None, :]
            p = float(p_target(self.bundle["model"], x)[0])
            self.last_p = p
            if p >= self.cfg.ml_threshold:
                out.append(Detection(self.name, ep.t_first, p, info={
                    "method": "ml", "p": round(p, 3), "span_s": round(ep.t_last - ep.t_first, 3),
                    "peaks": len(ep.peaks)}))
        return out

    def status(self) -> dict:
        st = self.rule.status()
        st.update(method="ml" if self.ml_active else "rule", model=self.model_file, model_ok=self.model_ok,
                  wanted=self.method, last_p=None if self.last_p is None else round(self.last_p, 2))
        if self.ml_active:
            st["label"] = f"{self.rule.label} (ML)"
        return st

    def calibrate(self, rest_raw: np.ndarray, gesture_raws: list[np.ndarray]) -> CalibrationResult:
        return self.rule.calibrate(rest_raw, gesture_raws)

    def apply_profile(self, params: dict) -> None:
        self.rule.apply_profile(params)

    @property
    def calibrated_threshold(self) -> float | None:
        return self.rule.calibrated_threshold


# ---------------------------------------------------------------- CLI
def _print_report(r: dict) -> None:
    print(f"Sessions: {', '.join(r['sessions'])}  ({r['minutes']} min)")
    print(f"Examples: {r['class_counts']}  + synthetic {r.get('synthetic', {})}")
    if r.get("weights"):
        print(f"Learned blink signal: {r['weights']}  rule threshold {r.get('rule_threshold')} uV"
              f"  (exploratory nod check: forehead/ear > {r.get('veto_ratio')})")
    print(f"Held-out evaluation on real data ({r['cv']}), P(triple blink) >= {r['threshold']}:")
    names = {"rule_old": "rule, forehead (old)", "rule": "rule, learned signal", "rule_plus": "rule + nod check",
             "ml_real": "ML, real data only", "ml": "ML, real + synthetic"}
    for name, label in names.items():
        if name not in r:
            continue
        m = r[name]
        print(f"  {label:22s} caught {m['caught']}/{m['prompts']} ({(m['recall'] or 0) * 100:3.0f} %)"
              f"  false {m['false_triggers']} ({m['false_per_min']}/min)  missed {m.get('missed_by_variant') or '-'}"
              f"  false in {m['false_by_context'] or '-'}")
    print("TP9/TP10 vs forehead (medians per class):")
    for cls, v in r["tp_observation"].items():
        print(f"  {cls:13s} n={v['n']:4d}  corr {v['tp_corr']:+.2f}  TP at peak {v['tp_at_peak']:+.2f}x  "
              f"|TP| {v['tp_ratio']:.2f}x  lag {v['tp_lag_s'] * 1000:+.0f} ms  width {v['width_main_s'] * 1000:.0f} ms")
    if r.get("top_features"):
        print("Most useful features: " + ", ".join(f"{k} {v:.2f}" for k, v in r["top_features"]))


def _fmt(m: dict) -> tuple[str, str]:
    lo, hi = wilson(m["caught"], m["prompts"])
    flo, fhi = rate_ci(m["false_triggers"], m.get("minutes") or 0)
    return (f"{m['caught']}/{m['prompts']} = **{(m['recall'] or 0) * 100:.0f} %** ({lo * 100:.0f}-{hi * 100:.0f} %)",
            f"{m['false_triggers']} = **{m['false_per_min']}/min** ({flo:.2f}-{fhi:.2f})")


def render_decision(r: dict) -> str:
    """docs/BLINK_DECISION.md: the pre-registered question, criteria, data, results and verdict."""
    d = r["decision"]
    status = "FINAL" if d["final"] else f"PRELIMINARY (still missing: {'; '.join(d['missing_protocols'])})"
    out = ["# Is a machine-learning model necessary for the triple-blink trigger?", "",
           f"> Generated by `python -m bayn.events.blink_ml decide` on {time.strftime('%Y-%m-%d %H:%M')}. "
           "Re-run it after every new recording session; the criteria below are fixed in code "
           "(`bayn.events.blink_ml`: RECALL_MARGIN, FALSE_MARGIN, RECALL_TOLERANCE) and must not be changed "
           "to fit a result.", "",
           f"**Status: {status}**", "",
           f"**Answer: {d['answer']}**", "",
           "## The question", "",
           "After the first real headset sessions showed that the blink detector had been looking at the wrong "
           "signal (the AF7/AF8 average; on the Muse, blinks are largest on TP9/TP10 because the reference sits on "
           "the forehead), the user asked (2026-11-10): *\"now that the misconception has been cleared up, is a "
           "machine learning model even necessary?\"* This test answers it with the person's own recordings.", "",
           "## The test (pre-registered 2026-11-10, before the deciding sessions were recorded)", "",
           "**Contenders.** Both are trained only on the sessions that are *not* being tested:",
           "- **Rule**: count three blink peaks (0.2-0.8 s apart) on the person's *learned blink signal* "
           "(weights learned from their labelled blinks), with a threshold learned from the same sessions. "
           "Explainable, about a dozen numbers, could be set up by a short calibration.",
           "- **ML**: a random-forest classifier on 31 EEG features of each candidate episode, trained on the "
           "person's real episodes plus synthetic ones recombined from their own blinks (synthetic data is used "
           "for training only).",
           "- Reported for reference, not deciding: the rule on the old AF7/AF8 average (before), the rule plus "
           "a learned nod check (exploratory), and the classifier without synthetic data.", "",
           "**Data and metric.** The person's real recording sessions, **leave one session out**: every session "
           "is scored by detectors that never saw it. A triple-blink prompt counts as caught if the detector "
           "fires within it; every other firing is a false trigger. Prompts are deliberately hard: double and "
           "four blinks, nods, head turns, looking around, talking.", "",
           "**Decision rule.** The classifier counts as *necessary* only if it is **clearly** better than the rule:",
           f"- it catches at least **{RECALL_MARGIN * 100:.0f} points more** triple blinks with no more false "
           "triggers, **or**",
           f"- it makes at least **{FALSE_MARGIN} fewer false triggers per minute** while catching at most "
           f"**{RECALL_TOLERANCE * 100:.0f} points fewer**.",
           "- Otherwise the simpler rule wins. The live system's *Auto* setting applies the same rule.", "",
           f"**Final** once the data includes a {', a '.join(sorted(FINAL_PROTOCOLS))} session, and sessions "
           f"done {' and '.join(sorted(FINAL_CONDITIONS))} (user, 2026-11-10: the demo may be given standing). "
           "Walking in place (spoken prompts) is reported per condition but not required. A badly recorded "
           "session may be deleted and re-recorded (dashboard: Train… → Saved sessions) - before running this "
           "test, never because of its result.", "",
           "## Data", "",
           "| Session | Protocol | Condition | Minutes | Triple-blink prompts |", "|---|---|---|---|---|",
           *[f"| `{s['id']}` | {s['protocol']} | {s['condition']} | {s['minutes']} | {s['triple_prompts']} |"
             for s in r.get("session_details", [])], "",
           f"Total {r['minutes']} min. Labelled episodes: {r['class_counts']}. "
           f"Synthetic (training only): {r.get('synthetic', {})}.", "",
           f"Learned blink signal: {r.get('weights')}; rule threshold {r.get('rule_threshold')} µV.", "",
           f"## Results (held-out, real data only: {r['cv']})", "",
           "| Detector | Triple blinks caught (95 % CI) | False triggers (95 % CI, per min) | Missed (variant) | False triggers during |",
           "|---|---|---|---|---|"]
    names = [("rule_old", "Rule on AF7/AF8 average (before)"), ("rule", "**Rule** on learned signal"),
             ("rule_plus", "Rule + nod check (exploratory)"), ("ml_real", "ML, real data only"),
             ("ml", "**ML**, real + synthetic")]
    for key, label in names:
        if key in r:
            m = r[key]
            caught, false = _fmt(m)
            out.append(f"| {label} | {caught} | {false} | {m.get('missed_by_variant') or '–'} | "
                       f"{m.get('false_by_context') or '–'} |")
    if len(r.get("by_condition", {})) > 1:
        out += ["", "**By condition** (rule vs ML):", "", "| Condition | Rule caught | Rule false | ML caught | ML false |",
                "|---|---|---|---|---|"]
        for cond, v in r["by_condition"].items():
            out.append(f"| {cond} | " + " | ".join(_fmt(v["rule"])) + " | " + " | ".join(_fmt(v["ml"])) + " |")
    rule, ml = r["rule"], r["ml"]
    out += ["", "## Verdict", "",
            f"- Recall: ML {(ml['recall'] or 0) * 100:.0f} % vs rule {(rule['recall'] or 0) * 100:.0f} % "
            f"(needs +{RECALL_MARGIN * 100:.0f} points).",
            f"- False triggers: ML {ml['false_per_min']}/min vs rule {rule['false_per_min']}/min "
            f"(needs {FALSE_MARGIN}/min fewer).",
            f"- **{d['answer']}**", "",
            "## Limits", "",
            "- One person, one headset, one fit per session. A different wearer needs their own sessions.",
            "- The prompts are adversarial on purpose; in normal use, triple-blink look-alikes are rarer. "
            "The 10-minute normal-activity test (TESTING.md, Phase 6) measures that separately.",
            "- With ~30 prompts per variant, differences of a few prompts are within the confidence intervals; "
            "the margins above exist so that noise can't decide.", "",
            "## Reproduce", "",
            "```powershell",
            ".\\bayn blink decide        # = python -m bayn.events.blink_ml decide --source muse",
            "```"]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["train", "inspect", "decide"],
                    help="train: evaluate + save the model | inspect: evaluate only | "
                         "decide: train + write docs/BLINK_DECISION.md (is ML necessary?)")
    ap.add_argument("--profile")
    ap.add_argument("--source", default=None, help="muse | lsl | sim (default: eeg.source)")
    args = ap.parse_args(argv)
    cfg = load_config()
    profile, source = args.profile or cfg.events.profile, args.source or cfg.eeg.source
    try:
        r = train_from_sessions(profile, source, cfg.events.triple_blink, save=args.cmd != "inspect")
    except ValueError as e:
        print(e)
        return 1
    _print_report(r)
    if r.get("model_path"):
        print(f"\nModel saved: {r['model_path']}")
    if args.cmd == "decide":
        DECISION_DOC.write_text(render_decision(r), encoding="utf-8")
        d = r["decision"]
        print(f"\n{'FINAL' if d['final'] else 'PRELIMINARY'}: {d['answer']}\nWrote {DECISION_DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
