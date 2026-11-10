"""Window features for the mock seizure model, identical offline (training on CHB-MIT) and live (Muse / sim).

4 s windows every 1 s on 4 channels (TP9, AF7, AF8, TP10), band-passed 0.5-40 Hz. Features are chosen to
transfer between recording setups (hospital electrodes vs Muse): mostly relative quantities.
  - change against the person's own recent baseline: log line length and log power vs the median of the
    windows 30-180 s earlier (seizures are a sudden, sustained change)
  - spectral shape: relative delta / theta / alpha / beta power, dominant 1-12 Hz frequency, its peak
    prominence and autocorrelation (seizures are often rhythmic)
  - Hjorth mobility / complexity, and synchrony (mean correlation between channels)
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.signal import welch

from bayn.eeg.filters import StreamingFilter

WINDOW_S, STEP_S = 4.0, 1.0
BASE_FROM_S, BASE_TO_S = 180.0, 30.0           # baseline = windows between 180 s and 30 s ago
BAND = (0.5, 40.0)
FEATURES = (
    "ll_vs_base", "ll_vs_base_max", "pow_vs_base", "pow_vs_base_max",
    "rel_delta", "rel_theta", "rel_alpha", "rel_beta",
    "peak_freq", "peak_prominence", "autocorr_peak",
    "hjorth_mobility", "hjorth_complexity", "channel_sync",
)


def preprocess(data: np.ndarray, fs: float, filt: StreamingFilter | None = None) -> np.ndarray:
    """Band-pass 0.5-40 Hz + mains notch (50 and 60 Hz), causal: same as live."""
    f = filt or StreamingFilter(fs, data.shape[0], BAND, notch="auto", order=4)
    return f.process(data)


def window_core(x: np.ndarray, fs: float) -> tuple[np.ndarray, float, float]:
    """(spectral/shape features, log line length, log power) of one (4, n) window, n = WINDOW_S * fs."""
    x = x - x.mean(axis=1, keepdims=True)
    ll = np.log(np.abs(np.diff(x, axis=1)).sum(axis=1) / x.shape[1] + 1e-6)
    pw = np.log(x.var(axis=1) + 1e-6)
    f, p = welch(x, fs=fs, nperseg=min(x.shape[1], int(2 * fs)), axis=1)
    band = lambda lo, hi: p[:, (f >= lo) & (f < hi)].sum(axis=1)
    tot = band(1, 30) + 1e-12
    rel = [band(1, 4) / tot, band(4, 8) / tot, band(8, 13) / tot, band(13, 30) / tot]
    m = (f >= 1) & (f <= 12)
    pm = p[:, m].mean(axis=0)
    peak_freq = float(f[m][np.argmax(pm)])
    prominence = float(np.log(pm.max() / (pm.mean() + 1e-12)))
    # autocorrelation peak at lags 0.1-1 s (rhythmicity), channel mean
    lo, hi = int(0.1 * fs), int(1.0 * fs)
    ac = []
    for ch in x:
        c = np.correlate(ch, ch, mode="full")[len(ch) - 1:]
        c = c / (c[0] + 1e-12)
        ac.append(c[lo:hi].max())
    d1, d2 = np.diff(x, axis=1), np.diff(x, n=2, axis=1)
    mob = np.sqrt(d1.var(axis=1) / (x.var(axis=1) + 1e-12))
    comp = np.sqrt(d2.var(axis=1) / (d1.var(axis=1) + 1e-12)) / (mob + 1e-12)
    cc = np.corrcoef(x)
    sync = float(np.nanmean(np.abs(cc[np.triu_indices(len(x), 1)])))
    core = np.array([*[float(r.mean()) for r in rel], peak_freq, prominence, float(np.mean(ac)),
                     float(mob.mean()), float(comp.mean()), sync])
    return core, ll, pw


class FeatureStream:
    """Causal feature computation: feed band-passed samples, get a feature row every STEP_S seconds."""

    def __init__(self, fs: float, n_ch: int = 4) -> None:
        self.fs = float(fs)
        self.win, self.step = int(WINDOW_S * fs), int(STEP_S * fs)
        self.buf = np.zeros((n_ch, 0), np.float32)
        self.buf_start = 0                 # absolute sample index of buf[:, 0]
        self.total = 0                     # samples received so far
        self.next_end = self.win           # absolute end of the next window
        self.history: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=int(BASE_FROM_S / STEP_S) + 1)

    def push(self, x: np.ndarray) -> list[np.ndarray]:
        """x: (n_ch, n) band-passed. Returns 0..k feature rows (one per completed step)."""
        out = []
        self.buf = np.concatenate([self.buf, x], axis=1)
        self.total += x.shape[1]
        while self.total >= self.next_end:
            a = self.next_end - self.win - self.buf_start
            out.append(self._row(self.buf[:, a:a + self.win]))
            self.next_end += self.step
        keep = self.total - (self.next_end - self.win)          # samples still needed for the next window
        drop = self.buf.shape[1] - keep
        if drop > 0:
            self.buf, self.buf_start = self.buf[:, drop:], self.buf_start + drop
        return out

    def _row(self, w: np.ndarray) -> np.ndarray:
        core, ll, pw = window_core(w, self.fs)
        n_old = int((BASE_FROM_S - BASE_TO_S) / STEP_S)
        past = list(self.history)[:-int(BASE_TO_S / STEP_S)] if len(self.history) > BASE_TO_S / STEP_S else []
        past = past[-n_old:]
        if past:
            base_ll = np.median(np.array([h[0] for h in past]), axis=0)
            base_pw = np.median(np.array([h[1] for h in past]), axis=0)
            dll, dpw = ll - base_ll, pw - base_pw
        else:
            dll, dpw = np.zeros_like(ll), np.zeros_like(pw)     # no baseline yet: "no change"
        self.history.append((ll, pw))
        return np.concatenate([[dll.mean(), dll.max(), dpw.mean(), dpw.max()], core])

    @property
    def has_baseline(self) -> bool:
        return len(self.history) > BASE_TO_S / STEP_S + 10


def recording_features(data: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Offline: (features (n_windows, F), window end times in s) for a whole recording, causally."""
    x = preprocess(data, fs)
    fsx = FeatureStream(fs, data.shape[0])
    rows = []
    chunk = int(fs)
    for i in range(0, x.shape[1], chunk):
        rows += fsx.push(x[:, i:i + chunk])
    ends = (int(WINDOW_S * fs) + np.arange(len(rows)) * int(STEP_S * fs)) / fs
    return np.array(rows).reshape(-1, len(FEATURES)), ends
