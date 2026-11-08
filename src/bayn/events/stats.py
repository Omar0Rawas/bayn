"""Basic EEG statistics for an event window (before vs after the trigger).

These are descriptive signal statistics for the summary prompt (Phase 5) and the
event timeline - not clinical measures.
"""

from __future__ import annotations

import numpy as np

from bayn.eeg.bands import BANDS, band_powers
from bayn.eeg.quality import channel_quality, robust_amplitude


def _segment_stats(raw: np.ndarray, filt: np.ndarray, fs: float, channels: list[str]) -> dict:
    if raw.shape[1] < int(fs):                      # less than 1 s: not enough for spectra
        return {"seconds": round(raw.shape[1] / fs, 2)}
    _, rel = band_powers(raw, fs)
    names = list(BANDS)
    quality = {q.channel: q.level for q in channel_quality(filt, channels)}
    return {
        "seconds": round(raw.shape[1] / fs, 2),
        "bands": {ch: {b: round(float(rel[i, j]), 3) for j, b in enumerate(names)} for i, ch in enumerate(channels)},
        "dominant": {ch: names[int(np.argmax(rel[i]))] for i, ch in enumerate(channels)},
        "amplitude_uv": {ch: round(float(a), 1) for ch, a in zip(channels, robust_amplitude(filt))},
        "quality": quality,
    }


def event_window_stats(raw: np.ndarray, filt: np.ndarray, ts: np.ndarray, fs: float,
                       channels: list[str], t_event: float) -> dict:
    pre = ts < t_event
    return {
        "before": _segment_stats(raw[:, pre], filt[:, pre], fs, channels),
        "after": _segment_stats(raw[:, ~pre], filt[:, ~pre], fs, channels),
    }
