"""Per-channel signal quality and mains (50/60 Hz) detection.

BrainFlow does not expose the Muse "horseshoe" contact indicator, so quality is
estimated from the signal itself. Thresholds are deliberately simple; they are
re-tuned against the real headset in Phase 6.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

from bayn.common.schemas import ChannelQuality

FLAT_UV = 0.5      # below this the electrode is probably not touching skin / railed
NOISY_UV = 40.0    # robust amplitude above this = poor contact or movement
BAD_UV = 100.0


def robust_amplitude(x: np.ndarray) -> np.ndarray:
    """MAD-based std estimate per channel; ignores short blinks/spikes."""
    med = np.median(x, axis=-1, keepdims=True)
    return 1.4826 * np.median(np.abs(x - med), axis=-1)


def _band_power(f: np.ndarray, p: np.ndarray, lo: float, hi: float) -> np.ndarray:
    m = (f >= lo) & (f <= hi)
    return p[..., m].mean(axis=-1) if m.any() else np.zeros(p.shape[:-1])


def line_noise_ratio(raw: np.ndarray, fs: float, mains: float) -> np.ndarray:
    """Power at mains +-1 Hz relative to the neighbourhood, per channel."""
    f, p = welch(raw, fs=fs, nperseg=min(raw.shape[-1], int(fs * 2)), axis=-1)
    peak = _band_power(f, p, mains - 1, mains + 1)
    side = (_band_power(f, p, mains - 6, mains - 3) + _band_power(f, p, mains + 3, mains + 6)) / 2
    return peak / np.maximum(side, 1e-12)


def detect_mains(raw: np.ndarray, fs: float, min_ratio: float = 4.0) -> int | None:
    """Return 50 or 60 if one clearly dominates (median over channels), else None."""
    if raw.shape[-1] < fs * 2:
        return None
    r50 = float(np.median(line_noise_ratio(raw, fs, 50.0)))
    r60 = float(np.median(line_noise_ratio(raw, fs, 60.0))) if fs / 2 > 61 else 0.0
    best, other = (50, r50) if r50 >= r60 else (60, r60)
    lesser = min(r50, r60)
    return best if other >= min_ratio and other >= 2 * lesser else None


def channel_quality(filtered: np.ndarray, channels: list[str]) -> list[ChannelQuality]:
    """Quality from the 1-40 Hz signal over the last couple of seconds."""
    amp = robust_amplitude(filtered)
    out = []
    for ch, a in zip(channels, amp):
        a = float(a)
        if a < FLAT_UV:
            q = ChannelQuality(ch, "bad", a, "flat - no contact?")
        elif a > BAD_UV:
            q = ChannelQuality(ch, "bad", a, "very noisy - adjust headband")
        elif a > NOISY_UV:
            q = ChannelQuality(ch, "ok", a, "noisy")
        else:
            q = ChannelQuality(ch, "good", a, "")
        out.append(q)
    return out
