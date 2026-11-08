"""Brainwave band powers (delta ... gamma) per sensor, like the Muse app shows.

Computed from the raw signal (DC removed) with Welch's method. Gamma stops at 44 Hz
so 50/60 Hz mains hum is never counted as brain activity.
Note: on AF7/AF8, eye blinks show up as delta; on TP9/TP10, jaw tension as beta/gamma.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

# name -> (low Hz, high Hz); same bands as the Muse app
BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 44.0),
}


def band_powers(data: np.ndarray, fs: float, bands: dict[str, tuple[float, float]] = BANDS
                ) -> tuple[np.ndarray, np.ndarray]:
    """data (n_ch, n) µV -> (absolute µV² per band, relative fraction per band), both (n_ch, n_bands).

    Relative powers of a channel sum to 1 over the given bands.
    """
    x = data - data.mean(axis=-1, keepdims=True)
    nperseg = int(min(x.shape[-1], fs))          # 1 s windows -> 1 Hz resolution
    f, psd = welch(x, fs=fs, nperseg=nperseg, axis=-1)
    df = f[1] - f[0]
    absolute = np.stack([psd[:, (f >= lo) & (f < hi)].sum(axis=-1) * df for lo, hi in bands.values()], axis=-1)
    total = absolute.sum(axis=-1, keepdims=True)
    relative = absolute / np.maximum(total, 1e-12)
    return absolute, relative


def bands_message(data: np.ndarray, fs: float, channels: list[str], t: float) -> dict:
    """Bus payload for TOPIC_EEG_BANDS."""
    absolute, relative = band_powers(data, fs)
    return {"t": t, "channels": list(channels), "bands": {k: list(v) for k, v in BANDS.items()},
            "relative": relative.astype(np.float32), "absolute_db": (10 * np.log10(absolute + 1e-12)).astype(np.float32)}
