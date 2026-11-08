"""Causal streaming filters (state carried across chunks) + offline helpers."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos


def notch_frequencies(setting: str, fs: float) -> list[float]:
    """'auto' -> both 50 and 60 Hz (both outside the 1-40 Hz band, so it is free)."""
    freqs = {"auto": [50.0, 60.0], "50": [50.0], "60": [60.0], "off": []}[setting]
    return [f for f in freqs if f < fs / 2]


def design_sos(fs: float, band: tuple[float, float] | None, notches: list[float],
               order: int = 4, q: float = 30.0) -> np.ndarray:
    """Second-order sections for an optional bandpass followed by notch filters."""
    sections = []
    if band is not None:
        lo, hi = band
        hi = min(hi, 0.45 * fs)
        sections.append(butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos"))
    for f0 in notches:
        b, a = iirnotch(f0, q, fs=fs)
        sections.append(tf2sos(b, a))
    if not sections:
        raise ValueError("filter has no sections")
    return np.vstack(sections)


class StreamingFilter:
    """Applies the same SOS filter to every channel, chunk by chunk, without edge artefacts.

    The first chunk initialises filter state at steady state for its first sample,
    which avoids a huge transient from the electrodes' DC offset.
    """

    def __init__(self, fs: float, n_channels: int, band: tuple[float, float] | None = (1.0, 40.0),
                 notch: str = "auto", order: int = 4, extra_notches: tuple[float, ...] = ()) -> None:
        self.fs = fs
        self.n_channels = n_channels
        self.notches = notch_frequencies(notch, fs) + [f for f in extra_notches if f < fs / 2]
        self.sos = design_sos(fs, band, self.notches, order)
        self._zi_unit = sosfilt_zi(self.sos)            # (n_sections, 2)
        self._zi: np.ndarray | None = None

    def reset(self) -> None:
        self._zi = None

    def process(self, data: np.ndarray) -> np.ndarray:
        """data: (n_channels, n_samples) -> filtered float32, same shape."""
        x = np.asarray(data, dtype=np.float64)
        if self._zi is None:
            self._zi = self._zi_unit[:, None, :] * x[:, 0][None, :, None]
        y, self._zi = sosfilt(self.sos, x, axis=-1, zi=self._zi)
        return y.astype(np.float32)


def bandpass_offline(data: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    """Zero-phase bandpass for analysis of a stored window (not for streaming)."""
    from scipy.signal import sosfiltfilt

    sos = butter(order, [lo, min(hi, 0.45 * fs)], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, data, axis=-1)
