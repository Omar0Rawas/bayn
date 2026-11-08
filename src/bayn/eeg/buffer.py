"""Thread-safe rolling buffer of multichannel samples with per-sample timestamps."""

from __future__ import annotations

import threading

import numpy as np


class RingBuffer:
    def __init__(self, n_channels: int, capacity: int) -> None:
        self.n_channels = n_channels
        self.capacity = capacity
        self._data = np.zeros((n_channels, capacity), dtype=np.float32)
        self._ts = np.zeros(capacity, dtype=np.float64)
        self._write = 0       # next write position
        self._count = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return self._count

    def append(self, data: np.ndarray, timestamps: np.ndarray) -> None:
        """data: (n_channels, n); timestamps: (n,)."""
        n = data.shape[1]
        if n >= self.capacity:  # keep only the newest `capacity` samples
            data, timestamps, n = data[:, -self.capacity:], timestamps[-self.capacity:], self.capacity
        with self._lock:
            idx = (self._write + np.arange(n)) % self.capacity
            self._data[:, idx] = data
            self._ts[idx] = timestamps
            self._write = (self._write + n) % self.capacity
            self._count = min(self._count + n, self.capacity)

    def _ordered_idx(self) -> np.ndarray:
        start = (self._write - self._count) % self.capacity
        return (start + np.arange(self._count)) % self.capacity

    def latest(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Newest n samples, oldest first: (data (n_ch, m), ts (m,)) with m <= n."""
        with self._lock:
            idx = self._ordered_idx()[-n:] if n > 0 else np.array([], dtype=int)
            return self._data[:, idx].copy(), self._ts[idx].copy()

    def window(self, t_start: float, t_end: float) -> tuple[np.ndarray, np.ndarray]:
        """All samples with t_start <= ts < t_end, oldest first."""
        with self._lock:
            idx = self._ordered_idx()
            ts = self._ts[idx]
            sel = idx[(ts >= t_start) & (ts < t_end)]
            return self._data[:, sel].copy(), self._ts[sel].copy()

    @property
    def last_timestamp(self) -> float | None:
        with self._lock:
            return None if self._count == 0 else float(self._ts[(self._write - 1) % self.capacity])
