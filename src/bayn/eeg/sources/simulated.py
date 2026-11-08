"""Simulated Muse 2: realistic-looking 4-channel EEG with injectable gestures.

Signal model per channel (µV):
  1/f "pink" background + waxing/waning alpha (~10 Hz, stronger on TP9/TP10)
  + theta + mains hum (50 or 60 Hz) + electrode DC offset and slow drift
  + random natural single blinks (AF7/AF8) and weak muscle bursts (distractors)
Injectable gestures:
  clench        1.5-2 s broadband 30-110 Hz EMG burst, strong on TP9/TP10
  double_blink  two large blink deflections 0.35-0.5 s apart on AF7/AF8
  blink         one natural-size blink (should NOT trigger the double-blink detector)
"""

from __future__ import annotations

import threading
import time

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi, sosfilt

from bayn.common.config import EEGConfig, SimulatorConfig
from bayn.common.schemas import GESTURES, EEGChunk
from bayn.eeg.sources.base import EEGSource

# Kasdin/Kellet IIR approximation of a 1/f (pink) spectrum.
_PINK_B = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
_PINK_A = np.array([1.0, -2.494956002, 2.017265875, -0.522189400])

FRONTAL = ("AF7", "AF8")
TEMPORAL = ("TP9", "TP10")


class SimulatedEEG:
    """Pure signal generator (no timing). generate(n) returns the next n samples."""

    def __init__(self, channels: list[str], fs: float = 256.0, seed: int | None = None,
                 mains_hz: float = 50.0, natural_blinks: bool = True, natural_emg: bool = True,
                 background_uv: float = 10.0) -> None:
        self.channels = list(channels)
        self.fs = float(fs)
        self.n_ch = len(self.channels)
        self.mains_hz = mains_hz
        self.natural_blinks = natural_blinks
        self.natural_emg = natural_emg
        self.rng = np.random.default_rng(seed)
        self.k = 0                                        # absolute sample index
        self._pending: list[tuple[int, np.ndarray]] = []  # (start sample, (n_ch, L) overlay)
        self._lock = threading.Lock()

        # channel weights: frontal channels see blinks, temporal channels see jaw EMG + alpha
        self._is_front = np.array([c in FRONTAL for c in self.channels])
        self._is_temp = np.array([c in TEMPORAL for c in self.channels])

        # pink-noise filter state, gain normalised so background std ~= background_uv
        self._pink_zi = np.tile(lfilter_zi(_PINK_B, _PINK_A), (self.n_ch, 1)) * 0.0
        probe = lfilter(_PINK_B, _PINK_A, np.random.default_rng(0).standard_normal(50_000))
        self._pink_gain = background_uv / probe[5000:].std()

        self._alpha_f = self.rng.uniform(9.3, 10.7, self.n_ch)
        self._alpha_amp = np.where(self._is_temp, 9.0, 5.0)
        self._alpha_phase = self.rng.uniform(0, 2 * np.pi, self.n_ch)
        self._theta_phase = self.rng.uniform(0, 2 * np.pi, self.n_ch)
        self._dc = self.rng.uniform(-150, 150, self.n_ch)
        self._mains_amp = self.rng.uniform(1.5, 4.0, self.n_ch)

        self._next_blink = self._draw_interval(6.0)
        self._next_emg = self._draw_interval(20.0)

    # ------------------------------------------------------------ helpers
    def _draw_interval(self, mean_s: float) -> int:
        return self.k + int((2.0 + self.rng.exponential(mean_s)) * self.fs)

    def _blink_wave(self, amp_uv: float) -> np.ndarray:
        """Asymmetric bump (fast rise, slower fall) ~0.35 s long, on all channels."""
        t = np.arange(int(0.45 * self.fs)) / self.fs
        peak = 0.10
        shape = np.where(t < peak, np.exp(-0.5 * ((t - peak) / 0.035) ** 2),
                         np.exp(-0.5 * ((t - peak) / 0.075) ** 2))
        w = np.where(self._is_front, 1.0, -0.12)[:, None] * amp_uv * shape[None, :]
        return w * self.rng.uniform(0.85, 1.0, self.n_ch)[:, None]

    def _emg_burst(self, dur_s: float, temporal_uv: float, frontal_uv: float) -> np.ndarray:
        n = int(dur_s * self.fs)
        hi = min(110.0, 0.45 * self.fs)
        sos = butter(4, [30.0, hi], btype="bandpass", fs=self.fs, output="sos")
        noise = sosfilt(sos, self.rng.standard_normal((self.n_ch, n + 200)), axis=1)[:, 200:]
        noise /= noise.std(axis=1, keepdims=True)
        ramp = min(int(0.12 * self.fs), n // 4)
        env = np.ones(n)
        env[:ramp] = np.linspace(0, 1, ramp)
        env[-ramp:] = np.linspace(1, 0, ramp)
        env *= 1.0 + 0.25 * np.sin(2 * np.pi * self.rng.uniform(1.5, 3.5) * np.arange(n) / self.fs)
        amp = np.where(self._is_temp, temporal_uv, frontal_uv)[:, None]
        return noise * amp * env[None, :]

    def _schedule(self, overlay: np.ndarray, delay_samples: int = 0) -> int:
        start = self.k + delay_samples
        self._pending.append((start, overlay.astype(np.float64)))
        return start

    # ------------------------------------------------------------ public API
    def inject(self, gesture: str, delay_s: float = 0.05) -> int:
        """Schedule a gesture; returns the absolute sample index where it starts."""
        if gesture not in GESTURES:
            raise ValueError(f"unknown gesture {gesture!r}; expected one of {GESTURES}")
        d = int(delay_s * self.fs)
        with self._lock:
            if gesture == "clench":
                ov = self._emg_burst(self.rng.uniform(1.5, 2.0), self.rng.uniform(35, 55),
                                     self.rng.uniform(8, 14))
            elif gesture == "double_blink":
                gap = int(self.rng.uniform(0.35, 0.5) * self.fs)
                b1, b2 = self._blink_wave(self.rng.uniform(170, 230)), self._blink_wave(self.rng.uniform(170, 230))
                ov = np.zeros((self.n_ch, gap + b2.shape[1]))
                ov[:, :b1.shape[1]] += b1
                ov[:, gap:gap + b2.shape[1]] += b2
            else:  # single natural blink
                ov = self._blink_wave(self.rng.uniform(80, 140))
            return self._schedule(ov, d)

    def generate(self, n: int) -> np.ndarray:
        """Next n samples as (n_ch, n) float32 µV."""
        with self._lock:
            k0 = self.k
            t = (k0 + np.arange(n)) / self.fs

            # time-major draw: same seed -> same signal regardless of chunk size
            white = self.rng.standard_normal((n, self.n_ch)).T
            pink, self._pink_zi = lfilter(_PINK_B, _PINK_A, white, axis=1, zi=self._pink_zi)
            x = pink * self._pink_gain

            # alpha waxes and wanes on a ~6-12 s cycle
            env = 0.55 + 0.45 * np.sin(2 * np.pi * 0.11 * t[None, :] + self._alpha_phase[:, None] * 0.5)
            x += self._alpha_amp[:, None] * env * np.sin(2 * np.pi * self._alpha_f[:, None] * t + self._alpha_phase[:, None])
            x += 3.0 * np.sin(2 * np.pi * 6.0 * t + self._theta_phase[:, None])
            if self.mains_hz:
                x += self._mains_amp[:, None] * np.sin(2 * np.pi * self.mains_hz * t)
            x += self._dc[:, None] + 20.0 * np.sin(2 * np.pi * 0.03 * t)[None, :]

            # random distractors
            if self.natural_blinks and k0 + n > self._next_blink:
                self._schedule(self._blink_wave(self.rng.uniform(70, 130)), self._next_blink - k0)
                self._next_blink = self._draw_interval(6.0) + (self._next_blink - k0)
            if self.natural_emg and k0 + n > self._next_emg:
                self._schedule(self._emg_burst(self.rng.uniform(0.2, 0.5), 10.0, 6.0), self._next_emg - k0)
                self._next_emg = self._draw_interval(20.0) + (self._next_emg - k0)

            # add scheduled overlays that intersect [k0, k0+n)
            keep = []
            for start, ov in self._pending:
                end = start + ov.shape[1]
                a, b = max(start, k0), min(end, k0 + n)
                if a < b:
                    x[:, a - k0:b - k0] += ov[:, a - start:b - start]
                if end > k0 + n:
                    keep.append((start, ov))
            self._pending = keep
            self.k += n
            return x.astype(np.float32)


class SimulatedSource(EEGSource):
    """Real-time-paced wrapper around SimulatedEEG, with optional scripted gestures."""

    name = "sim"
    supports_injection = True

    def __init__(self, eeg: EEGConfig, sim: SimulatorConfig) -> None:
        super().__init__(eeg.channels, eeg.fs)
        self.chunk = eeg.chunk_samples
        self.auto = sim.auto_gestures
        self.gen = SimulatedEEG(eeg.channels, eeg.fs, seed=sim.seed, mains_hz=sim.mains_hz,
                                natural_blinks=sim.natural_blinks, natural_emg=sim.natural_emg)
        self._t_start = 0.0      # wall clock of sample 0
        self._p_start = 0.0      # perf_counter of sample 0 (for drift-free pacing)
        self._seq = 0
        self._auto_idx = 0
        self._next_auto = 0.0
        self.injected: list[tuple[str, float]] = []   # ground truth for tests/tuning

    def start(self) -> None:
        self._t_start, self._p_start = time.time(), time.perf_counter()
        self._next_auto = self.auto.first_after_s

    def read(self, timeout: float = 1.0) -> EEGChunk | None:
        k = self.gen.k
        due = self._p_start + (k + self.chunk) / self.fs   # when the last sample "exists"
        wait = due - time.perf_counter()
        if wait > timeout:
            time.sleep(timeout)
            return None
        if wait > 0:
            time.sleep(wait)
        self._maybe_auto_inject(k / self.fs)
        data = self.gen.generate(self.chunk)
        chunk = EEGChunk(t0=self._t_start + k / self.fs, fs=self.fs, channels=self.channels,
                         data=data, seq=self._seq)
        self._seq += 1
        return chunk

    def _maybe_auto_inject(self, elapsed_s: float) -> None:
        if self.auto.enabled and self.auto.sequence and elapsed_s >= self._next_auto:
            self.inject(self.auto.sequence[self._auto_idx % len(self.auto.sequence)])
            self._auto_idx += 1
            self._next_auto += self.auto.every_s

    def inject(self, gesture: str) -> float | None:
        start = self.gen.inject(gesture)
        onset = self._t_start + start / self.fs
        self.injected.append((gesture, onset))
        return onset

    def stop(self) -> None:
        pass
