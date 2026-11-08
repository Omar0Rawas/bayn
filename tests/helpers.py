"""Shared helpers: drive detectors / the event manager with simulated EEG, offline and fast."""

from __future__ import annotations

from typing import Callable, Iterator

import numpy as np

from bayn.common.schemas import EEGChunk
from bayn.eeg.filters import StreamingFilter
from bayn.eeg.sources.simulated import SimulatedEEG

CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
FS = 256.0
CHUNK = 32
T0 = 1_800_000_000.0          # fake unix start time


def sim_chunks(gen: SimulatedEEG, seconds: float, t0: float = T0,
               on_time: Callable[[float], None] | None = None) -> Iterator[tuple[EEGChunk, EEGChunk]]:
    """Yield (raw, filtered) chunk pairs like the bridge publishes them.

    on_time(t_rel) is called before each chunk - used to inject gestures at known times.
    """
    filt = StreamingFilter(FS, len(gen.channels))
    k0 = gen.k
    while (gen.k - k0) / FS < seconds:
        t_rel = (gen.k - k0) / FS
        if on_time:
            on_time(t_rel)
        x = gen.generate(CHUNK)
        t = t0 + t_rel
        yield (EEGChunk(t, FS, gen.channels, x, gen.k // CHUNK),
               EEGChunk(t, FS, gen.channels, filt.process(x), gen.k // CHUNK))


def schedule_injections(gen: SimulatedEEG, gesture: str | None, n: int, first_s: float, every_s: float,
                        t0: float = T0) -> tuple[list[float], Callable[[float], None]]:
    """Returns (onset times filled in as injections happen, on_time callback)."""
    onsets: list[float] = []
    k0 = gen.k
    state = {"next": first_s}

    def on_time(t_rel: float) -> None:
        if gesture and len(onsets) < n and t_rel >= state["next"]:
            start = gen.inject(gesture, delay_s=0)
            onsets.append(t0 + (start - k0) / FS)
            state["next"] += every_s

    return onsets, on_time


def match(detected: list[float], truth: list[float], lo: float = -0.6, hi: float = 1.5) -> tuple[int, int, list[float]]:
    """(hits, false positives, onset errors) for detection times vs true onsets."""
    hits = sum(any(lo <= d - t <= hi for d in detected) for t in truth)
    fps = sum(not any(lo <= d - t <= hi for t in truth) for d in detected)
    errs = [d - t for d in detected for t in truth if lo <= d - t <= hi]
    return hits, fps, errs


def rng_seeds(n: int) -> list[int]:
    return list(np.arange(n) + 100)
