"""Acquisition backends and a factory that picks one from config."""

from __future__ import annotations

from bayn.common.config import AppConfig
from bayn.eeg.sources.base import EEGSource
from bayn.eeg.sources.simulated import SimulatedSource


def make_source(cfg: AppConfig) -> EEGSource:
    src = cfg.eeg.source
    if src == "sim":
        return SimulatedSource(cfg.eeg, cfg.simulator)
    if src in ("muse", "lsl"):
        # Real Muse 2 via BrainFlow (and muse-lsl fallback) arrives in Phase 2.
        raise NotImplementedError(f"EEG source '{src}' is implemented in Phase 2; use 'sim' for now.")
    raise ValueError(f"unknown EEG source: {src}")


__all__ = ["EEGSource", "SimulatedSource", "make_source"]
