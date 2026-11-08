"""Acquisition backends and a factory that picks one from config."""

from __future__ import annotations

from bayn.common.config import AppConfig
from bayn.eeg.sources.base import EEGSource
from bayn.eeg.sources.simulated import SimulatedSource


def make_source(cfg: AppConfig) -> EEGSource:
    src = cfg.eeg.source
    if src == "sim":
        return SimulatedSource(cfg.eeg, cfg.simulator)
    if src == "muse":
        from bayn.eeg.sources.brainflow_muse import BrainFlowSource  # imported lazily: native libs
        return BrainFlowSource(cfg.eeg)
    if src == "lsl":
        from bayn.eeg.sources.lsl_inlet import LSLSource
        return LSLSource(cfg.eeg)
    raise ValueError(f"unknown EEG source: {src}")


__all__ = ["EEGSource", "SimulatedSource", "make_source"]
