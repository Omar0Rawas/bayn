import numpy as np
import pytest

from bayn.eeg.sources.simulated import SimulatedEEG

CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
FS = 256.0


@pytest.fixture
def gen() -> SimulatedEEG:
    """Deterministic simulator without random distractors."""
    return SimulatedEEG(CHANNELS, FS, seed=1, mains_hz=50, natural_blinks=False, natural_emg=False)


def band_rms(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    from bayn.eeg.filters import bandpass_offline

    return np.sqrt(np.mean(bandpass_offline(x, fs, lo, hi) ** 2, axis=-1))
