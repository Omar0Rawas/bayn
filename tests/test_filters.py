import numpy as np
import pytest

from bayn.eeg.filters import StreamingFilter, notch_frequencies

pytestmark = pytest.mark.unit
FS = 256.0


def _sine(f: float, secs: float = 8.0, amp: float = 10.0) -> np.ndarray:
    t = np.arange(int(FS * secs)) / FS
    return np.tile(amp * np.sin(2 * np.pi * f * t), (4, 1))


def _gain(f: float, notch: str = "auto") -> float:
    y = StreamingFilter(FS, 4, notch=notch).process(_sine(f))[:, int(FS * 3):]
    return float(y.std() / (10 / np.sqrt(2)))


def test_passband_kept():
    for f in (5, 10, 20):
        assert 0.85 < _gain(f) < 1.1, f


def test_mains_removed_both_50_and_60_in_auto():
    assert _gain(50) < 0.03
    assert _gain(60) < 0.03


def test_dc_offset_and_drift_removed():
    t = np.arange(int(FS * 10)) / FS
    x = np.tile(800 + 30 * np.sin(2 * np.pi * 0.05 * t), (4, 1))
    y = StreamingFilter(FS, 4).process(x)
    assert np.abs(y).max() < 5      # no startup transient from the 800 µV offset


def test_streaming_matches_single_pass():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 2048)) * 20
    whole = StreamingFilter(FS, 4).process(x)
    f = StreamingFilter(FS, 4)
    parts = np.hstack([f.process(x[:, i:i + 32]) for i in range(0, 2048, 32)])
    np.testing.assert_allclose(whole, parts, atol=1e-3)


def test_notch_settings():
    assert notch_frequencies("auto", FS) == [50.0, 60.0]
    assert notch_frequencies("60", FS) == [60.0]
    assert notch_frequencies("off", FS) == []
    assert notch_frequencies("60", 100.0) == []     # above Nyquist -> skipped
