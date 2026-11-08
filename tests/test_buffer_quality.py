import numpy as np
import pytest

pytestmark = pytest.mark.unit

from bayn.eeg.buffer import RingBuffer
from bayn.eeg.quality import channel_quality, detect_mains
from bayn.eeg.sources.simulated import SimulatedEEG
from tests.conftest import CHANNELS, FS


def test_ringbuffer_wraps_in_order():
    rb = RingBuffer(2, 10)
    for i in range(4):  # 4 x 4 samples = 16 > capacity
        vals = np.arange(i * 4, i * 4 + 4, dtype=np.float32)
        rb.append(np.vstack([vals, -vals]), vals.astype(float))
    data, ts = rb.latest(10)
    assert len(rb) == 10
    np.testing.assert_array_equal(ts, np.arange(6, 16))
    np.testing.assert_array_equal(data[1], -np.arange(6, 16))
    assert rb.last_timestamp == 15


def test_ringbuffer_window_and_oversized_append():
    rb = RingBuffer(1, 5)
    rb.append(np.arange(8, dtype=np.float32)[None, :], np.arange(8.0))
    data, ts = rb.window(4.0, 6.0)
    np.testing.assert_array_equal(ts, [4.0, 5.0])
    assert rb.latest(0)[1].size == 0


def test_quality_levels():
    rng = np.random.default_rng(0)
    x = np.vstack([np.zeros(512), rng.standard_normal(512) * 10,
                   rng.standard_normal(512) * 60, rng.standard_normal(512) * 200])
    levels = [q.level for q in channel_quality(x, CHANNELS)]
    assert levels == ["bad", "good", "ok", "bad"]


def test_quality_ignores_single_blink():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 512)) * 10
    x[0, 200:260] += 150                                   # blink-sized spike
    assert channel_quality(x, ["AF7"])[0].level == "good"


def test_mains_detection_50_vs_60():
    for hz in (50, 60):
        g = SimulatedEEG(CHANNELS, FS, seed=2, mains_hz=hz, natural_blinks=False, natural_emg=False)
        assert detect_mains(g.generate(int(FS * 4)), FS) == hz
    g = SimulatedEEG(CHANNELS, FS, seed=2, mains_hz=0, natural_blinks=False, natural_emg=False)
    assert detect_mains(g.generate(int(FS * 4)), FS) is None
