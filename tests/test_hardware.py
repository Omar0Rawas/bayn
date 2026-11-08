"""Hardware-in-the-loop tests with a real Muse 2. Skipped unless BAYN_HARDWARE=1.

    $env:BAYN_HARDWARE=1; .\.venv\Scripts\python.exe -m pytest -m hardware -v     # Windows
    BAYN_HARDWARE=1 pytest -m hardware -v                                       # macOS/Linux

Wear the headset (sensors damp), sit still, phone app closed. Optional:
BAYN_EEG__BRAINFLOW__SERIAL_NUMBER=Muse-XXXX to pick a specific headset.
For the LSL test, `muselsl stream` must already be running.
"""

import os
import time

import numpy as np
import pytest

from bayn.common.config import load_config
from bayn.eeg.filters import StreamingFilter
from bayn.eeg.quality import channel_quality
from bayn.eeg.sources import make_source

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(os.environ.get("BAYN_HARDWARE") != "1", reason="set BAYN_HARDWARE=1 with a Muse 2 nearby"),
]


def _record(source: str, seconds: float = 8.0, connect_timeout: float = 60.0):
    os.environ["BAYN_EEG__SOURCE"] = source
    cfg = load_config()
    src = make_source(cfg)
    src.start()
    lags = []   # receive time - timestamp of the chunk's last sample
    try:
        chunks, end = [], time.monotonic() + connect_timeout
        while not chunks and time.monotonic() < end:
            if (c := src.read(timeout=1.0)) is not None:
                chunks.append(c)
        assert chunks, f"no data from {source} within {connect_timeout:.0f} s"
        rec_end = time.monotonic() + seconds
        while time.monotonic() < rec_end:
            if (c := src.read(timeout=1.0)) is not None:
                lags.append(time.time() - c.t_end)
                chunks.append(c)
    finally:
        src.stop()   # (BrainFlow's BLE release alone takes several seconds)
    return cfg, src, chunks, np.asarray(lags)


def _assert_good_stream(cfg, src, chunks, lags):
    raw = np.hstack([c.data for c in chunks])
    span = chunks[-1].t_end - chunks[0].t0
    rate = raw.shape[1] / span
    assert abs(rate - 256) < 3, f"effective rate {rate:.1f} Hz"
    assert src.channels == ["TP9", "AF7", "AF8", "TP10"]
    # timestamps are unix time and data arrives promptly (BLE packets ~ every 50 ms)
    med = float(np.median(lags))
    print(f"\n{src.name}: rate {rate:.1f} Hz, latency median {med * 1000:.0f} ms, max {lags.max() * 1000:.0f} ms")
    assert -0.5 < med < 0.5, f"median latency {med:.2f} s - timestamps wrong?"
    filt = StreamingFilter(256, 4, tuple(cfg.eeg.bandpass_hz), cfg.eeg.notch_hz).process(raw)[:, 256:]
    q = channel_quality(filt, src.channels)
    assert sum(x.level != "bad" for x in q) >= 3, [(x.channel, x.level, round(x.noise_uv, 1)) for x in q]


def test_muse_via_brainflow():
    _assert_good_stream(*_record("muse"))


def test_muse_via_lsl():
    _assert_good_stream(*_record("lsl", connect_timeout=20))
