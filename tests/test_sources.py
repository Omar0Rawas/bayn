"""Real-device source plumbing, tested without a headset."""

import threading
import time

import numpy as np
import pytest

from bayn.common.config import EEGConfig, LSLConfig, BrainFlowConfig, load_config
from bayn.eeg.sources import make_source
from bayn.eeg.sources.base import SampleClock


def _collect(src, seconds: float, connect_timeout: float = 20.0):
    src.start()
    try:
        chunks, t_end = [], time.monotonic() + connect_timeout
        while not chunks and time.monotonic() < t_end:
            if (c := src.read(timeout=0.5)) is not None:
                chunks.append(c)
        t_rec = time.monotonic() + seconds
        while chunks and time.monotonic() < t_rec:
            if (c := src.read(timeout=0.5)) is not None:
                chunks.append(c)
        return chunks
    finally:
        src.stop()


# ---------------------------------------------------------------- SampleClock
@pytest.mark.unit
def test_sample_clock_regularises_jitter():
    clk = SampleClock(256.0)
    rng = np.random.default_rng(0)
    t0s, k = [], 0
    for _ in range(50):
        jitter = rng.uniform(-0.02, 0.02)
        t0s.append(clk.next_t0(1000.0 + k / 256.0 + jitter, 12))
        k += 12
    np.testing.assert_allclose(np.diff(t0s), 12 / 256.0)      # perfectly regular
    assert clk.reanchors == 0


@pytest.mark.unit
def test_sample_clock_reanchors_after_gap():
    clk = SampleClock(256.0)
    clk.next_t0(1000.0, 256)
    t0 = clk.next_t0(1005.0, 256)                              # 4 s of missing data
    assert t0 == 1005.0 and clk.reanchors == 1


# ---------------------------------------------------------------- simulated source (real-time)
@pytest.mark.integration
def test_simulated_source_paces_in_real_time_and_records_ground_truth():
    from bayn.common.config import SimulatorConfig
    from bayn.eeg.sources.simulated import SimulatedSource

    src = SimulatedSource(EEGConfig(chunk_samples=32), SimulatorConfig(seed=3))
    src.start()
    t = time.perf_counter()
    chunks = [src.read() for _ in range(8)]                 # 8 x 125 ms
    elapsed = time.perf_counter() - t
    assert 0.8 <= elapsed <= 1.5
    assert [c.seq for c in chunks] == list(range(8))
    assert abs(chunks[1].t0 - chunks[0].t_end) < 1e-9       # contiguous timestamps
    onset = src.inject("clench")
    assert src.injected == [("clench", onset)]


# ---------------------------------------------------------------- BrainFlow (synthetic board)
@pytest.mark.integration
def test_brainflow_source_with_synthetic_board():
    pytest.importorskip("brainflow")
    eeg = EEGConfig(source="muse", brainflow=BrainFlowConfig(board="SYNTHETIC_BOARD"))
    cfg = load_config(environ={})
    cfg.eeg = eeg
    src = make_source(cfg)
    chunks = _collect(src, 1.5)
    assert chunks, "synthetic board produced no data"
    assert all(c.data.shape[0] == 4 for c in chunks)
    assert src.fs == 250.0                                      # synthetic board rate
    total = sum(c.n_samples for c in chunks)
    assert 200 < total < 700
    for a, b in zip(chunks, chunks[1:]):                        # contiguous timeline
        assert abs(b.t0 - a.t_end) < 0.3
    assert abs(chunks[-1].t_end - time.time()) < 5             # unix timestamps


@pytest.mark.unit
def test_unknown_brainflow_board_rejected():
    pytest.importorskip("brainflow")
    from bayn.eeg.sources.brainflow_muse import BrainFlowSource

    with pytest.raises(ValueError):
        BrainFlowSource(EEGConfig(brainflow=BrainFlowConfig(board="NOT_A_BOARD")))


# ---------------------------------------------------------------- LSL (fake muselsl outlet)
@pytest.mark.integration
@pytest.mark.parametrize("unix_stamps", [False, True], ids=["lsl-clock", "unix-clock-like-muselsl"])
def test_lsl_source_selects_channels_by_label(unix_stamps):
    pylsl = pytest.importorskip("pylsl")
    name = f"BaynTest-{time.time_ns()}"
    info = pylsl.StreamInfo(name, "EEG", 5, 256, "float32", name)
    chs = info.desc().append_child("channels")
    labels = ["TP9", "AF7", "AF8", "TP10", "Right AUX"]
    for lab in labels:
        chs.append_child("channel").append_child_value("label", lab)
    outlet = pylsl.StreamOutlet(info, chunk_size=12)

    stop = threading.Event()

    def push():  # channel i carries the constant value i*10 so we can check the mapping
        row = [float(i * 10) for i in range(5)]
        while not stop.is_set():
            if unix_stamps:   # muse-lsl stamps each sample with time.time()
                now = time.time()
                outlet.push_chunk([row] * 12, [now - (11 - i) / 256 for i in range(12)])
            else:             # standard LSL: outlet stamps with local_clock()
                outlet.push_chunk([row] * 12)
            time.sleep(12 / 256)

    threading.Thread(target=push, daemon=True).start()
    try:
        # config channel order deliberately differs from the stream's order
        eeg = EEGConfig(source="lsl", channels=["AF8", "TP9", "TP10", "AF7"],
                        lsl=LSLConfig(stream_name=name, resolve_timeout_s=5))
        cfg = load_config(environ={})
        cfg.eeg = eeg
        chunks = _collect(make_source(cfg), 1.0, connect_timeout=15)
    finally:
        stop.set()
    if not chunks:
        pytest.skip("LSL stream discovery not available on this machine/network")
    data = np.hstack([c.data for c in chunks])
    np.testing.assert_array_equal(data[:, -1], [20.0, 0.0, 30.0, 10.0])   # AF8, TP9, TP10, AF7
    assert chunks[0].fs == 256.0
    assert abs(chunks[-1].t_end - time.time()) < 2           # unix time either way
