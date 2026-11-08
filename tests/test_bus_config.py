import time

import numpy as np
import pytest

from bayn.common.bus import Publisher, Subscriber, pack, unpack
from bayn.common.config import load_config
from bayn.common.schemas import EEGChunk, EEGStatus, ChannelQuality


@pytest.mark.unit
def test_pack_roundtrip_ndarray():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = unpack(pack({"a": a, "x": np.float64(1.5)}))
    np.testing.assert_array_equal(out["a"], a)
    assert out["x"] == 1.5


@pytest.mark.unit
def test_schema_roundtrip():
    c = EEGChunk(10.0, 256.0, ["A", "B"], np.ones((2, 32), np.float32), seq=3)
    c2 = EEGChunk.from_msg(unpack(pack(c.to_msg())))
    assert c2.seq == 3 and c2.t_end == pytest.approx(10.125)
    s = EEGStatus("sim", True, 256.0, ["A"], [ChannelQuality("A", "good", 5.0)], mains_hz=50)
    assert EEGStatus.from_msg(unpack(pack(s.to_msg()))) == s


@pytest.mark.integration
def test_pubsub_over_tcp():
    pub = Publisher("127.0.0.1", 0)
    sub = Subscriber("127.0.0.1", pub.port, ["eeg.filt"])
    got = None
    for _ in range(50):  # ZeroMQ "slow joiner": retry until the subscription is live
        pub.send("eeg.raw", {"skip": True})
        pub.send("eeg.filt", {"v": 1})
        got = sub.recv(timeout_ms=50)
        if got:
            break
    pub.close(); sub.close()
    assert got == ("eeg.filt", {"v": 1})


@pytest.mark.integration
def test_threaded_subscriber_feeds_asyncio_default_loop():
    """Regression: zmq.asyncio fails on Windows' Proactor loop; the threaded bridge must not."""
    import asyncio

    from bayn.common.bus import ThreadedSubscriber

    pub = Publisher("127.0.0.1", 0)

    async def main():
        sub = ThreadedSubscriber("127.0.0.1", pub.port, ["eeg.filt"], asyncio.get_running_loop())
        try:
            for _ in range(50):
                pub.send("eeg.filt", {"v": 2})
                try:
                    return await asyncio.wait_for(sub.queue.get(), 0.1)
                except asyncio.TimeoutError:
                    continue
        finally:
            sub.close()

    got = asyncio.run(main())
    pub.close()
    assert got == ("eeg.filt", {"v": 2})


@pytest.mark.unit
def test_env_overrides_typed():
    cfg = load_config(environ={"BAYN_EEG__SOURCE": "muse", "BAYN_EEG__NOTCH_HZ": "60",
                               "BAYN_SIMULATOR__AUTO_GESTURES__ENABLED": "true",
                               "BAYN_UI__PORT": "9001", "UNRELATED": "x"})
    assert cfg.eeg.source == "muse"
    assert cfg.eeg.notch_hz == "60"
    assert cfg.simulator.auto_gestures.enabled is True
    assert cfg.ui.port == 9001


@pytest.mark.unit
def test_bad_notch_rejected():
    with pytest.raises(Exception):
        load_config(environ={"BAYN_EEG__NOTCH_HZ": "55"})
