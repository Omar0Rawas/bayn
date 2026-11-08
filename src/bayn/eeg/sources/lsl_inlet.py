"""Read EEG from a Lab Streaming Layer stream (fallback: `muselsl stream`).

muse-lsl publishes 5 channels (TP9, AF7, AF8, TP10, Right AUX); we select the
configured channels by label. Timestamps may be LSL-clock or unix (muse-lsl); both
are detected and converted to unix time.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from bayn.common.config import EEGConfig
from bayn.eeg.sources.base import ThreadedDeviceSource

log = logging.getLogger("bayn.eeg")


def _channel_labels(info) -> list[str]:
    labels, ch = [], info.desc().child("channels").child("channel")
    for _ in range(info.channel_count()):
        labels.append(ch.child_value("label"))
        ch = ch.next_sibling()
    return labels


def _clock_offset_for(sample_ts: float) -> float:
    """Offset that turns a stream's timestamps into unix time.

    Standard LSL outlets stamp samples with pylsl.local_clock() (seconds since boot), but
    muse-lsl stamps them with unix time.time() itself. Whichever clock the newest sample
    is closest to wins.
    """
    import pylsl

    now_unix, now_lsl = time.time(), pylsl.local_clock()
    if abs(sample_ts - now_unix) < abs(sample_ts - now_lsl):
        log.info("LSL stream uses unix timestamps (muse-lsl style)")
        return 0.0
    return now_unix - now_lsl


class LSLSource(ThreadedDeviceSource):
    name = "lsl"

    def __init__(self, eeg: EEGConfig) -> None:
        super().__init__(eeg.channels, eeg.fs, stale_after_s=eeg.lsl.stale_after_s)
        self.cfg = eeg.lsl
        self._inlet = None
        self._idx: list[int] = []
        self._clock_offset = 0.0     # unix time - LSL local_clock

    def _searching_message(self) -> str:
        what = self.cfg.stream_name or f"type={self.cfg.stream_type}"
        return f"waiting for LSL stream ({what}) - is `muselsl stream` running?"

    def _connect(self) -> None:
        import pylsl

        prop, value = ("name", self.cfg.stream_name) if self.cfg.stream_name else ("type", self.cfg.stream_type)
        streams = pylsl.resolve_byprop(prop, value, timeout=self.cfg.resolve_timeout_s)
        if not streams:
            raise TimeoutError("no LSL stream found")
        self._inlet = pylsl.StreamInlet(streams[0], max_buflen=60,
                                        processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter)
        info = self._inlet.info(timeout=5.0)
        labels = [l.strip().upper() for l in _channel_labels(info)]
        wanted = [c.upper() for c in self.channels]
        if all(w in labels for w in wanted):
            self._idx = [labels.index(w) for w in wanted]
        else:
            log.warning("LSL labels %s do not match %s - using the first %d channels",
                        labels, self.channels, len(wanted))
            self._idx = list(range(len(wanted)))
        if info.nominal_srate() > 0:
            self.fs = float(info.nominal_srate())
        self._clock_offset = None     # decided from the first samples, see _pull
        log.info("LSL stream '%s' (%d ch @ %.0f Hz)", info.name(), info.channel_count(), self.fs)

    def _pull(self) -> tuple[np.ndarray, np.ndarray] | None:
        samples, ts = self._inlet.pull_chunk(timeout=0.0, max_samples=1024)
        if not ts:
            return None
        ts = np.asarray(ts)
        if self._clock_offset is None:
            self._clock_offset = _clock_offset_for(float(ts[-1]))
        data = np.asarray(samples, dtype=np.float32).T[self._idx]
        return data, ts + self._clock_offset

    def _disconnect(self) -> None:
        if self._inlet is not None:
            self._inlet.close_stream()
            self._inlet = None
