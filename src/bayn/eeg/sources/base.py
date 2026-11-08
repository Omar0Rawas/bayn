"""The EEGSource interface, plus shared machinery for real (Bluetooth) devices."""

from __future__ import annotations

import logging
import queue
import threading
import time
from abc import ABC, abstractmethod

import numpy as np

from bayn.common.schemas import EEGChunk

log = logging.getLogger("bayn.eeg")


class EEGSource(ABC):
    """A stream of EEG chunks in µV, channel order = `channels`.

    Lifecycle: start() -> read() repeatedly -> stop(). Also usable as a context manager.
    start() must not block on hardware; connection progress is reported via `status_message`.
    """

    name: str = "base"
    supports_injection: bool = False

    def __init__(self, channels: list[str], fs: float) -> None:
        self.channels = list(channels)
        self.fs = float(fs)
        self.status_message = ""

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def read(self, timeout: float = 1.0) -> EEGChunk | None:
        """Block up to `timeout` s for the next chunk; None if nothing arrived."""

    @abstractmethod
    def stop(self) -> None: ...

    def inject(self, gesture: str) -> float | None:
        """Simulator-only: schedule a fake gesture. Returns its onset time (unix s) or None."""
        return None

    def __enter__(self) -> "EEGSource":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


class SampleClock:
    """Turns jittery per-packet device timestamps into a regular sample timeline.

    Downstream code assumes chunk.t0 + i/fs. We anchor to the device clock once and
    count samples; if the device clock drifts away by more than `max_drift_s`
    (dropped packets, reconnect), we re-anchor.
    """

    def __init__(self, fs: float, max_drift_s: float = 0.25) -> None:
        self.fs = fs
        self.max_drift_s = max_drift_s
        self._anchor: float | None = None
        self._k = 0
        self.reanchors = 0

    def reset(self) -> None:
        self._anchor, self._k = None, 0

    def next_t0(self, device_t_first: float, n: int) -> float:
        if self._anchor is None:
            self._anchor, self._k = device_t_first, 0
        t0 = self._anchor + self._k / self.fs
        if abs(device_t_first - t0) > self.max_drift_s:
            log.debug("re-anchoring sample clock (drift %.3f s)", device_t_first - t0)
            self._anchor, self._k, t0 = device_t_first, 0, device_t_first
            self.reanchors += 1
        self._k += n
        return t0


class ThreadedDeviceSource(EEGSource):
    """Base for real devices: a worker thread connects, pulls, and reconnects forever.

    Subclasses implement _connect / _pull / _disconnect; read() just drains a queue.
    """

    poll_interval_s = 0.04
    retry_delay_s = 2.0

    def __init__(self, channels: list[str], fs: float, stale_after_s: float = 3.0) -> None:
        super().__init__(channels, fs)
        self.stale_after_s = stale_after_s
        self._q: queue.Queue[EEGChunk] = queue.Queue(maxsize=2000)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._clock = SampleClock(fs)
        self._seq = 0
        self.connected = False

    # -- subclass hooks (all run on the worker thread)
    @abstractmethod
    def _connect(self) -> None:
        """Open the device; raise on failure. May update self.fs / self.channels."""

    @abstractmethod
    def _pull(self) -> tuple[np.ndarray, np.ndarray] | None:
        """New samples since last call: (data (n_ch, n) µV, device unix timestamps (n,)) or None."""

    @abstractmethod
    def _disconnect(self) -> None: ...

    # -- EEGSource API
    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, name=f"{self.name}-worker", daemon=True)
        self._thread.start()

    def read(self, timeout: float = 1.0) -> EEGChunk | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self, join_timeout: float = 25.0) -> None:
        """Ask the worker to finish. A BLE connect in progress cannot be interrupted, so we
        wait for it (up to the discovery timeout) to release the headset cleanly."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=join_timeout)

    # -- worker
    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                self.status_message = self._searching_message()
                self._connect()
            except Exception as e:  # device not found, BLE busy, ...
                if self._stop.is_set():   # stop() while a blocking connect was in progress
                    return
                self.status_message = f"{self._searching_message()} ({_short(e)})"
                log.warning("connect failed: %s - retrying in %.0f s", _short(e), self.retry_delay_s)
                self._stop.wait(self.retry_delay_s)
                continue
            self.connected = True
            self.status_message = "connected"
            self._clock = SampleClock(self.fs)
            log.info("%s connected (%s @ %.0f Hz)", self.name, ",".join(self.channels), self.fs)
            last_data = time.monotonic()
            try:
                while not self._stop.is_set():
                    got = self._pull()
                    if got is not None and got[0].shape[1] > 0:
                        self._emit(*got)
                        last_data = time.monotonic()
                    elif time.monotonic() - last_data > self.stale_after_s:
                        raise TimeoutError(f"no data for {self.stale_after_s:.0f} s")
                    self._stop.wait(self.poll_interval_s)
            except Exception as e:
                log.warning("%s connection lost: %s - reconnecting", self.name, _short(e))
                self.status_message = "connection lost - reconnecting"
            finally:
                self.connected = False
                try:
                    self._disconnect()
                except Exception as e:  # best effort
                    log.debug("disconnect error: %s", e)

    def _emit(self, data: np.ndarray, device_ts: np.ndarray) -> None:
        t0 = self._clock.next_t0(float(device_ts[0]), data.shape[1])
        chunk = EEGChunk(t0=t0, fs=self.fs, channels=self.channels,
                         data=np.ascontiguousarray(data, dtype=np.float32), seq=self._seq)
        self._seq += 1
        if self._q.full():  # consumer stalled: drop oldest, keep latency bounded
            self._q.get_nowait()
        self._q.put_nowait(chunk)

    def _searching_message(self) -> str:
        return "searching for device..."


def _short(e: BaseException) -> str:
    s = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
    return s[:160]
