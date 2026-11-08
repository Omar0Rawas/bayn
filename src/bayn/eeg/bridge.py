"""EEG bridge: source -> filters -> rolling buffer -> ZeroMQ publish.

Runs natively on the host (Bluetooth is not reachable from Docker on Windows/macOS).

    python -m bayn.eeg.bridge [--source sim|muse|lsl] [--auto-gestures]
"""

from __future__ import annotations

import argparse
import os
import threading
import time

from bayn.common.bus import CommandReceiver, Publisher
from bayn.common.config import AppConfig, load_config
from bayn.common.log import setup_logging
from bayn.common.schemas import (GESTURES, TOPIC_EEG_FILT, TOPIC_EEG_INJECTED, TOPIC_EEG_RAW,
                                 TOPIC_EEG_STATUS, EEGChunk, EEGStatus)
from bayn.eeg.buffer import RingBuffer
from bayn.eeg.filters import StreamingFilter
from bayn.eeg.quality import channel_quality, detect_mains
from bayn.eeg.sources import EEGSource, make_source

log = setup_logging(name="bayn.eeg")

STATUS_EVERY_S = 1.0
QUALITY_WINDOW_S = 2.0
MAINS_WINDOW_S = 4.0


class Bridge:
    def __init__(self, cfg: AppConfig, source: EEGSource | None = None) -> None:
        self.cfg = cfg
        self.source = source or make_source(cfg)
        e, b = cfg.eeg, cfg.bus
        n_ch, fs = len(self.source.channels), self.source.fs
        self.filter = StreamingFilter(fs, n_ch, tuple(e.bandpass_hz), e.notch_hz)
        cap = int(fs * e.buffer_seconds)
        self.raw_buf = RingBuffer(n_ch, cap)
        self.filt_buf = RingBuffer(n_ch, cap)
        self.pub = Publisher(b.bind_host, b.eeg_pub_port)
        self.cmd = CommandReceiver(b.bind_host, b.eeg_cmd_port)
        self.mains_hz: int | None = None
        self._last_status = 0.0
        self._last_mains = 0.0
        self._connected = False

    # ------------------------------------------------------------ main loop
    def run(self, stop: threading.Event | None = None) -> None:
        stop = stop or threading.Event()
        log.info("starting source '%s' (%s @ %.0f Hz), notch=%s, publishing on :%d",
                 self.source.name, ",".join(self.source.channels), self.source.fs,
                 self.cfg.eeg.notch_hz, self.pub.port)
        self.source.start()
        try:
            while not stop.is_set():
                chunk = self.source.read(timeout=1.0)
                if chunk is not None:
                    self._handle_chunk(chunk)
                self._connected = chunk is not None
                self._handle_commands()
                self._maybe_status()
        finally:
            self.source.stop()
            self.pub.close()
            self.cmd.close()
            log.info("bridge stopped")

    def _handle_chunk(self, chunk: EEGChunk) -> None:
        filtered = self.filter.process(chunk.data)
        ts = chunk.timestamps()
        self.raw_buf.append(chunk.data, ts)
        self.filt_buf.append(filtered, ts)
        self.pub.send(TOPIC_EEG_RAW, chunk.to_msg())
        self.pub.send(TOPIC_EEG_FILT, EEGChunk(chunk.t0, chunk.fs, chunk.channels, filtered, chunk.seq).to_msg())

    def _handle_commands(self) -> None:
        for c in self.cmd.poll():
            if c.get("cmd") == "inject":
                g = c.get("gesture", "")
                onset = self.source.inject(g) if g in GESTURES else None
                log.info("inject %s -> %s", g, "ok" if onset else "not supported by this source")
                self.pub.send(TOPIC_EEG_INJECTED, {"gesture": g, "t": onset, "ok": onset is not None})

    def _maybe_status(self) -> None:
        now = time.time()
        if now - self._last_status < STATUS_EVERY_S:
            return
        self._last_status = now
        fs = self.source.fs
        quality = []
        if self._connected and len(self.filt_buf) >= fs * QUALITY_WINDOW_S:
            filt, _ = self.filt_buf.latest(int(fs * QUALITY_WINDOW_S))
            quality = channel_quality(filt, self.source.channels)
        if now - self._last_mains > 5.0 and len(self.raw_buf) >= fs * MAINS_WINDOW_S:
            self._last_mains = now
            raw, _ = self.raw_buf.latest(int(fs * MAINS_WINDOW_S))
            detected = detect_mains(raw - raw.mean(axis=1, keepdims=True), fs)
            if detected and detected != self.mains_hz:
                log.info("mains hum detected at %d Hz", detected)
            self.mains_hz = detected or self.mains_hz
        status = EEGStatus(
            source=self.source.name, connected=self._connected, fs=fs, channels=self.source.channels,
            quality=quality, mains_hz=self.mains_hz, notch=self.cfg.eeg.notch_hz,
            supports_injection=self.source.supports_injection, t=now,
            message="" if self._connected else "waiting for EEG data...",
        )
        self.pub.send(TOPIC_EEG_STATUS, status.to_msg())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Bayn EEG bridge")
    ap.add_argument("--source", choices=["sim", "muse", "lsl"], help="override eeg.source")
    ap.add_argument("--auto-gestures", action="store_true", help="simulator: inject scripted gestures")
    args = ap.parse_args(argv)
    if args.source:
        os.environ["BAYN_EEG__SOURCE"] = args.source
    if args.auto_gestures:
        os.environ["BAYN_SIMULATOR__AUTO_GESTURES__ENABLED"] = "true"
    cfg = load_config()
    setup_logging(cfg.logging.level)
    try:
        Bridge(cfg).run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
