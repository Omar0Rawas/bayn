"""Events service: EEG stream -> EventManager -> events on the bus.

    python -m bayn.events.service

Subscribes to the bridge (eeg.raw + eeg.filt + eeg.status), accepts commands
(manual trigger, calibration, gesture choice) and publishes trigger status,
events, saved windows and calibration prompts.
"""

from __future__ import annotations

import logging
import time

from bayn.common.bus import CommandReceiver, CommandSender, Publisher, Subscriber
from bayn.common.config import AppConfig, load_config
from bayn.common.log import setup_logging
from bayn.common.schemas import (TOPIC_CALIBRATION, TOPIC_EEG_FILT, TOPIC_EEG_RAW, TOPIC_EEG_STATUS,
                                 TOPIC_EVENTS_STATUS, EEGChunk)
from bayn.events.detectors import GESTURE_CHOICES
from bayn.events.manager import INJECT, EventManager
from bayn.events.store import EventStore

log = logging.getLogger("bayn.events")
STATUS_EVERY_S = 0.25


class EventsService:
    def __init__(self, cfg: AppConfig, store: EventStore | None = None) -> None:
        self.cfg = cfg
        b = cfg.bus
        self.sub = Subscriber(b.eeg_host, b.eeg_pub_port, [TOPIC_EEG_RAW, TOPIC_EEG_FILT, TOPIC_EEG_STATUS])
        self.pub = Publisher(b.bind_host, b.events_pub_port)
        self.cmd = CommandReceiver(b.bind_host, b.events_cmd_port)
        self.bridge = CommandSender(b.eeg_host, b.eeg_cmd_port)
        self.store = store or EventStore()
        self.manager: EventManager | None = None
        self.eeg_source = cfg.eeg.source
        self.can_inject = False
        self._raw: dict[int, EEGChunk] = {}       # raw chunks waiting for their filtered twin
        self._last_status = 0.0

    def _ensure_manager(self, fs: float | None = None, channels: list[str] | None = None) -> EventManager:
        fs = fs or (self.manager.fs if self.manager else self.cfg.eeg.fs)
        channels = channels or (self.manager.channels if self.manager else self.cfg.eeg.channels)
        m = self.manager
        if m is None or m.fs != fs or m.channels != channels or m.source != self.eeg_source:
            gesture = m.gesture if m else self.cfg.events.gesture
            self.manager = EventManager(self.cfg.events, fs, channels, self.store, source=self.eeg_source)
            if gesture != self.manager.gesture:
                self.manager.set_gesture(gesture)
        return self.manager

    def _publish(self, msgs: list[tuple[str, dict]]) -> None:
        for topic, payload in msgs:
            if topic == INJECT:
                if self.can_inject:
                    self.bridge.send({"cmd": "inject", "gesture": payload["gesture"]})
            else:
                self.pub.send(topic, payload)

    def _handle_command(self, c: dict) -> None:
        m = self._ensure_manager()
        cmd = c.get("cmd")
        if cmd == "trigger":
            self._publish(m.trigger_manual(c.get("source", "button")))
        elif cmd == "calibrate":
            if c.get("action") == "cancel":
                self._publish(m.cancel_calibration())
            elif m.data_now == 0:
                self.pub.send(TOPIC_CALIBRATION, {"state": "failed", "t": time.time(),
                                                  "prompt": "No EEG signal yet - connect the headset first"})
            else:
                self._publish(m.start_calibration())
        elif cmd == "set_gesture" and c.get("gesture") in GESTURE_CHOICES:
            m.set_gesture(c["gesture"])
        else:
            log.warning("unknown command %s", c)

    def run(self) -> None:
        log.info("events service: gesture=%s cooldown=%.0fs window=-%.0f/+%.0fs, publishing on :%d",
                 self.cfg.events.gesture, self.cfg.events.cooldown_s, self.cfg.events.pre_s,
                 self.cfg.events.post_s, self.pub.port)
        try:
            while True:
                msg = self.sub.recv(timeout_ms=100)
                if msg is not None:
                    self._on_bus(*msg)
                for c in self.cmd.poll():
                    self._handle_command(c)
                now = time.time()
                if self.manager:
                    self._publish(self.manager.tick(now))
                if now - self._last_status >= STATUS_EVERY_S:
                    self._last_status = now
                    st = self._ensure_manager().status(now)
                    st["eeg_source"], st["sim"] = self.eeg_source, self.can_inject
                    self.pub.send(TOPIC_EVENTS_STATUS, st)
        finally:
            self.sub.close(); self.pub.close(); self.cmd.close(); self.bridge.close()

    def _on_bus(self, topic: str, m: dict) -> None:
        if topic == TOPIC_EEG_STATUS:
            self.eeg_source = m.get("source", self.eeg_source)
            self.can_inject = bool(m.get("supports_injection"))
        elif topic == TOPIC_EEG_RAW:
            c = EEGChunk.from_msg(m)
            self._raw[c.seq] = c
            if len(self._raw) > 50:                   # filtered twin never came: drop old
                for k in sorted(self._raw)[:-50]:
                    del self._raw[k]
        elif topic == TOPIC_EEG_FILT:
            f = EEGChunk.from_msg(m)
            raw = self._raw.pop(f.seq, None)
            if raw is not None and abs(raw.t0 - f.t0) < 1e-6:
                mgr = self._ensure_manager(raw.fs, raw.channels)
                self._publish(mgr.on_chunk(raw, f))


def main() -> None:
    cfg = load_config()
    setup_logging(cfg.logging.level)
    try:
        EventsService(cfg).run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
