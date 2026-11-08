"""Event manager: detections -> cooldown -> event -> saved EEG window -> pipeline.

Pure logic, driven by EEG data time (chunk timestamps), so tests can feed minutes of
simulated data in milliseconds. The service process (service.py) wires it to the bus.

Every public method returns a list of (topic, payload) messages to publish.
A special topic "_inject" asks the service to make the simulator perform a gesture.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bayn.common.config import DATA_DIR, EventsConfig
from bayn.common.schemas import TOPIC_CALIBRATION, TOPIC_EVENT, TOPIC_EVENT_DONE, EEGChunk
from bayn.eeg.buffer import RingBuffer
from bayn.events.calibration import CalibrationSession
from bayn.events.detectors import Detection, Detector, ManualDetector, make_gesture_detectors
from bayn.events.stats import event_window_stats
from bayn.events.store import EventStore

log = logging.getLogger("bayn.events")
Out = list[tuple[str, dict]]
INJECT = "_inject"
STALE_GRACE_S = 5.0          # finalize by wall clock if EEG stopped arriving

LABELS = {"clench": "Jaw clench", "double_blink": "Double blink", "manual": "Manual trigger",
          "seizure_stub": "Seizure detector"}


@dataclass(slots=True)
class Pending:
    meta: dict
    finalize_at: float


class EventManager:
    def __init__(self, cfg: EventsConfig, fs: float, channels: list[str], store: EventStore | None = None,
                 source: str = "sim", profiles_dir: Path | None = None) -> None:
        self.cfg = cfg
        self.fs = float(fs)
        self.channels = list(channels)
        self.store = store
        self.source = source
        self.profiles_dir = Path(profiles_dir or DATA_DIR / "profiles")
        cap = int((cfg.pre_s + cfg.post_s + 20) * fs)
        self.raw_buf = RingBuffer(len(channels), cap)
        self.filt_buf = RingBuffer(len(channels), cap)
        self.manual = ManualDetector(fs, channels)
        self.gesture = cfg.gesture
        self.detectors: list[Detector] = []
        self.set_gesture(cfg.gesture)
        self.pending: list[Pending] = []
        self.last_event_t = -1e18
        self.last_manual_t = -1e18
        self.suppressed = 0            # gestures ignored because of the cooldown
        self.calibration: CalibrationSession | None = None
        self.last_calibration: dict | None = None
        self.data_now = 0.0
        self._last_end: float | None = None

    # ------------------------------------------------------------ configuration
    def set_gesture(self, gesture: str) -> None:
        self.detectors = make_gesture_detectors(gesture, self.cfg, self.fs, self.channels)
        self.gesture = gesture
        self.load_profile()
        log.info("trigger gesture: %s", gesture)

    @property
    def profile_path(self) -> Path:
        return self.profiles_dir / f"{self.cfg.profile}-{self.source}.json"

    def load_profile(self) -> dict | None:
        try:
            prof = json.loads(self.profile_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        for d in self.detectors:
            if d.name in prof.get("detectors", {}):
                d.apply_profile(prof["detectors"][d.name])
        return prof

    def _save_profile(self, results: dict) -> None:
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        try:
            prof = json.loads(self.profile_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            prof = {"detectors": {}}
        prof["detectors"].update(results)
        prof.update({"profile": self.cfg.profile, "source": self.source, "updated": time.time()})
        self.profile_path.write_text(json.dumps(prof, indent=2), encoding="utf-8")

    # ------------------------------------------------------------ data
    def on_chunk(self, raw: EEGChunk, filt: EEGChunk) -> Out:
        out: Out = []
        if self._last_end is not None and abs(raw.t0 - self._last_end) > 0.5:
            for d in self.detectors:          # signal gap (reconnect): restart filters/state
                d.reset()
        self._last_end = raw.t_end
        ts = raw.timestamps()
        self.raw_buf.append(raw.data, ts)
        self.filt_buf.append(filt.data, ts)
        self.data_now = raw.t_end

        detections: list[Detection] = []
        for d in self.detectors:
            detections += d.update(raw, filt)   # always update (keeps levels live during calibration)
        if self.calibration:
            out += self._step_calibration(self.data_now)
        else:
            for det in detections:
                out += self._on_detection(det)
        out += self._finalize_ready(self.data_now)
        return out

    def tick(self, wall_now: float) -> Out:
        """Called periodically by the service: finalize events if EEG stopped flowing."""
        return self._finalize_ready(self.data_now, wall_now)

    # ------------------------------------------------------------ triggering
    def trigger_manual(self, source: str = "button", t: float | None = None) -> Out:
        t = t if t is not None else time.time()
        if t - self.last_manual_t < self.cfg.manual_debounce_s:
            return []
        self.last_manual_t = t
        self.manual.fire(source, t)
        out: Out = []
        for det in self.manual.drain():
            out += self._create_event(det, source=source)
        return out

    def _on_detection(self, det: Detection) -> Out:
        if det.t - self.last_event_t < self.cfg.cooldown_s:
            self.suppressed += 1
            log.info("%s ignored (cooldown)", det.detector)
            return []
        return self._create_event(det, source="gesture")

    def _create_event(self, det: Detection, source: str) -> Out:
        self.last_event_t = det.t
        lt = time.localtime(det.t)
        eid = time.strftime("evt-%Y%m%d-%H%M%S", lt) + f"-{int((det.t % 1) * 1000):03d}"
        meta = {
            "id": eid, "t": det.t, "detector": det.detector, "label": LABELS.get(det.detector, det.detector),
            "kind": det.kind, "source": source, "confidence": round(det.confidence, 3), "info": det.info,
            "pre_s": self.cfg.pre_s, "post_s": self.cfg.post_s, "eeg_source": self.source,
            "state": "recording", "created": time.time(),
        }
        self.pending.append(Pending(meta, det.t + self.cfg.post_s))
        if self.store:
            self.store.upsert(meta)
        log.info("EVENT %s by %s (%s)", eid, det.detector, source)
        return [(TOPIC_EVENT, meta)]

    def _finalize_ready(self, data_now: float, wall_now: float | None = None) -> Out:
        out: Out = []
        for p in list(self.pending):
            stale = wall_now is not None and wall_now >= p.finalize_at + STALE_GRACE_S
            if data_now >= p.finalize_at or stale:
                self.pending.remove(p)
                out.append((TOPIC_EVENT_DONE, self._finalize(p.meta, partial=stale and data_now < p.finalize_at)))
        return out

    def _finalize(self, meta: dict, partial: bool) -> dict:
        t = meta["t"]
        raw, ts = self.raw_buf.window(t - self.cfg.pre_s, t + self.cfg.post_s)
        filt, _ = self.filt_buf.window(t - self.cfg.pre_s, t + self.cfg.post_s)
        expected = (self.cfg.pre_s + self.cfg.post_s) * self.fs
        partial = partial or ts.size < expected - 2          # e.g. event right after the headset connected
        meta = {**meta, "state": "saved", "partial": partial, "n_samples": int(ts.size),
                "window_s": round(ts.size / self.fs, 2) if ts.size else 0.0, "finalized": time.time()}
        meta["stats"] = event_window_stats(raw, filt, ts, self.fs, self.channels, t) if ts.size else {}
        if self.store and ts.size:
            meta["files"] = self.store.save_window(meta, raw, filt, ts, self.channels, self.fs)
        elif self.store:
            self.store.upsert(meta)
        log.info("event %s saved: %.1f s of EEG%s", meta["id"], meta["window_s"], " (partial)" if partial else "")
        return meta

    # ------------------------------------------------------------ calibration
    def start_calibration(self, now: float | None = None) -> Out:
        gestures = [d.name for d in self.detectors if d.calibratable]
        if not gestures:
            msg = {"state": "failed", "prompt": "Choose a gesture (clench or double blink) to calibrate", "t": now}
            return [(TOPIC_CALIBRATION, msg)]
        start = (now if now is not None else self.data_now) + 0.5
        self.calibration = CalibrationSession(self.cfg.calibration, gestures, start)
        log.info("calibration started (%s, %.0f s)", ",".join(gestures), self.calibration.total_s)
        return self._step_calibration(start)

    def cancel_calibration(self) -> Out:
        if not self.calibration:
            return []
        self.calibration = None
        return [(TOPIC_CALIBRATION, {"state": "cancelled", "prompt": "Calibration cancelled", "t": self.data_now})]

    def _step_calibration(self, now: float) -> Out:
        cal = self.calibration
        step, finished = cal.advance(now)
        if finished:
            return self._finish_calibration(now)
        out: Out = []
        if step is not None:
            out.append((TOPIC_CALIBRATION, cal.message(now)))
            if step.kind == "do" and self.cfg.calibration.auto_inject_sim:
                out.append((INJECT, {"gesture": step.gesture}))
        return out

    def _finish_calibration(self, now: float) -> Out:
        cal, self.calibration = self.calibration, None
        warm = 1.0                                        # extra signal before each segment for filters
        rs, re_ = cal.rest_window()
        rest, _ = self.raw_buf.window(rs - warm, re_)
        results, report = {}, {}
        for d in self.detectors:
            if not d.calibratable:
                continue
            segs = [self.raw_buf.window(s - warm, e)[0] for s, e in cal.gesture_windows(d.name)]
            r = d.calibrate(rest, segs)
            report[d.name] = {**r.to_dict(), "label": d.label}
            if r.quality != "failed":
                results[d.name] = r.to_dict()
        if results:
            self._save_profile(results)
        ok = all(v["quality"] != "failed" for v in report.values())
        msg = {"state": "done" if ok else "failed", "t": now, "results": report, "progress": 1.0,
               "prompt": "Calibration complete" if ok else "Calibration did not work - see details and retry"}
        self.last_calibration = msg
        log.info("calibration %s: %s", msg["state"], report)
        return [(TOPIC_CALIBRATION, msg)]

    # ------------------------------------------------------------ status
    def status(self, wall_now: float | None = None) -> dict:
        now = self.data_now or (wall_now or time.time())
        remaining = max(0.0, self.cfg.cooldown_s - (now - self.last_event_t))
        if self.calibration:
            state = "calibrating"
        elif self.gesture == "manual":
            state = "manual"
        elif remaining > 0:
            state = "cooldown"
        else:
            state = "armed"
        return {
            "state": state, "gesture": self.gesture, "cooldown_remaining_s": round(remaining, 1),
            "cooldown_s": self.cfg.cooldown_s, "detectors": [d.status() for d in self.detectors],
            "recording": [{"id": p.meta["id"], "ends_at": p.finalize_at} for p in self.pending],
            "suppressed": self.suppressed, "profile": self.profile_path.name,
            "calibrated": all(getattr(d, "calibrated_threshold", None) is not None
                              for d in self.detectors if d.calibratable) and bool(self.detectors),
            "t": now,
        }
