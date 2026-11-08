"""Event manager: cooldown, debounce, EEG window saving, calibration flow, profiles."""

import json

import numpy as np
import pytest

from bayn.common.config import CalibrationConfig, EventsConfig
from bayn.common.schemas import TOPIC_CALIBRATION, TOPIC_EVENT, TOPIC_EVENT_DONE
from bayn.eeg.sources.simulated import SimulatedEEG
from bayn.events.manager import INJECT, EventManager
from bayn.events.store import EventStore
from tests.helpers import CHANNELS, FS, T0, schedule_injections, sim_chunks

pytestmark = pytest.mark.unit


def _mgr(tmp_path, **kw) -> EventManager:
    cfg = EventsConfig(**{"gesture": "clench", "cooldown_s": 15, "pre_s": 10, "post_s": 30, **kw})
    return EventManager(cfg, FS, CHANNELS, EventStore(tmp_path), source="sim", profiles_dir=tmp_path / "profiles")


def _drive(mgr, seconds, seed=1, gesture=None, n=0, first_s=12.0, every_s=20.0, extra=None):
    g = SimulatedEEG(CHANNELS, FS, seed=seed, natural_blinks=False, natural_emg=False)
    onsets, on_time = schedule_injections(g, gesture, n, first_s, every_s)
    out = []

    def hook(t_rel):
        on_time(t_rel)
        if extra:
            extra(t_rel, g, out)

    for raw, filt in sim_chunks(g, seconds, on_time=hook):
        msgs = mgr.on_chunk(raw, filt)
        out += msgs
        for topic, payload in msgs:                       # play the service's role for injections
            if topic == INJECT:
                g.inject(payload["gesture"], delay_s=0.3)
    return onsets, out


def _topics(out, topic):
    return [p for t, p in out if t == topic]


def test_gesture_creates_event_and_saves_exact_window(tmp_path):
    mgr = _mgr(tmp_path)
    onsets, out = _drive(mgr, 50, gesture="clench", n=1)
    ev, done = _topics(out, TOPIC_EVENT), _topics(out, TOPIC_EVENT_DONE)
    assert len(ev) == 1 and len(done) == 1
    assert abs(ev[0]["t"] - onsets[0]) < 0.5 and ev[0]["detector"] == "clench"
    d = done[0]
    assert d["state"] == "saved" and not d["partial"]
    assert abs(d["n_samples"] - 40 * FS) <= 1               # -10 s / +30 s, +-1 sample
    z = np.load(tmp_path / d["files"]["eeg.npz"])
    assert z["raw"].shape == (4, d["n_samples"]) and z["filt"].shape == z["raw"].shape
    assert z["ts"][0] >= d["t"] - 10 - 1 / FS and z["ts"][-1] < d["t"] + 30
    meta = json.loads((tmp_path / d["files"]["meta.json"]).read_text())
    assert set(meta["stats"]) == {"before", "after"}
    assert meta["stats"]["after"]["bands"]["TP9"]["gamma"] >= 0
    csv = (tmp_path / d["files"]["eeg.csv"]).read_text().splitlines()
    assert csv[0] == "t_rel_s,TP9_uv,AF7_uv,AF8_uv,TP10_uv" and len(csv) == d["n_samples"] + 1
    assert mgr.store.get(d["id"])["state"] == "saved"


def test_cooldown_suppresses_second_gesture(tmp_path):
    mgr = _mgr(tmp_path, post_s=5)
    onsets, out = _drive(mgr, 40, gesture="clench", n=2, every_s=6.0)   # 2nd clench 6 s later < 15 s
    assert len(onsets) == 2
    assert len(_topics(out, TOPIC_EVENT)) == 1 and mgr.suppressed == 1


def test_manual_trigger_debounce_and_cooldown_interaction(tmp_path):
    mgr = _mgr(tmp_path)
    assert len(mgr.trigger_manual("button", t=T0 + 1)) == 1
    assert mgr.trigger_manual("hotkey", t=T0 + 1.5) == []                    # debounced
    assert len(mgr.trigger_manual("hotkey", t=T0 + 4)) == 1                  # manual ignores cooldown
    assert mgr.status(T0 + 4)["state"] in ("cooldown", "armed")


def test_manual_event_finalizes_by_wall_clock_without_eeg(tmp_path):
    mgr = _mgr(tmp_path, post_s=2)
    (topic, ev), = mgr.trigger_manual("button", t=T0)
    assert topic == TOPIC_EVENT and mgr.tick(T0 + 1) == []
    (topic, done), = mgr.tick(T0 + 2 + 6)
    assert topic == TOPIC_EVENT_DONE and done["partial"] and done["n_samples"] == 0


def test_event_right_after_connect_is_marked_partial(tmp_path):
    mgr = _mgr(tmp_path, post_s=3)
    _, out = _drive(mgr, 12, gesture="clench", n=1, first_s=2.0)    # only ~2 s of pre-event data
    done = _topics(out, TOPIC_EVENT_DONE)[0]
    assert done["partial"] and done["n_samples"] < 13 * FS


def test_manual_only_mode_ignores_gestures(tmp_path):
    mgr = _mgr(tmp_path, gesture="manual")
    _, out = _drive(mgr, 30, gesture="clench", n=1)
    assert _topics(out, TOPIC_EVENT) == [] and mgr.status()["state"] == "manual"


def test_set_gesture_at_runtime(tmp_path):
    mgr = _mgr(tmp_path, post_s=3)
    mgr.set_gesture("double_blink")
    onsets, out = _drive(mgr, 25, gesture="double_blink", n=1)
    assert [e["detector"] for e in _topics(out, TOPIC_EVENT)] == ["double_blink"]


# ---------------------------------------------------------------- calibration
def _calibrate(tmp_path, gesture, seconds=45):
    cal = CalibrationConfig(rest_s=8, reps=3, ready_s=1, do_s=3, release_s=2)
    mgr = _mgr(tmp_path, gesture=gesture, calibration=cal)
    started = {}

    def extra(t_rel, g, out):
        if t_rel >= 3 and not started:
            started["x"] = True
            out += mgr.start_calibration()

    _, out = _drive(mgr, seconds, extra=extra)
    return mgr, out


@pytest.mark.parametrize("gesture", ["clench", "double_blink"])
def test_guided_calibration_end_to_end(tmp_path, gesture):
    mgr, out = _calibrate(tmp_path, gesture)
    cal = _topics(out, TOPIC_CALIBRATION)
    steps = [m["step"] for m in cal if m["state"] == "running"]
    assert steps[0] == "rest" and steps.count("do") == 3
    final = cal[-1]
    assert final["state"] == "done", final
    r = final["results"][gesture]
    assert r["quality"] == "good" and r["rest_level"] < r["threshold"] < r["gesture_level"]
    assert _topics(out, TOPIC_EVENT) == []                 # no events fire while calibrating
    prof = json.loads(mgr.profile_path.read_text())
    assert prof["detectors"][gesture]["threshold"] == pytest.approx(r["threshold"])
    # a new manager (restart) loads the saved profile
    mgr2 = _mgr(tmp_path, gesture=gesture)
    assert mgr2.detectors[0].calibrated_threshold == pytest.approx(r["threshold"])
    assert mgr2.status()["calibrated"]


def test_calibration_cancel(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.data_now = T0
    assert mgr.start_calibration(T0)[0][1]["state"] == "running"
    (topic, m), = mgr.cancel_calibration()
    assert m["state"] == "cancelled" and mgr.calibration is None


def test_calibration_needs_a_gesture(tmp_path):
    mgr = _mgr(tmp_path, gesture="manual")
    (topic, m), = mgr.start_calibration(T0)
    assert m["state"] == "failed"


def test_store_rejects_path_traversal(tmp_path):
    store = EventStore(tmp_path)
    with pytest.raises(ValueError):
        store.dir_for("../../secrets")
