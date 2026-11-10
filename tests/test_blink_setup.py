"""Setup without blink training (2026-11-11): the built-in Muse blink pattern, and Calibrate learning the
person's own blink pattern from three triple blinks, kept across restarts.

The simulated "Muse person" is the simulator with its sensors relabelled, so that blinks land on the sensors
named TP9/TP10 and the sensors named AF7/AF8 carry the ear-side jaw EMG and alpha (on the user's real Muse the
blinks are largest on TP9/TP10 and the forehead is the noisy side). The plain forehead signal then misses
blinks, as it did on the real headset (34 % caught)."""

from unittest import mock

import numpy as np
import pytest

from bayn.common.config import CalibrationConfig, EventsConfig
from bayn.common.schemas import TOPIC_CALIBRATION, TOPIC_EVENT
from bayn.eeg.sources.simulated import SimulatedEEG
from bayn.events import blink_ml
from bayn.events.detectors import BlinkCountDetector
from bayn.events.detectors.blinks import BUILTIN_PATH, builtin_pattern, learn_pattern
from bayn.events.manager import INJECT, EventManager
from tests.helpers import CHANNELS, FS, T0, sim_chunks

LABELS = ["AF7", "TP9", "TP10", "AF8"]           # the simulator's TP9, AF7, AF8, TP10 rows, renamed


def _mgr(tmp_path, source="sim", **kw):
    cfg = EventsConfig(**{"gesture": "triple_blink", "cooldown_s": 15, "pre_s": 10, "post_s": 30, **kw})
    with mock.patch.object(blink_ml, "MODELS_DIR", tmp_path / "models"):      # no trained model
        return EventManager(cfg, FS, LABELS, store=None, source=source, profiles_dir=tmp_path / "profiles")


def _run(mgr, seconds, seed, plan=(), calibrate_at=None):
    """Drive a manager with the simulated Muse person; plan = [(t_rel, gesture)]."""
    g = SimulatedEEG(CHANNELS, FS, seed=seed)
    todo, out, onsets, k0 = list(plan), [], [], g.k
    state = {"cal": calibrate_at}

    def hook(t_rel):
        if state["cal"] is not None and t_rel >= state["cal"]:
            state["cal"] = None
            out.extend(mgr.start_calibration())
        while todo and t_rel >= todo[0][0]:
            _, gesture = todo.pop(0)
            onsets.append(T0 + (g.inject(gesture, delay_s=0) - k0) / FS)

    for raw, filt in sim_chunks(g, seconds, on_time=hook):
        msgs = mgr.on_chunk(raw, filt)
        out += msgs
        for topic, payload in msgs:
            if topic == INJECT:
                g.inject(payload["gesture"], delay_s=0.3)
    return onsets, out


def _caught(out, onsets):
    ev = [p["t"] for t, p in out if t == TOPIC_EVENT]
    return sum(any(-0.6 <= e - o <= 1.5 for e in ev) for o in onsets), len(ev)


TRIPLES = [(5 + 17 * i, "triple_blink") for i in range(6)]          # 17 s apart: beyond the 15 s cooldown


# ---------------------------------------------------------------- built-in pattern (unit)
@pytest.mark.unit
def test_builtin_pattern_is_shipped_and_follows_the_channel_order():
    assert BUILTIN_PATH.exists()
    w, thr = builtin_pattern(CHANNELS)
    assert thr > 0 and abs(np.linalg.norm(w) - 1) < 0.01
    ears = np.abs(w[[CHANNELS.index("TP9"), CHANNELS.index("TP10")]])
    assert ears.min() > np.abs(w[[CHANNELS.index("AF7"), CHANNELS.index("AF8")]]).max()   # Muse: blinks on the ears
    flipped = ["TP10", "AF8", "AF7", "TP9"]
    assert np.allclose(builtin_pattern(flipped)[0], w[::-1])
    assert builtin_pattern(["C3", "C4"]) is None                                         # another headset


@pytest.mark.unit
def test_real_headset_starts_on_the_builtin_pattern_simulator_on_the_forehead(tmp_path):
    real = _mgr(tmp_path / "a", source="muse").detectors[0]
    assert real.status()["pattern"] == "built-in" and real.rule.threshold == pytest.approx(builtin_pattern(LABELS)[1])
    assert not real.status()["calibrated"]                         # shipped numbers are not a calibration
    assert _mgr(tmp_path / "b", source="sim").detectors[0].status()["pattern"] == "front"
    rule_only = _mgr(tmp_path / "c", source="muse",
                     triple_blink={"method": "rule"}).detectors[0]
    assert isinstance(rule_only, BlinkCountDetector) and rule_only.pattern == "built-in"


@pytest.mark.unit
def test_learn_pattern_needs_three_blinks():
    rng = np.random.default_rng(0)
    seg = rng.standard_normal((4, 1024))
    assert learn_pattern([seg, seg], FS, (0.3, 10.0)) is None
    w = learn_pattern([seg] * 3, FS, (0.3, 10.0))
    assert w.shape == (4,) and abs(np.linalg.norm(w) - 1) < 1e-6


# ---------------------------------------------------------------- Calibrate learns the pattern (integration)
@pytest.mark.integration
def test_calibration_learns_the_persons_blink_pattern_and_keeps_it(tmp_path):
    cal = CalibrationConfig(rest_s=13, reps=3, ready_s=2, do_s=3, release_s=3)
    mgr = _mgr(tmp_path, source="sim", calibration=cal)
    assert mgr.detectors[0].status()["pattern"] == "front"
    _, out = _run(mgr, 45, seed=21, calibrate_at=2)
    final = [p for t, p in out if t == TOPIC_CALIBRATION][-1]
    assert final["state"] == "done", final
    r = final["results"]["triple_blink"]
    assert r["quality"] == "good" and set(r["weights"]) == set(LABELS)
    w = r["weights"]
    assert min(abs(w["TP9"]), abs(w["TP10"])) > max(abs(w["AF7"]), abs(w["AF8"]))   # found the ear sensors
    # a restart keeps the learned pattern and its threshold
    again = _mgr(tmp_path, source="sim").detectors[0]
    assert again.status()["pattern"] == "calibrated" and again.status()["calibrated"]
    assert again.rule.threshold == pytest.approx(r["threshold"])


@pytest.mark.integration
def test_calibrated_pattern_catches_blinks_the_forehead_signal_misses(tmp_path):
    cal = CalibrationConfig(rest_s=13, reps=3, ready_s=2, do_s=3, release_s=3)
    _run(_mgr(tmp_path / "cal", calibration=cal), 45, seed=22, calibrate_at=2)
    calibrated = _mgr(tmp_path / "cal")
    forehead = _mgr(tmp_path / "plain")
    seconds = TRIPLES[-1][0] + 8
    hit_c, n_c = _caught(*reversed(_run(calibrated, seconds, seed=23, plan=TRIPLES)))
    hit_f, _ = _caught(*reversed(_run(forehead, seconds, seed=23, plan=TRIPLES)))
    assert hit_c >= 5 and n_c - hit_c == 0
    assert hit_c > hit_f


@pytest.mark.integration
def test_trained_model_keeps_its_pattern_through_calibration(tmp_path):
    """A trained model is 30 min of the person's data: Calibrate then only sets the threshold on its signal."""
    det = blink_ml.MLBlinkDetector(FS, LABELS)
    w = np.array([-0.7, 0.0, -0.1, -0.7]); w /= np.linalg.norm(w)
    det.rule.set_weights(w, "w:test", 300.0)                       # what load_model does with a bundle
    g = SimulatedEEG(CHANNELS, FS, seed=24, natural_blinks=False)
    rest = g.generate(int(14 * FS))
    gestures = []
    for _ in range(3):
        g.inject("triple_blink", delay_s=1.2)
        gestures.append(g.generate(int(4 * FS)))
    r = det.calibrate(rest, gestures)
    assert r.weights is None and det.rule.pattern == "trained" and det.rule.weights_id == "w:test"
