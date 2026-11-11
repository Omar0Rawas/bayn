"""Phase 6: the EEG-only ML triple-blink detector (the trigger is three blinks since 2026-11-10).

Simulator artefacts that fool the rule (nods, head turns), the labelled training protocols, the
candidate finder and features, labels from prompts, training + held-out evaluation, the live
detector with its rule fallback, and the event manager running a recording session."""

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from bayn.common.config import BlinkConfig, EventsConfig, TrainingConfig
from bayn.common.schemas import TOPIC_TRAINING
from bayn.eeg.sources.simulated import SimulatedEEG
from bayn.events import blink_ml, training
from bayn.events.blink_ml import (FEATURES, MLBlinkDetector, episode_features, find_episodes,
                                  label_episodes, segment_bounds)
from bayn.events.detectors import BlinkCountDetector, make_gesture_detectors
from bayn.events.manager import INJECT, TRAIN, EventManager
from bayn.events.training import Recording, TrainingSession, build_steps, simulate_session
from tests.helpers import CHANNELS, FS, T0, sim_chunks


def _stream(det, seed: int, plan: list[tuple[float, str]], seconds: float):
    """Run a detector on simulated EEG with gestures at planned times; returns (true onsets per gesture, detections)."""
    g = SimulatedEEG(CHANNELS, FS, seed=seed)
    k0, todo, onsets = g.k, list(plan), {}

    def on_time(t_rel):
        while todo and t_rel >= todo[0][0]:
            _, gesture = todo.pop(0)
            onsets.setdefault(gesture, []).append(T0 + (g.inject(gesture, delay_s=0) - k0) / FS)

    found = []
    for raw, filt in sim_chunks(g, seconds, on_time=on_time):
        found += [d.t for d in det.update(raw, filt)]
    return onsets, found


def _hits(found, truth, lo=-0.6, hi=1.5):
    return sum(any(lo <= f - t <= hi for f in found) for t in truth)


PLAN = [(8 + 6 * i, g) for i, g in enumerate(["triple_blink", "nod", "double_blink", "head_turn", "triple_blink", "nod",
                                              "look_side", "triple_blink", "double_blink", "nod", "talk",
                                              "triple_blink", "head_turn", "nod", "blink", "triple_blink", "nod",
                                              "double_blink", "triple_blink", "head_turn", "nod", "triple_blink",
                                              "look_side", "nod", "triple_blink"])]


# ---------------------------------------------------------------- simulator artefacts (unit)
@pytest.mark.unit
def test_simulator_artefacts_have_their_signatures():
    g = SimulatedEEG(CHANNELS, FS, seed=3, natural_blinks=False, natural_emg=False, background_uv=0.1, mains_hz=0)
    i7, i8, i9 = CHANNELS.index("AF7"), CHANNELS.index("AF8"), CHANNELS.index("TP9")
    blink = g._blink_wave(200)
    assert blink[i7].max() > 150 and blink[i9].max() > 25         # synchronised TP9/TP10 peak (as on the real headset)
    look = g._look_side()
    assert np.sign(look[i7, len(look[0]) // 2]) == -np.sign(look[i8, len(look[0]) // 2])   # AF7/AF8 opposite
    nod = g._nod()
    assert nod.shape[1] > 0.6 * FS and np.abs(nod[i9]).max() > 50   # wide, and large on TP9 too
    for gesture in ("head_turn", "nod", "look_side", "talk"):
        g.inject(gesture)
    assert np.isfinite(g.generate(int(8 * FS))).all()


@pytest.mark.unit
def test_rule_detector_false_triggers_on_nods():
    """Reproduces finding F1 in simulation: repeated nodding ("yes, yes, yes") looks like blinks to the rule."""
    onsets, found = _stream(BlinkCountDetector(FS, CHANNELS), 5, [(8 + 5 * i, "nod") for i in range(16)], 90)
    assert _hits(found, onsets["nod"]) >= 2


# ---------------------------------------------------------------- protocols + sessions (unit)
@pytest.mark.unit
def test_protocols_have_the_planned_prompts_and_length():
    std = TrainingSession("standard", T0, seed=1)
    assert std.counts() == {"triple_blink": 12, "double_blink": 8, "blink": 6, "head_turn": 8, "nod": 8,
                            "look_side": 8}
    assert 4.5 * 60 < std.total_s < 5.5 * 60
    noisy = TrainingSession("noisy", T0, seed=1)
    assert noisy.counts() == {"triple_blink": 16} and 4 * 60 < noisy.total_s < 5.5 * 60
    assert [s.label for s in build_steps("standard", T0, seed=1)] != [s.label for s in build_steps("standard", T0, seed=2)]
    with pytest.raises(ValueError):
        TrainingSession("marathon", T0)


@pytest.mark.unit
def test_session_advances_and_saves(tmp_path):
    s = TrainingSession("standard", T0 + 1, seed=1, scale=0.1)
    assert s.advance(T0) == (None, False)                       # before the start
    step, done = s.advance(T0 + 1.01)
    assert step.kind == "rest" and not done
    s.record(np.ones((4, 32), np.float32), T0 + np.arange(32) / FS)
    assert s.advance(s.t_end)[1] is True
    rec = Recording.load(s.save(tmp_path, CHANNELS, FS))
    assert rec.data.shape == (4, 32) and rec.channels == CHANNELS and len(rec.steps) == len(s.steps)


# ---------------------------------------------------------------- finder, features, labels (unit)
@pytest.fixture(scope="module")
def std_recording(tmp_path_factory) -> Recording:
    return Recording.load(simulate_session("standard", tmp_path_factory.mktemp("rec"), seed=21))


@pytest.mark.unit
def test_finder_groups_a_triple_blink_into_one_episode():
    g = SimulatedEEG(CHANNELS, FS, seed=4, natural_blinks=False, natural_emg=False)
    g.k = 0
    data = g.generate(int(6 * FS))
    g.inject("triple_blink", delay_s=0)
    data = np.hstack([data, g.generate(int(4 * FS))])
    eps = find_episodes(data, T0 + np.arange(data.shape[1]) / FS, FS, CHANNELS)
    assert len(eps) == 1
    same = [p for p in eps[0].peaks if np.sign(p[1]) == np.sign(eps[0].peaks[0][1])]
    assert len(same) == 3 and all(0.3 <= b[0] - a[0] <= 0.55 for a, b in zip(same, same[1:]))


@pytest.mark.unit
def test_features_are_finite_and_complete(std_recording):
    rec = std_recording
    eps = find_episodes(rec.data, rec.ts, rec.fs, rec.channels)
    X, kept = blink_ml.features_for(rec.data, rec.ts, rec.fs, rec.channels, eps)
    assert X.shape == (len(kept), len(FEATURES)) and len(kept) > 40 and np.isfinite(X).all()


@pytest.mark.unit
def test_labels_follow_the_prompts(std_recording):
    rec = std_recording
    eps = find_episodes(rec.data, rec.ts, rec.fs, rec.channels)
    labels = label_episodes(rec, eps)
    n_triple = sum(1 for s in rec.steps if s.label == "triple_blink")
    assert labels.count("triple_blink") >= n_triple - 1          # nearly every prompt yields its episode
    assert labels.count("blinks") >= 8                            # double + single blinks: the near misses
    assert labels.count("movement") >= 10
    for e, lab in zip(eps, labels):
        if lab == "triple_blink":
            assert any(s.label == "triple_blink" and s.start <= e.t_first < s.end + 1 for s in rec.steps)


# ---------------------------------------------------------------- training + live detector (integration)
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("bench")
    with mock.patch.object(training, "TRAINING_DIR", root / "training"), \
            mock.patch.object(blink_ml, "MODELS_DIR", root / "models"):
        folder = training.session_folder("default", "sim")
        # ~19 min: what a user records in 2 standard + 2 noisy sessions (2 sessions are not enough, see TESTING)
        for i, proto in enumerate(["standard", "noisy", "standard", "noisy"]):
            simulate_session(proto, folder, seed=40 + i, t0=T0 + i * 1000)
        report = blink_ml.train_from_sessions("default", "sim")
        yield report, blink_ml.model_path("default", "sim")


@pytest.mark.integration
def test_ml_beats_the_rule_on_held_out_sessions(trained):
    report, path = trained
    assert report["cv"] == "leave one session out"
    ml, rule = report["ml"], report["rule"]
    assert ml["recall"] >= 0.85
    assert ml["false_per_min"] <= rule["false_per_min"]
    assert path.exists() and path.with_suffix(".json").exists()
    assert {"triple_blink", "blinks", "movement"} <= set(report["tp_observation"])


@pytest.mark.integration
def test_live_ml_detector_on_a_fresh_stream(trained):
    _, path = trained
    ml = MLBlinkDetector(FS, CHANNELS)
    assert ml.load_model(path) and ml.status()["method"] == "ml"
    onsets, found = _stream(ml, 77, PLAN, PLAN[-1][0] + 8)
    hits = _hits(found, onsets["triple_blink"])
    false = len(found) - hits
    _, rule_found = _stream(BlinkCountDetector(FS, CHANNELS), 77, PLAN, PLAN[-1][0] + 8)
    rule_false = len(rule_found) - _hits(rule_found, onsets["triple_blink"])
    assert hits >= len(onsets["triple_blink"]) - 2
    assert false <= 2 and false <= rule_false
    # a double blink can genuinely become three blinks when a natural blink lands next to it: allow one
    assert sum(_hits(found, onsets[g]) for g in ("double_blink", "blink")) <= 1


@pytest.mark.unit
def test_ml_detector_falls_back_to_the_rule(tmp_path):
    ml = MLBlinkDetector(FS, CHANNELS)
    assert not ml.load_model(tmp_path / "missing.joblib") and ml.status()["method"] == "rule"
    onsets, found = _stream(ml, 6, [(8, "triple_blink"), (20, "triple_blink")], 30)
    assert _hits(found, onsets["triple_blink"]) == 2              # the rule still works without a model
    cfg = EventsConfig(triple_blink=BlinkConfig(method="rule"))
    assert type(make_gesture_detectors("triple_blink", cfg, FS, CHANNELS)[0]) is BlinkCountDetector
    assert type(make_gesture_detectors("triple_blink", EventsConfig(), FS, CHANNELS)[0]) is MLBlinkDetector


@pytest.mark.integration
def test_auto_uses_ml_only_if_it_beat_the_rule(trained, tmp_path):
    import joblib

    _, path = trained
    bundle = joblib.load(path)
    bundle["report"] = {**bundle["report"], "ml": {"recall": 0.6, "false_per_min": 2.0},
                        "rule": {"recall": 0.9, "false_per_min": 0.5}}
    worse = tmp_path / "worse.joblib"
    joblib.dump(bundle, worse)
    auto = MLBlinkDetector(FS, CHANNELS)
    assert auto.load_model(worse) and auto.status()["method"] == "rule"          # auto keeps the rule
    forced = MLBlinkDetector(FS, CHANNELS, BlinkConfig(method="ml"))
    assert forced.load_model(worse) and forced.status()["method"] == "ml"        # ml forces the classifier
    assert blink_ml.ml_not_worse({"ml": {"recall": 0.9, "false_per_min": 0.1}, "rule": {"recall": 0.93, "false_per_min": 1}})
    assert not blink_ml.ml_not_worse({})


@pytest.mark.integration
def test_incompatible_model_is_rejected(trained):
    _, path = trained
    other = MLBlinkDetector(FS, ["TP9", "AF7", "AF8", "TP10", "AUX"])
    assert not other.load_model(path)
    assert MLBlinkDetector(FS, ["C3", "C4"]).load_model(path) is False      # no forehead/ear sensors


@pytest.mark.unit
def test_episode_features_match_offline_and_live():
    """The live detector computes features from its ring buffer; offline from the whole recording."""
    g = SimulatedEEG(CHANNELS, FS, seed=8, natural_blinks=False)
    data = g.generate(int(5 * FS))
    g.inject("triple_blink", delay_s=0)
    data = np.hstack([data, g.generate(int(4 * FS))]).astype(np.float32)
    ts = T0 + np.arange(data.shape[1]) / FS
    ep = find_episodes(data, ts, FS, CHANNELS)[0]
    t0, t1 = segment_bounds(ep)
    i0, i1 = np.searchsorted(ts, [t0, t1])
    offline = episode_features(data[:, i0:i1], ts[i0], FS, CHANNELS, ep)
    buf = blink_ml.RingBuffer(4, int(10 * FS))
    buf.append(data, ts)
    seg, sts = buf.window(t0, t1)
    assert np.allclose(offline, episode_features(seg, sts[0], FS, CHANNELS, ep))


# ---------------------------------------------------------------- event manager runs a session (unit)
@pytest.mark.unit
def test_manager_records_a_session_and_asks_for_training(tmp_path):
    cfg = EventsConfig(gesture="triple_blink", training=TrainingConfig(scale=0.05))
    with mock.patch.object(training, "TRAINING_DIR", tmp_path / "training"):
        m = EventManager(cfg, FS, CHANNELS, store=None, source="sim", profiles_dir=tmp_path / "profiles")
        g = SimulatedEEG(CHANNELS, FS, seed=9)
        msgs, started = [], False
        for raw, filt in sim_chunks(g, 30):
            if not started and raw.t0 > T0 + 3:
                msgs += m.start_training("standard")
                started = True
                assert m.status()["state"] == "training"
            out = m.on_chunk(raw, filt)
            msgs += out
            if any(t == TRAIN for t, _ in out):
                break
        kinds = [t for t, _ in msgs]
        assert TRAIN in kinds and INJECT in kinds                    # sim performs the prompts itself
        saved = [p for t, p in msgs if t == TOPIC_TRAINING and p["state"] == "saved"][0]
        assert Path(saved["path"]).exists() and saved["sessions"] == 1
        assert not any(t == "events.event" for t in kinds)            # no events fire while recording
        assert m.start_training("standard")[0][1]["state"] == "running"
        assert m.cancel_training()[0][1]["state"] == "cancelled"


# ---------------------------------------------------------------- learned blink signal + synthetic data
@pytest.fixture(scope="module")
def tp_heavy_recording(tmp_path_factory) -> Recording:
    """Like the user's Muse: blinks are largest, and inverted, on TP9/TP10 (forehead reference)."""
    return Recording.load(simulate_session("standard", tmp_path_factory.mktemp("tp"), seed=51, tp_blink_coupling=-1.5))


@pytest.mark.unit
def test_learned_weights_find_the_persons_blink_pattern(tp_heavy_recording):
    w = blink_ml.learn_weights([tp_heavy_recording])
    ch = tp_heavy_recording.channels
    tp, af = w[[ch.index("TP9"), ch.index("TP10")]], w[[ch.index("AF7"), ch.index("AF8")]]
    assert np.abs(tp).min() > np.abs(af).max()                   # the ear sensors carry the blink
    assert np.sign(tp[0]) == np.sign(tp[1]) == -np.sign(af[0])   # ...inverted against the forehead
    assert abs(np.linalg.norm(w) - 1) < 1e-9


@pytest.mark.unit
def test_synthetic_data_is_built_from_the_persons_blinks(tp_heavy_recording):
    w = blink_ml.learn_weights([tp_heavy_recording])
    templates, gaps = blink_ml.blink_templates([tp_heavy_recording], w)
    assert len(templates) >= 30 and all(t.shape[0] == 4 for t in templates)
    assert gaps and 0.25 <= float(np.median(gaps)) <= 0.6        # their real triple-blink rhythm
    X, y = blink_ml.synthesize([tp_heavy_recording], w, n=40, seed=3)
    assert X.shape[1] == len(FEATURES) and np.isfinite(X).all()
    assert {"triple_blink", "blinks"} <= set(y)


@pytest.mark.integration
def test_report_compares_before_learned_and_synthetic(trained):
    report, _ = trained
    assert {"rule_old", "rule", "ml_real", "ml"} <= set(report)
    assert set(report["weights"]) == set(CHANNELS) and report["rule_threshold"]
    assert sum(report["synthetic"].values()) > 0


@pytest.mark.integration
def test_trained_model_moves_the_rule_onto_the_learned_signal(trained):
    _, path = trained
    ml = MLBlinkDetector(FS, CHANNELS)
    assert ml.load_model(path)
    assert ml.rule.weights_id.startswith("w:") and ml.rule.threshold == pytest.approx(
        __import__("joblib").load(path)["rule_threshold"])
    before = ml.rule.threshold
    ml.apply_profile({"threshold": 5.0})                          # old forehead calibration: other signal
    assert ml.rule.threshold == before
    ml.apply_profile({"threshold": 5.0, "space": ml.rule.weights_id})   # calibrated on this signal
    assert ml.rule.threshold == 5.0
    assert not ml.load_model(Path("missing.joblib")) and ml.rule.weights_id == "front"   # back to defaults


# ---------------------------------------------------------------- the "is ML necessary?" decision test
def _rep(ml_recall, ml_false, rule_recall, rule_false, protocols=("standard", "noisy", "boundaries"),
         conditions=("seated", "standing")):
    return {"ml": {"recall": ml_recall, "false_per_min": ml_false}, "rule": {"recall": rule_recall, "false_per_min": rule_false},
            "protocols": list(protocols), "conditions": list(conditions)}


@pytest.mark.unit
@pytest.mark.parametrize("ml_r, ml_f, rule_r, rule_f, necessary", [
    (0.95, 0.4, 0.84, 0.4, True),      # >= 10 points more caught, no more false triggers
    (0.95, 0.6, 0.84, 0.4, False),     # ...but more false triggers
    (0.86, 0.3, 0.86, 0.9, True),      # 0.6/min fewer false triggers, same recall
    (0.80, 0.3, 0.86, 0.9, False),     # ...but 6 points fewer caught
    (0.82, 0.81, 0.86, 0.41, False),   # the user's first two sessions: the rule is simpler and as good
    (0.90, 0.40, 0.86, 0.41, False),   # slightly better is not clearly better: the simpler rule wins
])
def test_pre_registered_decision_rule(ml_r, ml_f, rule_r, rule_f, necessary):
    assert blink_ml.ml_clearly_better(_rep(ml_r, ml_f, rule_r, rule_f)) is necessary


@pytest.mark.unit
def test_decision_is_final_only_with_all_protocols_and_standing():
    assert blink_ml.decision(_rep(0.8, 0.5, 0.8, 0.5))["final"]
    d = blink_ml.decision(_rep(0.8, 0.5, 0.8, 0.5, protocols=("standard", "noisy"), conditions=("seated",)))
    assert not d["final"] and d["missing_protocols"] == ["a boundaries session", "a session done standing"]


@pytest.mark.integration
def test_decision_document_states_question_criteria_and_verdict(trained):
    report, _ = trained
    md = blink_ml.render_decision(report)
    assert "Is a machine-learning model necessary" in md and "PRELIMINARY" in md          # simulated: no standing
    assert f"{blink_ml.RECALL_MARGIN * 100:.0f} points more" in md and str(blink_ml.FALSE_MARGIN) in md
    assert report["decision"]["answer"] in md and "| **Rule** on learned signal |" in md
    assert {"rule_old", "rule", "rule_plus", "ml_real", "ml", "by_condition"} <= set(report)


@pytest.mark.unit
def test_session_conditions_list_and_delete(tmp_path):
    with mock.patch.object(training, "TRAINING_DIR", tmp_path):
        folder = training.session_folder("default", "muse")
        s = TrainingSession("standard", T0, seed=1, scale=0.05, condition="standing")
        assert s.id.endswith("-standard-standing") and s.message(T0)["condition"] == "standing"
        s.record(np.ones((4, 32), np.float32), T0 + np.arange(32) / FS)
        rec = Recording.load(s.save(folder, CHANNELS, FS))
        assert rec.condition == "standing"
        listed = training.session_summaries("default", "muse")
        assert [x["id"] for x in listed] == [s.id] and listed[0]["triple_prompts"] == 12
        assert not training.delete_recording("default", "muse", "../../evil")      # only plain session ids
        assert training.delete_recording("default", "muse", s.id) and training.session_summaries("default", "muse") == []
    with pytest.raises(ValueError):
        TrainingSession("standard", T0, condition="upside_down")
