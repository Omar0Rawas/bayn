"""The seizure socket with its MOCK model (trained on CHB-MIT; not validated on Muse data, not for clinical use).

Unit: EDF reader, Muse-like channel mapping, annotation parser, streaming features (offline == live), labels,
alarm logic, the shipped model file. Integration: the detector on simulated EEG fires on the replayed CHB-MIT
seizure clip and not on gestures / artefacts; the event manager turns its alarm into a seizure-kind event."""

import json

import numpy as np
import pytest

from bayn.common.config import EventsConfig, SeizureConfig
from bayn.common.schemas import TOPIC_EVENT
from bayn.eeg.sources.simulated import SimulatedEEG
from bayn.events.detectors import SeizureModelDetector, make_gesture_detectors
from bayn.events.detectors.seizure_model import MODEL_PATH, load_model
from bayn.events.manager import EventManager
from bayn.seizure import chbmit, features, train
from tests.helpers import CHANNELS, FS, T0, sim_chunks

ASSETS_READY = MODEL_PATH.exists() and train.CLIP_PATH.exists()
needs_assets = pytest.mark.skipif(not ASSETS_READY, reason="python -m bayn.seizure.train --clip")


def _write_edf(path, labels, fs, data_uv, record_s=1):
    """Tiny EDF writer for the reader test (16-bit, physical range +-3276.7 uV)."""
    ns, spr = len(labels), int(fs * record_s)
    n_rec = data_uv.shape[1] // spr
    f = lambda s, w: s.ljust(w)[:w].encode("latin-1")
    hdr = f("0", 8) + f("test", 80) + f("test", 80) + f("01.01.01", 8) + f("00.00.00", 8) + f(str(256 * (ns + 1)), 8) \
        + f("", 44) + f(str(n_rec), 8) + f(str(record_s), 8) + f(str(ns), 4)
    for w, vals in ((16, labels), (80, [""] * ns), (8, ["uV"] * ns), (8, ["-3276.7"] * ns), (8, ["3276.7"] * ns),
                    (8, ["-32767"] * ns), (8, ["32767"] * ns), (80, [""] * ns), (8, [str(spr)] * ns), (32, [""] * ns)):
        hdr += b"".join(f(v, w) for v in vals)
    digital = np.clip(np.round(data_uv * 10), -32767, 32767).astype("<i2")
    body = b"".join(digital[:, r * spr:(r + 1) * spr].tobytes() for r in range(n_rec))
    path.write_bytes(hdr + body)


# ---------------------------------------------------------------- data handling (unit)
@pytest.mark.unit
def test_edf_reader_and_muse_like_mapping(tmp_path):
    labels = ["FP1-F7", "F7-T7", "T7-P7", "FP2-F8", "F8-T8"]
    data = np.vstack([np.full(512, v) for v in (10.0, 20.0, 5.0, -8.0, 4.0)])
    _write_edf(tmp_path / "x.edf", labels, 256, data)
    edf = chbmit.read_edf(tmp_path / "x.edf")
    assert edf.fs == 256 and edf.labels == labels and np.allclose(edf.data, data, atol=0.1)
    m = chbmit.muse_like(edf)                         # TP9, AF7, AF8, TP10
    assert np.allclose(m[:, 0], [-(10 + 20), -10 / 2, 8 / 2, -(-8 + 4)], atol=0.1)


@pytest.mark.unit
def test_summary_parser_handles_both_formats():
    text = """File Name: chb01_03.edf
Number of Seizures in File: 1
Seizure Start Time: 2996 seconds
Seizure End Time: 3036 seconds

File Name: chb06_01.edf
Number of Seizures in File: 2
Seizure 1 Start Time: 1724 seconds
Seizure 1 End Time: 1738 seconds
Seizure 2 Start Time: 7461 seconds
Seizure 2 End Time: 7476 seconds

File Name: chb06_02.edf
Number of Seizures in File: 0
"""
    files = chbmit.parse_summary(text)
    assert [f.seizures for f in files] == [[(2996, 3036)], [(1724, 1738), (7461, 7476)], []]


@pytest.mark.unit
def test_streaming_features_equal_offline_and_need_a_baseline():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, int(60 * FS))) * 10
    X, ends = features.recording_features(data, FS)
    assert X.shape == (57, len(features.FEATURES)) and ends[0] == features.WINDOW_S
    # live: odd chunk sizes, same features
    x = features.preprocess(data, FS)
    fs_ = features.FeatureStream(FS)
    live = []
    for i in range(0, x.shape[1], 37):
        live += fs_.push(x[:, i:i + 37])
    assert np.allclose(np.array(live), X)
    assert np.all(X[:20, 0] == 0)                     # no baseline yet in the first 30 s: "no change"


@pytest.mark.unit
def test_labels_exclude_the_transition_around_a_seizure():
    ends = np.arange(4, 400, 1.0)
    y = train.labels_for(ends, [(200, 240)])
    centre = ends - 2
    assert (y[(centre >= 200) & (centre <= 240)] == 1).all()
    assert (y[(centre >= 140) & (centre < 200)] == -1).all() and (y[(centre > 240) & (centre <= 360)] == -1).all()
    assert (y[centre < 140] == 0).all()


@pytest.mark.unit
def test_alarm_needs_consecutive_windows_and_respects_the_refractory_period():
    t = np.arange(0, 300.0)
    p = np.zeros_like(t)
    p[50:53] = 0.95                                   # 3 windows: too short
    p[100:110] = 0.95                                 # fires once
    p[150:160] = 0.95                                 # within 120 s: suppressed
    p[250:260] = 0.95                                 # fires again
    assert train.alarms(p, t, 0.8) == [104.0, 254.0]


@needs_assets
@pytest.mark.unit
def test_shipped_model_matches_the_features_and_says_what_it_is():
    m = load_model()
    assert m["features"] == list(features.FEATURES) and len(m["coef"]) == len(features.FEATURES)
    assert "Not validated on Muse data" in m["disclaimer"] and "CHB-MIT" in m["trained_on"]
    assert m["channels"] == ["TP9", "AF7", "AF8", "TP10"] and m["fs"] == 256.0


# ---------------------------------------------------------------- the detector in the socket (integration)
def _run(det, seed, plan, seconds):
    g = SimulatedEEG(CHANNELS, FS, seed=seed)
    todo, onsets = list(plan), {}
    k0 = g.k

    def on_time(t_rel):
        while todo and t_rel >= todo[0][0]:
            _, gesture = todo.pop(0)
            onsets.setdefault(gesture, []).append(T0 + (g.inject(gesture, delay_s=0) - k0) / FS)

    found = []
    for raw, filt in sim_chunks(g, seconds, on_time=on_time):
        found += det.update(raw, filt)
    return onsets, found


@needs_assets
@pytest.mark.integration
def test_detector_fires_on_the_real_seizure_clip():
    det = SeizureModelDetector(FS, CHANNELS)
    assert det.usable
    onsets, found = _run(det, 3, [(60, "seizure")], 150)
    clip_onset = onsets["seizure"][0] + 15.0          # the clip starts 15 s before the seizure
    assert len(found) == 1 and found[0].kind == "seizure"
    assert -10 <= found[0].t - clip_onset <= 45
    assert "not validated on Muse" in found[0].info["disclaimer"]


@needs_assets
@pytest.mark.integration
def test_detector_ignores_gestures_and_artefacts():
    det = SeizureModelDetector(FS, CHANNELS)
    plan = [(50 + 12 * i, g) for i, g in enumerate(["clench", "triple_blink", "talk", "nod", "head_turn", "look_side",
                                                    "clench", "eyebrows", "squeeze", "talk", "triple_blink", "nod"])]
    _, found = _run(det, 4, plan, 50 + 12 * len(plan) + 20)
    assert found == []


@pytest.mark.unit
def test_seizure_socket_is_off_by_default_and_plugs_in_beside_any_gesture():
    assert not any(d.kind == "seizure" for d in make_gesture_detectors("clench", EventsConfig(), FS, CHANNELS))
    cfg = EventsConfig(seizure=SeizureConfig(enabled=True))
    for gesture in ("clench", "triple_blink", "manual"):
        assert [d.name for d in make_gesture_detectors(gesture, cfg, FS, CHANNELS)][-1] == "seizure_model"
    assert not SeizureModelDetector(FS, ["C3", "C4"]).usable          # needs the Muse channels


@needs_assets
@pytest.mark.integration
def test_manager_turns_the_alarm_into_a_seizure_event(tmp_path):
    cfg = EventsConfig(gesture="manual", seizure=SeizureConfig(enabled=True))
    m = EventManager(cfg, FS, CHANNELS, store=None, source="sim", profiles_dir=tmp_path)
    g = SimulatedEEG(CHANNELS, FS, seed=5)
    injected = {}

    def on_time(t_rel):
        if t_rel >= 60 and not injected:
            injected["t"] = g.inject("seizure", delay_s=0)

    events = []
    for raw, filt in sim_chunks(g, 140, on_time=on_time):
        events += [p for t, p in m.on_chunk(raw, filt) if t == TOPIC_EVENT]
    assert len(events) == 1
    e = events[0]
    assert e["kind"] == "seizure" and e["detector"] == "seizure_model" and "mock" in e["label"]
    assert e["source"] == "automatic"
    json.dumps(e)                                      # serialisable for the bus / store


@needs_assets
@pytest.mark.integration
def test_seizure_alarm_is_not_swallowed_by_a_gesture_cooldown(tmp_path):
    """Showcase run (D32): the replayed clip first set off the triple-blink rule; the seizure alarm 4 s later
    fell in that event's 15 s cooldown and was dropped."""
    cfg = EventsConfig(gesture="manual", seizure=SeizureConfig(enabled=True))
    m = EventManager(cfg, FS, CHANNELS, store=None, source="sim", profiles_dir=tmp_path)
    g = SimulatedEEG(CHANNELS, FS, seed=5)
    injected = {}

    def on_time(t_rel):
        if t_rel >= 60 and not injected:
            injected["t"] = g.inject("seizure", delay_s=0)

    kinds = []
    for raw, filt in sim_chunks(g, 140, on_time=on_time):
        if injected:
            m.last_event_t = raw.t0 - 2.0                 # a gesture event fired 2 s ago, all the time
        kinds += [p["kind"] for t, p in m.on_chunk(raw, filt) if t == TOPIC_EVENT]
    assert kinds == ["seizure"]


@pytest.mark.unit
def test_changing_the_gesture_keeps_the_seizure_monitor_running(tmp_path):
    cfg = EventsConfig(gesture="clench", seizure=SeizureConfig(enabled=True))
    m = EventManager(cfg, FS, CHANNELS, store=None, source="sim", profiles_dir=tmp_path)
    before = [d for d in m.detectors if d.kind == "seizure"][0]
    for g in ("triple_blink", "manual", "both"):
        m.set_gesture(g)
        assert [d for d in m.detectors if d.kind == "seizure"] == [before]
    m.set_seizure(False)
    assert not any(d.kind == "seizure" for d in m.detectors)
