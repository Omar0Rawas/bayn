"""Detectors: the shared interface contract, detection performance on simulated EEG
with known gesture times, false-trigger rate with realistic distractors, calibration."""

import pytest

from bayn.common.config import BlinkConfig, ClenchConfig, EventsConfig
from bayn.eeg.sources.simulated import SimulatedEEG
from bayn.events.detectors import (BlinkCountDetector, ClenchDetector, Detection, Detector, ManualDetector,
                                   SeizureDetectorStub, make_gesture_detectors)
from tests.helpers import CHANNELS, FS, match, schedule_injections, sim_chunks

pytestmark = pytest.mark.unit

ALL_DETECTORS = [
    lambda: ClenchDetector(FS, CHANNELS),
    lambda: BlinkCountDetector(FS, CHANNELS),
    lambda: ManualDetector(FS, CHANNELS),
    lambda: SeizureDetectorStub(FS, CHANNELS),
]
CONTRACT = ALL_DETECTORS + [lambda: __import__("bayn.events.blink_ml", fromlist=["x"]).MLBlinkDetector(FS, CHANNELS)]


def _run(det: Detector, seconds: float, seed: int, gesture: str | None = None, n: int = 0,
         every_s: float = 20.0, distractors: bool = True, mains: float = 50.0):
    g = SimulatedEEG(CHANNELS, FS, seed=seed, mains_hz=mains, natural_blinks=distractors, natural_emg=distractors)
    onsets, on_time = schedule_injections(g, gesture, n, first_s=8.0, every_s=every_s)
    found = []
    for raw, filt in sim_chunks(g, seconds, on_time=on_time):
        found += [d.t for d in det.update(raw, filt)]
    return onsets, found


# ---------------------------------------------------------------- interface contract
@pytest.mark.parametrize("factory", CONTRACT, ids=["clench", "triple_blink", "manual", "seizure_stub", "ml_blink"])
def test_detector_contract(factory):
    det = factory()
    assert isinstance(det, Detector) and det.name and det.label
    g = SimulatedEEG(CHANNELS, FS, seed=1)
    for raw, filt in sim_chunks(g, 3.0):
        out = det.update(raw, filt)
        assert isinstance(out, list) and all(isinstance(d, Detection) for d in out)
    st = det.status()
    assert {"name", "level", "threshold", "unit"} <= set(st)
    det.reset()                                           # must not raise


def test_seizure_stub_never_fires():
    _, found = _run(SeizureDetectorStub(FS, CHANNELS), 60, seed=2, gesture="clench", n=2)
    assert found == []


def test_factory_follows_gesture_setting():
    cfg = EventsConfig()
    assert [d.name for d in make_gesture_detectors("both", cfg, FS, CHANNELS)] == ["clench", "triple_blink"]
    assert make_gesture_detectors("manual", cfg, FS, CHANNELS) == []
    with pytest.raises(ValueError):
        make_gesture_detectors("wink", cfg, FS, CHANNELS)


# ---------------------------------------------------------------- clench
def test_clench_detection_rate_and_onset_accuracy():
    hits = total = fps = 0
    errs = []
    for seed in range(3):
        onsets, found = _run(ClenchDetector(FS, CHANNELS), 8 + 10 * 20, seed, "clench", 10, mains=50 + 10 * (seed % 2))
        h, fp, e = match(found, onsets)
        hits, total, fps, errs = hits + h, total + len(onsets), fps + fp, errs + e
    assert hits / total >= 0.95, f"{hits}/{total}"
    assert fps == 0
    assert max(abs(x) for x in errs) <= 0.5                # onset within 0.5 s


def test_clench_needs_hold_time():
    """A burst shorter than hold_s (talking, swallowing) must not fire."""
    det = ClenchDetector(FS, CHANNELS, ClenchConfig(hold_s=2.5))   # sim clenches last 1.5-2.0 s
    _, found = _run(det, 80, seed=3, gesture="clench", n=3)
    assert found == []


def test_clench_fires_once_per_clench():
    onsets, found = _run(ClenchDetector(FS, CHANNELS), 70, seed=4, gesture="clench", n=3)
    assert len(found) == len(onsets) == 3


# ---------------------------------------------------------------- triple blink (the trigger since 2026-11-10)
def test_triple_blink_detection_rate():
    hits = total = fps = 0
    for seed in range(3):
        onsets, found = _run(BlinkCountDetector(FS, CHANNELS), 8 + 10 * 20, seed, "triple_blink", 10)
        h, fp, _ = match(found, onsets)
        hits, total, fps = hits + h, total + len(onsets), fps + fp
    assert hits / total >= 0.9, f"{hits}/{total}"          # sim target; real-headset tuning in Phase 6
    assert fps == 0


@pytest.mark.parametrize("gesture", ["blink", "double_blink"])
def test_single_and_double_blinks_do_not_fire(gesture):
    _, found = _run(BlinkCountDetector(FS, CHANNELS), 140, seed=5, gesture=gesture, n=6)
    assert found == []


def test_blink_count_is_configurable():
    onsets, found = _run(BlinkCountDetector(FS, CHANNELS, BlinkConfig(count=2)), 60, seed=4, gesture="double_blink", n=2)
    assert match(found, onsets)[0] == 2 and BlinkCountDetector(FS, CHANNELS, BlinkConfig(count=2)).label == "Double blink"


def test_rebound_is_not_a_third_blink():
    """Regression (D9): the band-pass rebound after a blink was counted as another blink."""
    det = BlinkCountDetector(FS, CHANNELS)
    onsets, found = _run(det, 30, seed=6, gesture="triple_blink", n=1, distractors=False)
    assert match(found, onsets)[0] == 1


# ---------------------------------------------------------------- false triggers
@pytest.mark.parametrize("factory", ALL_DETECTORS[:2], ids=["clench", "triple_blink"])
def test_no_false_triggers_in_10_minutes_of_distractors(factory):
    """Natural blinks + short muscle bursts, no deliberate gestures: <= 1 trigger per 10 min."""
    _, found = _run(factory(), 600, seed=7, gesture=None)
    assert len(found) <= 1


def test_gestures_dont_cross_trigger():
    _, f1 = _run(BlinkCountDetector(FS, CHANNELS), 100, seed=8, gesture="clench", n=4)
    _, f2 = _run(ClenchDetector(FS, CHANNELS), 100, seed=8, gesture="triple_blink", n=4)
    assert f1 == [] and f2 == []


# ---------------------------------------------------------------- calibration
def _segments(gesture: str, seed: int = 9):
    g = SimulatedEEG(CHANNELS, FS, seed=seed)
    rest = g.generate(int(FS * 14))
    segs = []
    for _ in range(3):
        g.generate(int(FS * 2))
        start = g.k
        g.inject(gesture, delay_s=1.0)                     # 1 s warm-up pad, then the gesture
        segs.append(g.generate(int(FS * 4)))
        assert start < g.k
    return rest, segs


@pytest.mark.parametrize("cls,gesture", [(ClenchDetector, "clench"), (BlinkCountDetector, "triple_blink")])
def test_calibration_separates_rest_and_gesture(cls, gesture):
    det = cls(FS, CHANNELS)
    rest, segs = _segments(gesture)
    r = det.calibrate(rest, segs)
    assert r.quality == "good", r
    assert r.rest_level < r.threshold < r.gesture_level
    assert det.calibrated_threshold == pytest.approx(r.threshold)


def test_calibration_fails_without_gesture():
    det = ClenchDetector(FS, CHANNELS)
    rest, _ = _segments("clench")
    g = SimulatedEEG(CHANNELS, FS, seed=10)
    r = det.calibrate(rest, [g.generate(int(FS * 4)) for _ in range(3)])   # user did nothing
    assert r.quality in ("failed", "weak")


def test_profile_threshold_applied():
    det = BlinkCountDetector(FS, CHANNELS, BlinkConfig())
    det.apply_profile({"threshold": 123.0})
    assert det.threshold == 123.0 and det.status()["calibrated"]
