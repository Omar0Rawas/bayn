"""`bayn showcase`: the scripted, hardware-free tour (scene timeline, stitched audio, trigger timing)."""

import json

import pytest

from bayn.common.schemas import AUDIO_SR
from bayn.showcase import SCRIPT, TRIGGERS, Driver, build, caption, plan
from bayn.transcription.audio_sources import read_wav

SCRIPT_DICT = {"lead_s": 4, "scenes": [
    {"title": "A", "caption": "a", "audio": "a.wav", "trigger": "clench", "trigger_at_s": 9, "after_s": 14},
    {"title": "B", "caption": "b", "audio": "b.wav", "trigger": "seizure", "trigger_at_s": -12, "after_s": 30},
]}


@pytest.mark.unit
def test_plan_places_scenes_and_triggers():
    tl = plan(SCRIPT_DICT, [30.0, 32.0])
    a, b = tl.scenes
    assert (a.start_s, a.audio_s, a.audio_end_s, a.trigger_s) == (4, 4, 34, 13)
    # the seizure clip needs EEG before the speech: the caption and the trigger come 12 s earlier
    assert (b.start_s, b.trigger_s, b.audio_s, b.audio_end_s) == (48, 48, 60, 92)
    assert tl.length_s == 122
    with pytest.raises(ValueError):
        plan({"scenes": [{**SCRIPT_DICT["scenes"][0], "trigger": "wink"}]}, [10.0])


@pytest.mark.unit
def test_event_windows_do_not_overlap_in_the_shipped_tour():
    """Each event keeps -10/+30 s of audio: it must hold only its own scene's speech."""
    script = json.loads(SCRIPT.read_text(encoding="utf-8"))
    assert {s["trigger"] for s in script["scenes"]} == set(TRIGGERS)
    durs = [len(read_wav(SCRIPT.parent / s["audio"])) / AUDIO_SR for s in script["scenes"]]
    tl = plan(script, durs)
    for prev, nxt in zip(tl.scenes, tl.scenes[1:]):
        if prev.trigger != "seizure":
            assert prev.trigger_s + 30 <= nxt.audio_s
        assert prev.audio_end_s < nxt.start_s


@pytest.mark.unit
def test_build_stitches_one_wav_with_the_speech_where_planned(tmp_path):
    out, tl = build(SCRIPT, tmp_path / "s.wav")
    x = read_wav(out)
    assert abs(len(x) / AUDIO_SR - tl.length_s) < 0.01
    first = tl.scenes[0]
    assert abs(x[:int(first.audio_s * AUDIO_SR) - 10]).max() == 0          # silence before the speech
    assert abs(x[int(first.audio_s * AUDIO_SR):int(first.audio_end_s * AUDIO_SR)]).max() > 0.05
    assert json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))["length_s"] == tl.length_s


@pytest.mark.unit
def test_driver_fires_each_trigger_once_per_loop_and_follows_the_audio():
    tl = plan(SCRIPT_DICT, [30.0, 32.0])
    d = Driver(tl)
    fired, captions = [], []
    t0 = 1000.0
    for k in range(int(2.5 * tl.length_s * 10)):              # 100 ms chunks, two and a half loops
        pos = d.position(t0 + k / 10, k, t0 + (k + 1) / 10)
        due, sc = d.step(pos)
        fired += [(round(pos, 1), s.trigger) for s in due]
        captions.append(sc.index if sc else 0)
    assert [f[1] for f in fired] == ["clench", "seizure"] * 3
    assert fired[0][0] == pytest.approx(13, abs=0.11) and fired[2][0] == pytest.approx(13 + 122, abs=0.11)
    assert captions[0] == 0 and 1 in captions and 2 in captions             # intro, then the scenes
    # the audio process restarts (seq 0): position starts again from the top, nothing fires twice at once
    pos = d.position(5000.0, 0, 5000.1)
    assert pos == pytest.approx(0.1) and d.step(pos)[0] == []


@pytest.mark.unit
def test_driver_never_fires_a_trigger_it_jumped_past():
    d = Driver(plan(SCRIPT_DICT, [30.0, 32.0]))
    d.position(0.0, 0, 0.1)
    due, _ = d.step(40.0)                                    # e.g. joined late: the clench (13 s) is long gone
    assert due == []


@pytest.mark.unit
def test_caption_messages():
    tl = plan(SCRIPT_DICT, [30.0, 32.0])
    assert caption(None, 2, 1.0)["index"] == 0 and caption(None, 2, 1.0, pos=3.256)["pos"] == 3.26
    c = caption(tl.scenes[1], 2, 1.0)
    assert c["index"] == 2 and c["trigger"] == "seizure" and c["title"] == "B"
