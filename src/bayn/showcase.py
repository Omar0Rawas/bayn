"""`bayn showcase`: a scripted tour for anyone without a headset or a microphone (e.g. judges).

Simulated EEG + recorded speech (samples/showcase/: a team member reading the fictional corpus scripts through the
wireless demo mic). Each scene plays one recording and fires one trigger at a fixed point in it: a jaw clench
(Arabic), the button (English), a triple blink (Arabic with English terms), and the mock seizure model on a
replayed CHB-MIT clip (an English lesson). Everything else is the real pipeline: detectors, saved EEG windows,
live Whisper transcripts and local LLM summaries.

    build()  -> one WAV of all scenes (data/showcase/showcase.wav) + the timeline     (run.py, before start-up)
    main()   -> the driver process: follows the audio's position on the bus, fires each trigger on time and
                publishes the scene caption (TOPIC_SHOWCASE) that the dashboard shows. Loops with the audio.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from bayn.common.bus import CommandSender, Publisher, Subscriber
from bayn.common.config import DATA_DIR, ROOT, AppConfig, load_config
from bayn.common.log import setup_logging
from bayn.common.schemas import AUDIO_SR, TOPIC_AUDIO_PCM, TOPIC_SHOWCASE
from bayn.transcription.audio_sources import read_wav, write_wav

log = logging.getLogger("bayn.showcase")
SCRIPT = ROOT / "samples" / "showcase" / "showcase.json"
OUT_WAV = DATA_DIR / "showcase" / "showcase.wav"
TRIGGERS = ("clench", "triple_blink", "manual", "seizure")
GESTURE_FOR = {"clench": "clench", "triple_blink": "triple_blink", "manual": "manual", "seizure": "manual"}
INTRO = {"title": "Bayn showcase", "caption": "Simulated EEG and recorded speech, no hardware needed. Four short "
         "scenes show the ways an event is triggered; watch the transcript and the summary fill in."}


@dataclass(slots=True)
class Scene:
    index: int
    title: str
    caption: str
    trigger: str
    start_s: float          # caption appears (position in the showcase audio)
    audio_s: float          # this scene's speech starts
    audio_end_s: float
    trigger_s: float        # trigger fires


@dataclass(slots=True)
class Timeline:
    scenes: list[Scene]
    length_s: float

    def to_dict(self) -> dict:
        return {"length_s": self.length_s, "scenes": [asdict(s) for s in self.scenes]}


def plan(script: dict, durations: list[float]) -> Timeline:
    """Scene times from the script and each recording's duration (pure: testable without audio)."""
    t = float(script.get("lead_s", 4.0))
    scenes = []
    for i, (sc, dur) in enumerate(zip(script["scenes"], durations)):
        if sc["trigger"] not in TRIGGERS:
            raise ValueError(f"scene {i + 1}: trigger must be one of {TRIGGERS}")
        lead = max(0.0, -float(sc["trigger_at_s"]))          # EEG must start before the speech (seizure clip)
        audio_s = t + lead
        scenes.append(Scene(i + 1, sc["title"], sc["caption"], sc["trigger"], start_s=t, audio_s=audio_s,
                            audio_end_s=audio_s + dur, trigger_s=audio_s + float(sc["trigger_at_s"])))
        t = audio_s + dur + float(sc.get("after_s", 14.0))
    return Timeline(scenes, t)


def build(script_path: Path = SCRIPT, out: Path = OUT_WAV) -> tuple[Path, Timeline]:
    """Stitch the scenes' recordings, with silence between them, into one WAV (16 kHz mono)."""
    script = json.loads(Path(script_path).read_text(encoding="utf-8"))
    clips = [read_wav(Path(script_path).parent / sc["audio"]) for sc in script["scenes"]]
    tl = plan(script, [len(c) / AUDIO_SR for c in clips])
    audio = np.zeros(int(round(tl.length_s * AUDIO_SR)), dtype=np.float32)
    for sc, clip in zip(tl.scenes, clips):
        i = int(round(sc.audio_s * AUDIO_SR))
        audio[i:i + len(clip)] = clip
    out.parent.mkdir(parents=True, exist_ok=True)
    write_wav(out, audio)
    out.with_suffix(".json").write_text(json.dumps(tl.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return out, tl


class Driver:
    """Follows the showcase audio's position (pure logic; the process loop feeds it audio chunk times)."""

    def __init__(self, tl: Timeline) -> None:
        self.tl = tl
        self.t0: float | None = None
        self.fired: set[tuple[int, int]] = set()      # (loop, scene)
        self.scene: Scene | None = None
        self.loop = 0

    def position(self, chunk_t0: float, seq: int, chunk_end: float) -> float:
        if self.t0 is None or seq == 0:                 # first chunk, or the audio process restarted
            self.t0, self.fired = chunk_t0, set()
        return chunk_end - self.t0

    def step(self, pos: float) -> tuple[list[Scene], Scene | None]:
        """Triggers due at this position, and the scene to caption now (None = intro). The dashboard's
        trigger setting follows the scene (set by run() when the scene changes)."""
        self.loop, within = divmod(pos, self.tl.length_s)
        due = []
        for sc in self.tl.scenes:
            key = (int(self.loop), sc.index)
            if within >= sc.trigger_s and key not in self.fired:
                self.fired.add(key)
                if within - sc.trigger_s < 3.0:          # never fire a trigger we jumped far past
                    due.append(sc)
        now = [sc for sc in self.tl.scenes if sc.start_s <= within]
        self.scene = now[-1] if now else None
        return due, self.scene


def caption(sc: Scene | None, total: int, t: float, pos: float = 0.0) -> dict:
    """pos = position in the tour (s), so a recording of the dashboard can be aligned with the scenes."""
    if sc is None:
        return {"index": 0, "total": total, "t": t, "pos": round(pos, 2), **INTRO}
    return {"index": sc.index, "total": total, "title": sc.title, "caption": sc.caption, "trigger": sc.trigger,
            "t": t, "pos": round(pos, 2)}


def fire(sc: Scene, eeg: CommandSender, events: CommandSender) -> None:
    if sc.trigger == "manual":
        events.send({"cmd": "trigger", "source": "button"})
    else:
        eeg.send({"cmd": "inject", "gesture": sc.trigger, "delay_s": 0.05})
    log.info("scene %d: %s", sc.index, sc.trigger)


def run(cfg: AppConfig) -> None:
    tl_path = OUT_WAV.with_suffix(".json")
    if not tl_path.exists():
        build()
    d = json.loads(tl_path.read_text(encoding="utf-8"))
    tl = Timeline([Scene(**s) for s in d["scenes"]], d["length_s"])
    drv = Driver(tl)
    audio = Subscriber(cfg.bus.audio_host, cfg.bus.audio_pub_port, [TOPIC_AUDIO_PCM])
    pub = Publisher(cfg.bus.bind_host, cfg.bus.showcase_pub_port)
    eeg = CommandSender(cfg.bus.eeg_host, cfg.bus.eeg_cmd_port)
    events = CommandSender(cfg.bus.events_host, cfg.bus.events_cmd_port)
    log.info("showcase: %d scenes, %.0f s, looping with the audio", len(tl.scenes), tl.length_s)
    last_caption, last_sent = None, 0.0
    events.send({"cmd": "set_seizure", "enabled": True})
    try:
        while True:
            msg = audio.recv(timeout_ms=500)
            if msg is not None:
                m = msg[1]
                n = len(m["pcm"]) // 2                    # int16 bytes
                pos = drv.position(m["t0"], m.get("seq", 1), m["t0"] + n / m.get("sr", AUDIO_SR))
                due, sc = drv.step(pos)
                for s in due:
                    fire(s, eeg, events)
                key = sc.index if sc else 0
                if key != last_caption and sc is not None:          # the dashboard's trigger follows the scene
                    events.send({"cmd": "set_gesture", "gesture": GESTURE_FOR[sc.trigger]})
                if key != last_caption or time.time() - last_sent > 2.0:   # repeat: pages opened later
                    pub.send(TOPIC_SHOWCASE, caption(sc, len(tl.scenes), time.time(), pos % tl.length_s))
                    last_caption, last_sent = key, time.time()
    finally:
        for s in (audio, pub, eeg, events):
            s.close()


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    setup_logging(cfg.logging.level)
    if argv is None:
        argv = sys.argv[1:]
    if argv[:1] == ["build"]:
        path, tl = build()
        print(f"{path} ({tl.length_s:.0f} s)")
        for s in tl.scenes:
            print(f"  {s.index}. {s.title}: caption {s.start_s:.0f} s, speech {s.audio_s:.0f}-{s.audio_end_s:.0f} s, "
                  f"{s.trigger} at {s.trigger_s:.0f} s")
        return 0
    try:
        run(cfg)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
