"""Guided, labelled recording sessions for training the double-blink classifier (Phase 6).

A session is a timeline of prompts driven by EEG data time (like calibration). Every prompt labels
its time window, so the recording can be cut into labelled examples afterwards:

  standard  (~5 min)    triple blinks (the trigger) plus what must NOT trigger: double and single blinks,
                        head turns, nods, sideways glances and talking, in random order, rest at the start.
  noisy     (~4.7 min)  continuous everyday activity (talking, looking around, moving the head, sitting
                        naturally) with triple-blink prompts in the middle of it. Measures robustness:
                        are triple blinks still caught, and does the activity cause false triggers?

Recordings are saved to data/training/<profile>-<source>/<session-id>.npz (raw µV + labels).
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from bayn.common.config import DATA_DIR

TRAINING_DIR = DATA_DIR / "training"

# label -> (prompt while doing it, what the person prepares for, simulator gesture to inject)
PROMPTS = {
    "triple_blink": ("Blink THREE times, firmly - now!", "Next: blink THREE times", "triple_blink"),
    "double_blink": ("Blink twice - now!", "Next: blink TWICE (not three)", "double_blink"),
    "blink": ("Blink once - now!", "Next: blink ONCE", "blink"),
    "head_turn": ("Turn your head to one side and back - now!", "Next: turn your head", "head_turn"),
    "nod": ("Nod a few times - now!", "Next: nod (yes, yes, yes)", "nod"),
    "look_side": ("Look to the side and back, eyes only - now!", "Next: look sideways (eyes only)", "look_side"),
}
ACTIVITIES = {   # label -> (prompt, simulator gestures injected during it, as offsets in s)
    "rest": ("Relax, look ahead, blink normally", []),
    "talk": ("Talk out loud: count, describe the room, anything", [(0.3, "talk"), (4.5, "talk")]),
    "look_around": ("Look around the room (eyes and head)", [(0.5, "look_side"), (4.0, "head_turn")]),
    "move": ("Move your head freely, stretch your neck", [(0.5, "head_turn"), (3.5, "nod")]),
    "natural": ("Sit naturally, blink normally", []),
}
CONDITIONS = ("seated", "standing", "walking")   # how the person did the session (walking: in place, spoken prompts)
MOVEMENTS = ("head_turn", "nod", "look_side", "look_around", "move", "look_updown", "eyebrows")
NEAR_MISSES = ("double_blink", "blink", "four_blinks", "squeeze", "natural_blinks")   # blinks that must NOT fire

# "boundaries" protocol (user idea, 2026-11-10): the person shows the EDGES of what counts as a triple blink
# and what doesn't, so the classifier (and synthetic data built from these recordings) covers the limits.
# (label, variant, prompt while doing it, what to prepare for, simulator gesture, duration s, repetitions)
BOUNDARY = [
    ("triple_blink", "normal", "Blink three times, your normal deliberate way - now!", "Next: THREE blinks, normal", "triple_blink", 4.0, 6),
    ("triple_blink", "soft", "Blink three times, as GENTLY as you still would on purpose - now!", "Next: THREE blinks, gentle", "triple_blink", 4.0, 5),
    ("triple_blink", "fast", "Blink three times, FAST - now!", "Next: THREE blinks, fast", "triple_blink", 3.5, 5),
    ("triple_blink", "slow", "Blink three times, SLOWLY (still one series) - now!", "Next: THREE blinks, slow", "triple_blink", 5.0, 4),
    ("triple_blink", "looking_down", "Look down at the desk and blink three times - now!", "Next: look down + THREE blinks", "triple_blink", 4.0, 4),
    ("triple_blink", "talking", "Keep talking and blink three times - now!", "Next: talk + THREE blinks", "triple_blink", 4.5, 4),
    ("double_blink", "", "Blink TWICE - now!", "Next: TWO blinks (not three)", "double_blink", 3.5, 5),
    ("four_blinks", "", "Blink FOUR times - now!", "Next: FOUR blinks", "four_blinks", 4.5, 4),
    ("blink", "hard", "One HARD blink - now!", "Next: ONE hard blink", "blink", 3.0, 4),
    ("squeeze", "", "Squeeze your eyes shut for a second - now!", "Next: squeeze your eyes shut", "squeeze", 3.5, 4),
    ("natural_blinks", "", "Blink naturally a few times, not on purpose", "Next: just blink naturally", "blink", 5.0, 4),
    ("nod", "three", "Nod THREE times (yes, yes, yes) - now!", "Next: nod three times", "nod", 4.0, 5),
    ("look_updown", "", "Look UP, then DOWN, eyes only - now!", "Next: look up and down (eyes only)", "look_updown", 3.5, 5),
    ("eyebrows", "", "Raise your eyebrows twice - now!", "Next: raise your eyebrows", "eyebrows", 3.5, 4),
]


@dataclass(slots=True)
class Step:
    kind: str                   # rest | pause | do | activity
    label: str                  # what happens in this window (PROMPTS / ACTIVITIES keys, or "pause")
    start: float
    end: float
    prompt: str
    inject: list = field(default_factory=list)      # simulator: [(offset_s, gesture)]
    variant: str = ""           # boundaries protocol: soft / fast / slow / ... (analysis only)


def build_steps(protocol: str, t0: float, seed: int | None = None, scale: float = 1.0) -> list[Step]:
    """The prompt timeline. scale < 1 shortens every duration (tests)."""
    rng = random.Random(seed)
    steps: list[Step] = []
    t = t0

    def add(kind, label, dur, prompt, inject=(), variant=""):
        nonlocal t
        steps.append(Step(kind, label, t, t + dur * scale, prompt, list(inject), variant))
        t += dur * scale

    if protocol == "standard":
        add("rest", "rest", 10, ACTIVITIES["rest"][0])
        order = ["triple_blink"] * 12 + ["double_blink"] * 8 + ["blink"] * 6 + ["head_turn"] * 8 + ["nod"] * 8 \
            + ["look_side"] * 8 + ["talk"] * 2
        rng.shuffle(order)
        for label in order:
            if label == "talk":
                add("pause", "pause", 2.0, "Next: talk for a few seconds")
                add("activity", "talk", 8.0, ACTIVITIES["talk"][0], ACTIVITIES["talk"][1])
            else:
                do, ready, gesture = PROMPTS[label]
                add("pause", "pause", 2.0, ready)
                add("do", label, 4.0 if label == "triple_blink" else 3.5, do, [(0.4, gesture)])
    elif protocol == "noisy":
        acts = ["talk", "look_around", "move", "natural"] * 2
        rng.shuffle(acts)
        for act in acts:
            prompt, inj = ACTIVITIES[act]
            add("activity", act, 8.0, prompt, inj)
            add("do", "triple_blink", 4.0, f"{prompt} - and blink THREE times now!", [(0.4, "triple_blink")])
            add("activity", act, 8.0, prompt, [(o * 0.8, g) for o, g in inj])
            add("do", "triple_blink", 4.0, f"{prompt} - and blink THREE times now!", [(0.4, "triple_blink")])
            add("activity", act, 5.0, prompt)
        add("activity", "natural", 30.0, ACTIVITIES["natural"][0])
    elif protocol == "boundaries":
        add("rest", "rest", 10, ACTIVITIES["rest"][0])
        order = [b for b in BOUNDARY for _ in range(b[6])]
        rng.shuffle(order)
        for label, variant, do, ready, gesture, dur, _ in order:
            add("pause", "pause", 2.0, ready)
            add("do", label, dur, do, [(0.4, gesture)], variant)
    else:
        raise ValueError(f"unknown protocol {protocol!r} (standard | noisy | boundaries)")
    return steps


class TrainingSession:
    """Runs a protocol against EEG data time and records every raw chunk."""

    def __init__(self, protocol: str, t0: float, seed: int | None = None, scale: float = 1.0,
                 condition: str = "seated") -> None:
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition {condition!r} ({' | '.join(CONDITIONS)})")
        self.protocol, self.condition = protocol, condition
        self.seed = seed if seed is not None else random.randrange(1 << 30)
        self.steps = build_steps(protocol, t0, self.seed, scale)
        self.t0, self.t_end = t0, self.steps[-1].end
        self.id = time.strftime("train-%Y%m%d-%H%M%S", time.localtime(t0)) + f"-{protocol}" \
            + ("" if condition == "seated" else f"-{condition}")
        self._idx = -1
        self._data: list[np.ndarray] = []
        self._ts: list[np.ndarray] = []

    @property
    def total_s(self) -> float:
        return self.t_end - self.t0

    def record(self, data: np.ndarray, ts: np.ndarray) -> None:
        if ts.size and ts[-1] >= self.t0 - 3.0:        # keep 3 s before the start (filter warm-up)
            self._data.append(np.asarray(data, np.float32))
            self._ts.append(np.asarray(ts, np.float64))

    def advance(self, now: float) -> tuple[Step | None, bool]:
        """(step that just started or None, finished?)."""
        if now >= self.t_end:
            return None, True
        if now < self.t0:
            return None, False
        i = next(k for k, s in enumerate(self.steps) if s.start <= now < s.end)
        if i != self._idx:
            self._idx = i
            return self.steps[i], False
        return None, False

    def current(self) -> Step | None:
        return self.steps[self._idx] if 0 <= self._idx < len(self.steps) else None

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.steps:
            if s.kind == "do":
                out[s.label] = out.get(s.label, 0) + 1
        return out

    def message(self, now: float, state: str = "running") -> dict:
        s = self.current()
        done = sum(1 for x in self.steps[: self._idx + 1] if x.kind == "do") if self._idx >= 0 else 0
        return {"state": state, "protocol": self.protocol, "condition": self.condition, "id": self.id, "step": s.kind if s else None,
                "label": s.label if s else None, "prompt": s.prompt if s else "Get ready...",
                "step_ends_at": s.end if s else self.t0, "t": now,
                "progress": round(min(1.0, max(0.0, (now - self.t0) / self.total_s)), 3),
                "prompts_done": done, "prompts_total": sum(self.counts().values())}

    def save(self, folder: Path, channels: list[str], fs: float, meta: dict | None = None) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{self.id}.npz"
        data = np.concatenate(self._data, axis=1) if self._data else np.zeros((len(channels), 0), np.float32)
        ts = np.concatenate(self._ts) if self._ts else np.zeros(0)
        labels = [asdict(s) for s in self.steps]
        info = {"id": self.id, "protocol": self.protocol, "condition": self.condition, "seed": self.seed,
                "channels": channels, "fs": fs,
                "t0": self.t0, "t_end": self.t_end, "created": time.time(), **(meta or {})}
        np.savez_compressed(path, data=data, ts=ts, labels=json.dumps(labels), info=json.dumps(info))
        return path


@dataclass
class Recording:
    """A saved training session, loaded for feature extraction / evaluation."""
    path: Path
    data: np.ndarray            # (n_ch, n) raw µV
    ts: np.ndarray              # (n,) unix s
    fs: float
    channels: list[str]
    steps: list[Step]
    info: dict

    @property
    def id(self) -> str:
        return self.info["id"]

    @classmethod
    def load(cls, path: Path) -> "Recording":
        z = np.load(path, allow_pickle=False)
        info = json.loads(str(z["info"]))
        steps = [Step(**{**s, "inject": [tuple(x) for x in s.get("inject", [])]}) for s in json.loads(str(z["labels"]))]
        return cls(Path(path), z["data"].astype(np.float32), z["ts"], float(info["fs"]), list(info["channels"]),
                   steps, info)

    def minutes(self) -> float:
        return (self.ts[-1] - self.ts[0]) / 60 if self.ts.size else 0.0

    @property
    def condition(self) -> str:
        return self.info.get("condition", "seated")       # sessions before 2026-11-10 were all seated


def simulate_session(protocol: str, folder: Path, seed: int = 0, scale: float = 1.0,
                     channels: tuple[str, ...] = ("TP9", "AF7", "AF8", "TP10"), fs: float = 256.0,
                     t0: float = 1_800_000_000.0, **sim_kw) -> Path:
    """A whole session recorded from the simulator, the simulated person doing every prompt
    (tests and the simulated demo; no real-time pacing, so it takes seconds)."""
    from bayn.eeg.sources.simulated import SimulatedEEG

    gen = SimulatedEEG(list(channels), fs, seed=seed, **sim_kw)
    session = TrainingSession(protocol, t0 + 3.0, seed, scale)
    k = 0
    while True:
        t = t0 + k / fs
        step, finished = session.advance(t)
        if finished:
            break
        if step is not None:
            for offset, gesture in step.inject:
                gen.inject(gesture, delay_s=offset)
        data = gen.generate(32)
        session.record(data, t + np.arange(32) / fs)
        k += 32
    return session.save(folder, list(channels), fs, {"simulated": True})


def session_summaries(profile: str, source: str) -> list[dict]:
    """Saved sessions (newest first) without loading the EEG: for the dashboard's list / re-record."""
    out = []
    for p in list_recordings(profile, source):
        try:
            with np.load(p, allow_pickle=False) as z:
                info = json.loads(str(z["info"]))
                n_triple = sum(1 for s in json.loads(str(z["labels"])) if s["kind"] == "do" and s["label"] == "triple_blink")
        except (OSError, ValueError, KeyError):
            continue
        out.append({"id": info["id"], "protocol": info.get("protocol"), "condition": info.get("condition", "seated"),
                    "minutes": round((info["t_end"] - info["t0"]) / 60, 1), "triple_prompts": n_triple,
                    "created": info.get("created")})
    return sorted(out, key=lambda s: -(s["created"] or 0))


def delete_recording(profile: str, source: str, session_id: str) -> bool:
    """Remove one saved session (to re-record it). Only plain session ids inside the folder are accepted."""
    if not session_id.startswith("train-") or any(c in session_id for c in "/\\.:"):
        return False
    path = session_folder(profile, source) / f"{session_id}.npz"
    if not path.exists():
        return False
    path.unlink()
    return True


def session_folder(profile: str, source: str) -> Path:
    return TRAINING_DIR / f"{profile}-{source}"


def list_recordings(profile: str, source: str) -> list[Path]:
    return sorted(session_folder(profile, source).glob("*.npz"))
