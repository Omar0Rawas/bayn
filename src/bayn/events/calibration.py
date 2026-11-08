"""Guided per-user calibration: rest, then N prompted repetitions of the gesture.

The session is a timeline of steps driven by EEG data time. The event manager feeds
it the current time, shows its prompts in the UI, and at the end hands the recorded
rest / gesture segments to each detector's calibrate().
(Phase 6 grows this into a Muse-app-style onboarding: "blink twice", "close your eyes"...)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bayn.common.config import CalibrationConfig

PROMPTS = {
    "clench": "Clench your jaw firmly and hold it",
    "double_blink": "Blink twice, firmly",
}


@dataclass(slots=True)
class Step:
    kind: str            # rest | ready | do | release
    gesture: str | None  # which gesture this step belongs to (None for rest)
    rep: int
    start: float
    end: float
    prompt: str


@dataclass
class CalibrationSession:
    cfg: CalibrationConfig
    gestures: list[str]                       # e.g. ["clench"] or ["clench", "double_blink"]
    t0: float
    steps: list[Step] = field(default_factory=list)
    _idx: int = -1

    def __post_init__(self) -> None:
        c, t = self.cfg, self.t0
        self.steps.append(Step("rest", None, 0, t, t + c.rest_s, "Relax, keep still and look ahead"))
        t += c.rest_s
        for g in self.gestures:
            for rep in range(1, c.reps + 1):
                for kind, dur, prompt in (("ready", c.ready_s, "Get ready..."),
                                          ("do", c.do_s, PROMPTS[g] + " - now!"),
                                          ("release", c.release_s, "Relax")):
                    self.steps.append(Step(kind, g, rep, t, t + dur, prompt))
                    t += dur
        self.t_end = t

    @property
    def total_s(self) -> float:
        return self.t_end - self.t0

    def advance(self, now: float) -> tuple[Step | None, bool]:
        """Returns (step that just started or None, finished?)."""
        if now >= self.t_end:
            return None, True
        i = next(k for k, s in enumerate(self.steps) if s.start <= now < s.end) if now >= self.t0 else 0
        if i != self._idx:
            self._idx = i
            return self.steps[i], False
        return None, False

    def current(self) -> Step | None:
        return self.steps[self._idx] if 0 <= self._idx < len(self.steps) else None

    def rest_window(self) -> tuple[float, float]:
        s = self.steps[0]
        return s.start + 2.0, s.end            # skip the first 2 s (settling)

    def gesture_windows(self, gesture: str) -> list[tuple[float, float]]:
        return [(s.start, s.end) for s in self.steps if s.kind == "do" and s.gesture == gesture]

    def message(self, now: float, state: str = "running") -> dict:
        s = self.current()
        reps = self.cfg.reps
        return {
            "state": state, "step": s.kind if s else None, "gesture": s.gesture if s else None,
            "rep": s.rep if s else 0, "reps": reps, "prompt": s.prompt if s else "",
            "step_ends_at": s.end if s else None, "t": now,
            "progress": round(min(1.0, max(0.0, (now - self.t0) / self.total_s)), 3),
            "gestures": self.gestures,
        }
