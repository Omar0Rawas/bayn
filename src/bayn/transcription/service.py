"""Transcription service: audio stream -> utterances (VAD) -> Whisper -> timestamped segments.

    python -m bayn.transcription.service

Transcribes continuously (so speech from *before* an event is available too). When an
event's EEG window is saved (events.done), it attaches the transcript of that window
to the event (transcript.json / transcript.txt in the event folder) and publishes
transcript.window -> input for the Phase 5 summary.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass

from bayn.common.bus import CommandReceiver, Publisher, Subscriber
from bayn.common.config import AppConfig, load_config
from bayn.common.log import setup_logging
from bayn.common.schemas import (TOPIC_AUDIO_PCM, TOPIC_EVENT_DONE, TOPIC_TRANSCRIPT, TOPIC_TRANSCRIPT_PARTIAL,
                                 TOPIC_TRANSCRIPT_STATUS, TOPIC_TRANSCRIPT_WINDOW, AudioChunk)
from bayn.events.store import EventStore
from bayn.transcription.engine import WhisperEngine, silero_vad
from bayn.transcription.engine_single import SingleEncodeWhisperEngine
from bayn.transcription.segmenter import Segmenter, Utterance
from bayn.transcription.store import TranscriptStore, format_transcript

log = logging.getLogger("bayn.stt")
LANGUAGES = ("mixed", "auto", "en", "ar")
STATUS_EVERY_S = 1.0
AUDIO_STALE_S = 3.0
TAIL_MAX_PAUSE_S = 1.5       # a sentence "continues" past the window end if the pause is shorter


@dataclass
class PendingWindow:
    event: dict
    t0: float
    t1: float


class TranscriptionService:
    def __init__(self, cfg: AppConfig, engine: WhisperEngine | None = None,
                 transcripts: TranscriptStore | None = None, events: EventStore | None = None) -> None:
        self.cfg = cfg
        b, tc = cfg.bus, cfg.transcription
        self.pub = Publisher(b.bind_host, b.stt_pub_port)
        self.language = tc.language
        self._status("loading", "loading Whisper model...")
        try:
            # the copy decides itself (auto: only on the CPU) once it knows where Whisper loaded
            self.engine = engine or (WhisperEngine if tc.single_encode is False else SingleEncodeWhisperEngine)(tc)
        except Exception:
            self.pub.close()          # free the port for degraded mode
            raise
        self.cmd = CommandReceiver(b.bind_host, b.stt_cmd_port)
        self.audio = Subscriber(b.audio_host, b.audio_pub_port, [TOPIC_AUDIO_PCM])
        self.events_sub = Subscriber(b.events_host, b.events_pub_port, [TOPIC_EVENT_DONE])
        self.segmenter = Segmenter(silero_vad(tc.vad_threshold), min_silence_s=tc.min_silence_s,
                                   max_utterance_s=tc.max_utterance_s)
        self.transcripts = transcripts or TranscriptStore()
        self.events = events or EventStore()
        self.pending: list[PendingWindow] = []
        self.last_audio_wall = 0.0
        self.last_partial = 0.0
        self.last_latency: float | None = None
        self.n_segments = 0
        self.speaking = False
        self._last_status = 0.0

    # ------------------------------------------------------------ helpers
    def _status(self, state: str, message: str = "") -> None:
        info = self.engine.describe() if hasattr(self, "engine") else {}
        self.pub.send(TOPIC_TRANSCRIPT_STATUS, {
            "state": state, "message": message, "language": self.language, **info,
            "speaking": getattr(self, "speaking", False), "last_latency_s": getattr(self, "last_latency", None),
            "segments": getattr(self, "n_segments", 0),
            "audio_ok": time.time() - getattr(self, "last_audio_wall", 0.0) < AUDIO_STALE_S, "t": time.time()})

    def _transcribe_final(self, u: Utterance) -> None:
        r = self.engine.transcribe(u.audio, self.language)
        uid = uuid.uuid4().hex[:12]
        now = time.time()
        self.last_latency = now - u.t_end            # end of speech -> text available
        for i, s in enumerate(r.segments):
            seg = {"id": f"{uid}-{i}", "utterance_id": uid, "t_start": u.t_start + s.start,
                   "t_end": u.t_start + min(s.end, u.seconds), "text": s.text, "lang": r.language,
                   "lang_prob": round(r.language_prob, 3), "latency_s": round(self.last_latency, 2)}
            self.transcripts.add(seg)
            self.pub.send(TOPIC_TRANSCRIPT, {**seg, "final": True, "forced_cut": u.forced})
            self.n_segments += 1
        # tell the UI the in-progress line is done (even if Whisper found no words)
        self.pub.send(TOPIC_TRANSCRIPT_PARTIAL, {"text": "", "t_start": u.t_start, "done": True, "t": now})
        if r.segments:
            log.info("[%s %.2fs] %s", r.language, r.seconds, r.text)

    def _transcribe_partial(self, u: Utterance) -> None:
        r = self.engine.transcribe(u.audio, self.language, fast=True)
        self.pub.send(TOPIC_TRANSCRIPT_PARTIAL, {"text": r.text, "lang": r.language, "t_start": u.t_start,
                                                 "done": False, "t": time.time()})

    # ------------------------------------------------------------ event windows
    def _on_event_done(self, meta: dict) -> None:
        t = meta["t"]
        self.pending.append(PendingWindow(meta, t - meta.get("pre_s", 10), t + meta.get("post_s", 30)))

    def _window_ready(self, w: PendingWindow, now: float) -> bool:
        tail = self.cfg.transcription.event_tail_s
        done_until = self.segmenter.processed_until()
        if done_until is not None and done_until >= w.t1 + tail:
            return True                                   # all speech up to the window end (+ tail) is transcribed
        audio_dead = now - self.last_audio_wall > AUDIO_STALE_S
        return now >= w.t1 + tail + (3.0 if audio_dead else self.cfg.transcription.event_grace_s)

    def window_segments(self, t0: float, t1: float) -> list[dict]:
        """Segments in [t0, t1], plus speech that continues past t1 without a real pause
        (up to event_tail_s), so the last sentence is not cut in half."""
        segs = self.transcripts.window(t0, t1)
        last_end = segs[-1]["t_end"] if segs else None
        for s in self.transcripts.window(t1, t1 + self.cfg.transcription.event_tail_s):
            if last_end is None or s["t_start"] > last_end + TAIL_MAX_PAUSE_S:
                break
            if s["id"] not in {x["id"] for x in segs}:
                segs.append(s)
                last_end = s["t_end"]
        return segs

    def _attach(self, w: PendingWindow) -> None:
        ev = w.event
        segs = self.window_segments(w.t0, w.t1)
        text = format_transcript(segs, ev["t"])
        langs = sorted({s["lang"] for s in segs})
        payload = {"event_id": ev["id"], "t_event": ev["t"], "t0": w.t0, "t1": w.t1, "segments": segs,
                   "text": text, "languages": langs, "n_segments": len(segs)}
        try:
            d = self.events.dir_for(ev["id"])
            d.mkdir(parents=True, exist_ok=True)
            (d / "transcript.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            (d / "transcript.txt").write_text(text + "\n", encoding="utf-8")
            meta = self.events.get(ev["id"]) or ev
            meta["transcript"] = {"n_segments": len(segs), "languages": langs, "chars": len(text)}
            meta.setdefault("files", {}).update({n: f"events/{ev['id']}/{n}" for n in ("transcript.txt", "transcript.json")})
            self.events.upsert(meta)
        except (OSError, ValueError) as e:
            log.warning("could not save transcript for %s: %s", ev["id"], e)
        self.pub.send(TOPIC_TRANSCRIPT_WINDOW, payload)
        log.info("transcript attached to %s: %d segments", ev["id"], len(segs))

    # ------------------------------------------------------------ main loop
    def run(self) -> None:
        tc = self.cfg.transcription
        log.info("transcription ready: %s, language=%s", self.engine.describe(), self.language)
        self._status("ready")
        try:
            while True:
                msg = self.audio.recv(timeout_ms=50)
                while msg is not None:                      # drain everything that arrived
                    c = AudioChunk.from_msg(msg[1])
                    self.segmenter.feed(c.samples, c.t0)
                    self.last_audio_wall = time.time()
                    msg = self.audio.recv(timeout_ms=0)
                while (ev := self.events_sub.recv(timeout_ms=0)) is not None:
                    self._on_event_done(ev[1])
                for c in self.cmd.poll():
                    if c.get("cmd") == "set_language" and c.get("language") in LANGUAGES:
                        self.language = c["language"]
                        log.info("language mode: %s", self.language)

                res = self.segmenter.step()
                self.speaking = res.speaking
                for u in res.finals:
                    self._transcribe_final(u)
                now = time.time()
                if (res.partial is not None and self.engine.partials_enabled and res.partial.seconds >= 1.0
                        and now - self.last_partial >= tc.partial_every_s):
                    self.last_partial = now
                    self._transcribe_partial(res.partial)
                if now - self.last_audio_wall > AUDIO_STALE_S and self.segmenter.buf_t0 is not None:
                    for u in self.segmenter.flush():       # audio stopped mid-sentence: finish it
                        self._transcribe_final(u)
                for w in list(self.pending):
                    if self._window_ready(w, now):
                        self.pending.remove(w)
                        self._attach(w)
                if now - self._last_status >= STATUS_EVERY_S:
                    self._last_status = now
                    self._status("ready")
        finally:
            self.audio.close(); self.events_sub.close(); self.pub.close(); self.cmd.close()


def _degraded(cfg: AppConfig, error: str) -> None:
    """Whisper unavailable: keep reporting why (the EEG/trigger side of the demo keeps working)."""
    pub = Publisher(cfg.bus.bind_host, cfg.bus.stt_pub_port)
    log.error("transcription disabled: %s", error)
    while True:
        pub.send(TOPIC_TRANSCRIPT_STATUS, {"state": "error", "message": error, "language": cfg.transcription.language,
                                           "t": time.time()})
        time.sleep(2.0)


def main() -> None:
    cfg = load_config()
    setup_logging(cfg.logging.level)
    try:
        try:
            svc = TranscriptionService(cfg)
        except Exception as e:  # no model downloaded, broken install, ...
            svc = None
            err = str(e)
        if svc is None:
            _degraded(cfg, err)
        svc.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
