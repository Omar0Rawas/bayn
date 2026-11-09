"""Summarizer service: transcript of an event window + EEG stats -> local LLM summary.

    python -m bayn.summarization.service

Waits for `transcript.window` (sent by the transcriber once an event's window is complete),
asks the local Ollama model for a structured summary, streams it to the dashboard while it
is being written, and saves summary.md / summary.json in the event folder.
"""

from __future__ import annotations

import json
import logging
import threading
import time

from bayn.common.bus import CommandReceiver, Publisher, Subscriber
from bayn.common.config import AppConfig, load_config
from bayn.common.log import setup_logging
from bayn.common.schemas import TOPIC_SUMMARY, TOPIC_SUMMARY_STATUS, TOPIC_SUMMARY_TOKEN, TOPIC_TRANSCRIPT_WINDOW
from bayn.events.store import EventStore
from bayn.summarization.models import resolve_model
from bayn.summarization.ollama_client import OllamaClient, OllamaError
from bayn.summarization.prompts import (TEXT, assemble, build_messages, eeg_section, output_languages,
                                        partial_render, render, schema)

log = logging.getLogger("bayn.summary")
LANGS = ("auto", "en", "ar", "both")
TOKEN_EVERY_S = 0.08          # throttle streamed updates to ~12 per second


class SummaryService:
    def __init__(self, cfg: AppConfig, client: OllamaClient | None = None, events: EventStore | None = None) -> None:
        self.cfg = cfg
        b, s = cfg.bus, cfg.summary
        self.pub = Publisher(b.bind_host, b.sum_pub_port)
        self.cmd = CommandReceiver(b.bind_host, b.sum_cmd_port)
        self.sub = Subscriber(b.stt_host, b.stt_pub_port, [TOPIC_TRANSCRIPT_WINDOW])
        self.client = client or OllamaClient(s.host, s.timeout_s)
        self.events = events or EventStore()
        self.language = s.language
        self.model: str | None = None
        self.state, self.message = "starting", ""
        self.last: dict = {}
        self.windows: dict[str, dict] = {}          # event_id -> transcript window (for regenerate)
        self._ready = threading.Event()
        threading.Thread(target=self._prepare, name="ollama-warmup", daemon=True).start()

    # ------------------------------------------------------------ model
    def _prepare(self) -> None:
        """Find Ollama + the model and load it into memory (the first load can take a minute)."""
        while True:
            if self.client.version() is None:
                self.state, self.message = "error", f"Ollama not running at {self.cfg.summary.host} - start the Ollama app"
                time.sleep(3)
                continue
            available = self.client.models()
            model = resolve_model(self.cfg.summary, available)
            if model not in available and f"{model}:latest" not in available:
                self.state, self.message = "error", f"model {model} not downloaded - run: bayn models"
                time.sleep(5)
                continue
            self.model = model
            self.state, self.message = "loading", f"loading {model}..."
            try:
                secs = self.client.warm(model, self.cfg.summary.keep_alive)
                self.state, self.message = "ready", ""
                log.info("summary model %s loaded in %.1f s", model, secs)
                self._ready.set()
                return
            except (OllamaError, Exception) as e:  # noqa: BLE001 - keep retrying, report why
                self.state, self.message = "error", f"could not load {model}: {e}"
                time.sleep(5)

    def _status(self) -> None:
        self.pub.send(TOPIC_SUMMARY_STATUS, {"state": self.state, "message": self.message, "model": self.model,
                                             "language": self.language, "last": self.last, "t": time.time()})

    # ------------------------------------------------------------ summarising
    def summarize(self, window: dict, langs: list[str] | None = None) -> list[dict]:
        ev_id = window["event_id"]
        self.windows[ev_id] = window
        event = self.events.get(ev_id) or {"id": ev_id, "t": window["t_event"], "label": "Event",
                                            "detector": "manual", "pre_s": 10, "post_s": 30}
        stats = event.get("stats")
        segs = window.get("segments", [])
        langs = langs or output_languages(self.language, segs)
        done = []
        for lang in langs:
            done.append(self._summarize_one(event, segs, stats, lang))
        self._save(event, done)
        return done

    def _summarize_one(self, event: dict, segs: list[dict], stats: dict | None, lang: str) -> dict:
        ev_id = event["id"]
        base = {"event_id": ev_id, "lang": lang, "model": self.model, "eeg": eeg_section(stats, lang)}
        if not segs:                                       # nothing was said: no LLM needed
            text = f"**{TEXT[lang]['no_speech']}**"
            msg = {**base, "llm_text": text, "markdown": assemble(event, text, stats, lang, self.model or "-"),
                   "seconds": 0.0, "tokens_per_s": 0, "error": None, "prompt": None}
            self.pub.send(TOPIC_SUMMARY, msg)
            return msg
        messages = build_messages(event, segs, lang, self.cfg.summary.max_key_points, structured=True)
        if not self._ready.wait(timeout=120):
            msg = {**base, "llm_text": "", "markdown": "", "error": self.message or "summary model not ready",
                   "seconds": 0.0, "tokens_per_s": 0, "prompt": messages}
            self.pub.send(TOPIC_SUMMARY, msg)
            return msg
        buf, last_sent = [], [0.0]

        def on_token(piece: str) -> None:
            buf.append(piece)
            now = time.time()
            if now - last_sent[0] < TOKEN_EVERY_S:
                return
            text = partial_render("".join(buf), lang)
            if text:                                   # nothing readable yet: don't use up the throttle
                last_sent[0] = now
                self.pub.send(TOPIC_SUMMARY_TOKEN, {"event_id": ev_id, "lang": lang, "text": text})

        self.pub.send(TOPIC_SUMMARY_TOKEN, {"event_id": ev_id, "lang": lang, "text": "", "started": True,
                                            "prompt": messages, "model": self.model, "eeg": base["eeg"]})
        error, text, r = None, "", None
        for attempt in (1, 2):                             # one retry: small models occasionally loop
            try:
                buf.clear()
                r = self.client.chat(self.model, messages, self.cfg.summary.temperature + 0.2 * (attempt - 1),
                                     self.cfg.summary.keep_alive, on_token,
                                     json_schema=schema(self.cfg.summary.max_key_points))
                text = render(json.loads(r.text), lang)
                error = None
                break
            except (OllamaError, ValueError) as e:
                error = str(e)
                log.warning("summary attempt %d for %s failed: %s", attempt, ev_id, e)
        msg = {**base, "llm_text": text, "error": error, "prompt": messages,
               "markdown": assemble(event, text, stats, lang, self.model) if text else "",
               "seconds": round(r.total_s, 2) if r else 0.0, "tokens_per_s": round(r.tokens_per_s) if r else 0,
               "first_token_s": round(r.first_token_s, 2) if r else None}
        self.last = {"event_id": ev_id, "seconds": msg["seconds"], "tokens_per_s": msg["tokens_per_s"]}
        self.pub.send(TOPIC_SUMMARY, msg)
        log.info("summary %s [%s] in %.1f s%s", ev_id, lang, msg["seconds"], f" (error: {error})" if error else "")
        return msg

    def _save(self, event: dict, results: list[dict]) -> None:
        ok = [r for r in results if r.get("markdown")]
        if not ok:
            return
        try:
            d = self.events.dir_for(event["id"])
            d.mkdir(parents=True, exist_ok=True)
            (d / "summary.md").write_text("\n\n---\n\n".join(r["markdown"] for r in ok) + "\n", encoding="utf-8")
            (d / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            meta = self.events.get(event["id"]) or event
            meta["summary"] = {"model": self.model, "languages": [r["lang"] for r in ok],
                               "seconds": sum(r["seconds"] for r in ok)}
            meta.setdefault("files", {}).update({n: f"events/{event['id']}/{n}" for n in ("summary.md", "summary.json")})
            self.events.upsert(meta)
        except (OSError, ValueError) as e:
            log.warning("could not save summary for %s: %s", event["id"], e)

    # ------------------------------------------------------------ main loop
    def run(self) -> None:
        last_status = 0.0
        try:
            while True:
                msg = self.sub.recv(timeout_ms=200)
                if msg is not None:
                    self.summarize(msg[1])
                for c in self.cmd.poll():
                    if c.get("cmd") == "set_language" and c.get("language") in LANGS:
                        self.language = c["language"]
                    elif c.get("cmd") == "regenerate" and c.get("event_id"):
                        w = self.windows.get(c["event_id"]) or self._load_window(c["event_id"])
                        if w:
                            lang = c.get("lang")
                            self.summarize(w, [lang] if lang in ("en", "ar") else None)
                now = time.time()
                if now - last_status >= 1.0:
                    last_status = now
                    self._status()
        finally:
            self.sub.close(); self.pub.close(); self.cmd.close()

    def _load_window(self, event_id: str) -> dict | None:
        try:
            return json.loads((self.events.dir_for(event_id) / "transcript.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None


def main() -> None:
    cfg = load_config()
    setup_logging(cfg.logging.level)
    try:
        SummaryService(cfg).run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
