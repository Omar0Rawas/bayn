"""A tiny stand-in for the Ollama HTTP API, so summarizer tests don't need a real LLM.

Speaks /api/version, /api/tags, /api/generate (warm-up) and streamed /api/chat. The chat
answer is a fixed structured summary in the language of the system prompt.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ANSWERS = {
    "en": {"summary": "A teacher explains that a cup of water left by the window lost water because the sun turned it into vapor, which is called evaporation.",
           "key_points": [{"time": "+1 s", "point": "A cup of water was left by the window."},
                          {"time": "+5 s", "point": "The water level went down."},
                          {"time": "+11 s", "point": "The sun heated the water into vapor: evaporation."}],
           "unclear": []},
    "ar": {"summary": "يشرح المعلم أن كأس الماء بجانب الشباك نقص لأن حرارة الشمس حولت الماء إلى بخار، وهذا هو التبخر.",
           "key_points": [{"time": "+4 s", "point": "وُضع كأس ماء بجانب الشباك."},
                          {"time": "+9 s", "point": "انخفض مستوى الماء."}],
           "unclear": ["فرحت (غالباً: فين راحت)"]},
}


class FakeOllama:
    def __init__(self, models: tuple[str, ...] = ("qwen2.5:3b", "qwen2.5:7b"), fail_chat: str | None = None,
                 rejects_think: bool = False) -> None:
        self.models = models
        self.fail_chat = fail_chat          # e.g. "token repeat limit reached"
        self.rejects_think = rejects_think  # like a server / model without the "think" switch
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):    # keep test output clean
                pass

            def _json(self, obj, code=200):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/api/version":
                    self._json({"version": "fake-0.1"})
                elif self.path == "/api/tags":
                    self._json({"models": [{"name": m} for m in outer.models]})
                elif self.path == "/api/ps":
                    self._json({"models": [{"name": outer.models[0], "size": 3 * 2**30, "size_vram": 2**31}]})
                else:
                    self._json({"error": "not found"}, 404)

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                outer.requests.append({"path": self.path, **req})
                if self.path == "/api/generate":
                    self._json({"done": True})
                    return
                if self.path != "/api/chat":
                    self._json({"error": "not found"}, 404)
                    return
                if outer.rejects_think and "think" in req:
                    self._json({"error": f'"{req["model"]}" does not support thinking'}, 400)
                    return
                system = req["messages"][0]["content"]
                lang = "ar" if "أنت" in system else "en"
                answer = json.dumps(ANSWERS[lang], ensure_ascii=False)
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                if outer.fail_chat:
                    self.wfile.write((json.dumps({"error": outer.fail_chat}) + "\n").encode())
                    return
                for i in range(0, len(answer), 12):                      # stream in small pieces
                    line = {"message": {"role": "assistant", "content": answer[i:i + 12]}, "done": False}
                    self.wfile.write((json.dumps(line, ensure_ascii=False) + "\n").encode())
                    self.wfile.flush()
                self.wfile.write((json.dumps({"done": True, "eval_count": 50, "eval_duration": 500_000_000}) + "\n").encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
