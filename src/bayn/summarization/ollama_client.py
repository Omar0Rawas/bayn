"""Minimal client for a LOCAL Ollama server (http://127.0.0.1:11434). No cloud endpoints."""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx


class OllamaError(RuntimeError):
    pass


def find_ollama() -> str | None:
    """Path of the ollama executable (PATH, or the default install folders per OS)."""
    if exe := shutil.which("ollama"):
        return exe
    candidates = [Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",   # Windows
                  Path("/Applications/Ollama.app/Contents/Resources/ollama"),                          # macOS
                  Path("/usr/local/bin/ollama"), Path("/usr/bin/ollama")]                              # Linux
    return next((str(p) for p in candidates if p.is_file()), None)


def ollama_env() -> dict[str, str]:
    """Settings for an Ollama server we start: local-only, and the user's model folder.

    On Windows, OLLAMA_MODELS set by `bayn setup` lives in the user's environment, which terminals
    opened *before* setup (e.g. VS Code) don't have yet - read it from the registry in that case.
    """
    env = {"OLLAMA_NO_CLOUD": "1"}
    if "OLLAMA_MODELS" in os.environ:
        env["OLLAMA_MODELS"] = os.environ["OLLAMA_MODELS"]
    elif os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                env["OLLAMA_MODELS"] = winreg.QueryValueEx(k, "OLLAMA_MODELS")[0]
        except OSError:
            pass
    return env


def ollama_serve_command(host: str) -> list[str] | None:
    """`ollama serve` if the configured server is local, not running, and Ollama is installed."""
    if not host.startswith(("http://127.0.0.1", "http://localhost")) or OllamaClient(host).version() is not None:
        return None
    exe = find_ollama()
    return [exe, "serve"] if exe else None


@dataclass
class ChatResult:
    text: str
    first_token_s: float
    total_s: float
    tokens: int
    tokens_per_s: float


class OllamaClient:
    def __init__(self, host: str = "http://127.0.0.1:11434", timeout_s: float = 180.0,
                 stall_s: float = 60.0) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout_s
        # streaming: give up if no new token arrives for stall_s (e.g. GPU memory exhausted)
        self.stream_timeout = httpx.Timeout(connect=5.0, read=stall_s, write=10.0, pool=5.0)

    def version(self) -> str | None:
        try:
            return httpx.get(f"{self.host}/api/version", timeout=2).json().get("version")
        except (httpx.HTTPError, ValueError):
            return None

    def models(self) -> list[str]:
        try:
            return [m["name"] for m in httpx.get(f"{self.host}/api/tags", timeout=5).json().get("models", [])]
        except (httpx.HTTPError, ValueError):
            return []

    def has_model(self, model: str) -> bool:
        names = self.models()
        return model in names or f"{model}:latest" in names

    def warm(self, model: str, keep_alive: str = "60m") -> float:
        """Load the model into (GPU) memory now, so the first real summary starts instantly."""
        t = time.perf_counter()
        r = httpx.post(f"{self.host}/api/generate", json={"model": model, "prompt": "", "keep_alive": keep_alive},
                       timeout=self.timeout)
        if r.status_code != 200:
            raise OllamaError(f"could not load {model}: {r.text[:200]}")
        return time.perf_counter() - t

    def unload(self, model: str) -> None:
        """Free the model's (GPU) memory now."""
        try:
            httpx.post(f"{self.host}/api/generate", json={"model": model, "keep_alive": 0}, timeout=30)
        except httpx.HTTPError:
            pass

    def chat(self, model: str, messages: list[dict], temperature: float = 0.2, keep_alive: str = "60m",
             on_token: Callable[[str], None] | None = None, num_ctx: int = 4096,
             json_schema: dict | None = None) -> ChatResult:
        """Streamed chat completion; on_token receives each piece of text as it arrives.
        json_schema: force the answer into this JSON shape (Ollama structured outputs)."""
        t0 = time.perf_counter()
        first, parts, stats = None, [], {}
        body = {"model": model, "messages": messages, "stream": True, "keep_alive": keep_alive,
                "options": {"temperature": temperature, "num_ctx": num_ctx, "repeat_penalty": 1.15}}
        if json_schema:
            body["format"] = json_schema
        try:
            with httpx.stream("POST", f"{self.host}/api/chat", json=body, timeout=self.stream_timeout) as r:
                if r.status_code != 200:
                    raise OllamaError(f"Ollama error {r.status_code}: {r.read().decode(errors='replace')[:200]}")
                for line in r.iter_lines():
                    if not line:
                        continue
                    m = json.loads(line)
                    if "error" in m:
                        raise OllamaError(m["error"])
                    piece = m.get("message", {}).get("content", "")
                    if piece:
                        first = first if first is not None else time.perf_counter() - t0
                        parts.append(piece)
                        if on_token:
                            on_token(piece)
                    if m.get("done"):
                        stats = m
        except httpx.ReadTimeout as e:
            raise OllamaError("the model stopped responding (GPU memory full? other models loaded?)") from e
        except httpx.HTTPError as e:
            raise OllamaError(f"Ollama not reachable at {self.host} ({e.__class__.__name__})") from e
        n = int(stats.get("eval_count", 0))
        dur = stats.get("eval_duration", 0) / 1e9
        return ChatResult("".join(parts), first or 0.0, time.perf_counter() - t0, n, n / dur if dur else 0.0)
