"""Minimal client for a LOCAL Ollama server (http://127.0.0.1:11434). No cloud endpoints."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

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


@contextmanager
def local_server(host: str) -> Iterator[bool]:
    """Make sure a local Ollama server runs for the duration (start `ollama serve` if needed, stop it after).

    Use this rather than the `ollama` CLI: on Windows `ollama list` without a server starts the tray app,
    which ignores OLLAMA_MODELS from stale environments and can't always start its own server (E4)."""
    proc = None
    if cmd := ollama_serve_command(host):
        proc = subprocess.Popen(cmd, env={**os.environ, **ollama_env()}, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for _ in range(60):
            if OllamaClient(host).version():
                break
            time.sleep(0.5)
    try:
        yield OllamaClient(host).version() is not None
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()


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

    def loaded(self) -> list[dict]:
        """Models in memory now (/api/ps): name, size (bytes), size_vram (bytes on the GPU)."""
        try:
            return httpx.get(f"{self.host}/api/ps", timeout=5).json().get("models", [])
        except (httpx.HTTPError, ValueError):
            return []

    def warm(self, model: str, keep_alive: str = "60m", options: dict | None = None) -> float:
        """Load the model into (GPU) memory now, so the first real summary starts instantly."""
        t = time.perf_counter()
        body = {"model": model, "prompt": "", "keep_alive": keep_alive}
        if options:
            body["options"] = options
        r = httpx.post(f"{self.host}/api/generate", json=body, timeout=self.timeout)
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
             json_schema: dict | None = None, options: dict | None = None) -> ChatResult:
        """Streamed chat completion; on_token receives each piece of text as it arrives.
        json_schema: force the answer into this JSON shape (Ollama structured outputs).
        options: extra Ollama options, e.g. {"num_gpu": 0, "num_thread": 4} = CPU only.
        Reasoning ("thinking") models are asked not to think: a summary doesn't need it and it is slow."""
        body = {"model": model, "messages": messages, "stream": True, "keep_alive": keep_alive, "think": False,
                "options": {"temperature": temperature, "num_ctx": num_ctx, "repeat_penalty": 1.15,
                            **(options or {})}}
        if json_schema:
            body["format"] = json_schema
        try:
            return self._chat(body, on_token)
        except OllamaError as e:
            if "think" not in str(e).lower():
                raise
            body.pop("think")                              # model/server without the thinking switch
            return self._chat(body, on_token)

    def _chat(self, body: dict, on_token: Callable[[str], None] | None) -> ChatResult:
        t0 = time.perf_counter()
        first, parts, stats = None, [], {}
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
