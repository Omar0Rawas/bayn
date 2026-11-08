"""Tiny ZeroMQ wrappers: PUB/SUB for streams, PUSH/PULL for commands.

Why ZeroMQ: plain TCP works identically between host processes and Docker
containers (LSL's multicast discovery does not on Docker Desktop for Windows).
Frames: [topic utf-8, msgpack payload]. NumPy arrays are packed as raw bytes.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import msgpack
import numpy as np
import zmq


# ---------------------------------------------------------------- encoding
def _encode(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return {"__nd__": True, "dtype": obj.dtype.str, "shape": list(obj.shape),
                "data": np.ascontiguousarray(obj).tobytes()}
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"cannot serialize {type(obj)!r}")


def _decode(obj: dict) -> Any:
    if obj.get("__nd__"):
        return np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])
    return obj


def pack(payload: Any) -> bytes:
    return msgpack.packb(payload, default=_encode, use_bin_type=True)


def unpack(raw: bytes) -> Any:
    return msgpack.unpackb(raw, object_hook=_decode, raw=False)


def endpoint(host: str, port: int) -> str:
    return f"tcp://{host}:{port}"


def _bind(sock: zmq.Socket, host: str, port: int) -> int:
    """Bind; port 0 picks a free port (handy in tests). Returns the bound port."""
    if port == 0:
        return sock.bind_to_random_port(f"tcp://{host}")
    sock.bind(endpoint(host, port))
    return port


# ---------------------------------------------------------------- sync sockets
class Publisher:
    def __init__(self, host: str, port: int, ctx: zmq.Context | None = None) -> None:
        self._sock = (ctx or zmq.Context.instance()).socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, 1000)
        self._sock.setsockopt(zmq.LINGER, 0)
        self.port = _bind(self._sock, host, port)

    def send(self, topic: str, payload: Any) -> None:
        self._sock.send_multipart([topic.encode(), pack(payload)])

    def close(self) -> None:
        self._sock.close()


class Subscriber:
    def __init__(self, host: str, port: int, topics: list[str] | tuple[str, ...] = ("",),
                 ctx: zmq.Context | None = None) -> None:
        self._sock = (ctx or zmq.Context.instance()).socket(zmq.SUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.RCVHWM, 1000)
        for t in topics:
            self._sock.setsockopt(zmq.SUBSCRIBE, t.encode())
        self._sock.connect(endpoint(host, port))

    def recv(self, timeout_ms: int = 1000) -> tuple[str, Any] | None:
        if not self._sock.poll(timeout_ms):
            return None
        topic, raw = self._sock.recv_multipart()
        return topic.decode(), unpack(raw)

    def close(self) -> None:
        self._sock.close()


class CommandReceiver:
    """PULL socket polled without blocking from a producer's main loop."""

    def __init__(self, host: str, port: int, ctx: zmq.Context | None = None) -> None:
        self._sock = (ctx or zmq.Context.instance()).socket(zmq.PULL)
        self._sock.setsockopt(zmq.LINGER, 0)
        self.port = _bind(self._sock, host, port)

    def poll(self) -> list[dict]:
        out = []
        while self._sock.poll(0):
            out.append(unpack(self._sock.recv()))
        return out

    def close(self) -> None:
        self._sock.close()


class CommandSender:
    """PUSH socket. send() never blocks: returns False if the receiver is gone and the queue is full."""

    def __init__(self, host: str, port: int, ctx: zmq.Context | None = None) -> None:
        self._sock = (ctx or zmq.Context.instance()).socket(zmq.PUSH)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.SNDHWM, 100)
        self._sock.connect(endpoint(host, port))

    def send(self, payload: dict) -> bool:
        try:
            self._sock.send(pack(payload), flags=zmq.NOBLOCK)
            return True
        except zmq.Again:
            return False

    def close(self) -> None:
        self._sock.close()


# ---------------------------------------------------------------- asyncio bridge
# pyzmq's asyncio sockets need a selector event loop (or tornado) on Windows, which
# uvicorn does not use. A plain socket in a daemon thread works on every OS.
class ThreadedSubscriber:
    """Receives in a background thread and hands each (topic, payload) to an asyncio queue."""

    def __init__(self, host: str, port: int, topics: list[str] | tuple[str, ...],
                 loop: asyncio.AbstractEventLoop, maxsize: int = 2000) -> None:
        self.queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize)
        self._args = (host, port, list(topics))
        self._loop = loop
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"zmq-sub-{port}", daemon=True)
        self._thread.start()

    def _put(self, item: tuple[str, Any]) -> None:
        if self.queue.full():           # drop oldest rather than block the UI on a slow client
            self.queue.get_nowait()
        self.queue.put_nowait(item)

    def _run(self) -> None:
        sub = Subscriber(*self._args)
        try:
            while not self._stop.is_set():
                if (msg := sub.recv(timeout_ms=200)) is not None:
                    self._loop.call_soon_threadsafe(self._put, msg)
        finally:
            sub.close()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
