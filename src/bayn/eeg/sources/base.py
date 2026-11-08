"""The EEGSource interface every acquisition backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

from bayn.common.schemas import EEGChunk


class EEGSource(ABC):
    """A stream of EEG chunks in µV, channel order = `channels`.

    Lifecycle: start() -> read() repeatedly -> stop(). Also usable as a context manager.
    """

    name: str = "base"
    supports_injection: bool = False

    def __init__(self, channels: list[str], fs: float) -> None:
        self.channels = list(channels)
        self.fs = float(fs)

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def read(self, timeout: float = 1.0) -> EEGChunk | None:
        """Block up to `timeout` s for the next chunk; None if nothing arrived."""

    @abstractmethod
    def stop(self) -> None: ...

    def inject(self, gesture: str) -> float | None:
        """Simulator-only: schedule a fake gesture. Returns its onset time (unix s) or None."""
        return None

    def __enter__(self) -> "EEGSource":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
