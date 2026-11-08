"""Messages exchanged on the bus. Plain dataclasses + to_msg/from_msg for msgpack."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

# ---- topics (ZeroMQ prefix matching: subscribing to "eeg." gets all EEG topics)
TOPIC_EEG_RAW = "eeg.raw"            # unfiltered µV (for EMG/clench detection)
TOPIC_EEG_FILT = "eeg.filt"          # 1-40 Hz + notch (display, stats, blink detection)
TOPIC_EEG_STATUS = "eeg.status"      # ~1 Hz: connection, per-channel quality, mains
TOPIC_EEG_INJECTED = "eeg.injected"  # simulator ground truth: a gesture was injected

GESTURES = ("clench", "double_blink", "blink")


@dataclass(slots=True)
class EEGChunk:
    """A block of consecutive samples. data: (n_channels, n_samples) float32 in µV."""

    t0: float                 # unix time (s) of the first sample
    fs: float
    channels: list[str]
    data: np.ndarray
    seq: int = 0              # chunk counter, lets consumers spot drops

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[1])

    @property
    def t_end(self) -> float:
        """Time just after the last sample (= t0 of the next contiguous chunk)."""
        return self.t0 + self.n_samples / self.fs

    def timestamps(self) -> np.ndarray:
        return self.t0 + np.arange(self.n_samples) / self.fs

    def to_msg(self) -> dict:
        return {"t0": self.t0, "fs": self.fs, "channels": self.channels,
                "data": np.ascontiguousarray(self.data, dtype=np.float32), "seq": self.seq}

    @classmethod
    def from_msg(cls, m: dict) -> "EEGChunk":
        return cls(t0=m["t0"], fs=m["fs"], channels=list(m["channels"]),
                   data=np.asarray(m["data"], dtype=np.float32), seq=m.get("seq", 0))


@dataclass(slots=True)
class ChannelQuality:
    channel: str
    level: str               # good | ok | bad
    noise_uv: float          # robust amplitude (MAD-based) of the filtered signal
    reason: str = ""


@dataclass(slots=True)
class EEGStatus:
    source: str              # sim | muse | lsl
    connected: bool
    fs: float
    channels: list[str]
    quality: list[ChannelQuality] = field(default_factory=list)
    mains_hz: int | None = None      # detected line-noise frequency, None = not (yet) clear
    notch: str = "auto"
    supports_injection: bool = False
    t: float = 0.0
    message: str = ""

    def to_msg(self) -> dict:
        return asdict(self)

    @classmethod
    def from_msg(cls, m: dict) -> "EEGStatus":
        m = dict(m)
        m["quality"] = [ChannelQuality(**q) for q in m.get("quality", [])]
        return cls(**m)
