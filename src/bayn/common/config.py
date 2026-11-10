"""Configuration: config/default.yaml <- config/local.yaml <- .env <- environment variables.

Env override syntax: BAYN_<SECTION>__<KEY>[__<SUBKEY>]=value (values parsed as YAML,
so "true", "60", "[1, 40]" get proper types).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, field_validator


def _find_root() -> Path:
    """Repo root: BAYN_ROOT env, else the source checkout, else the working directory."""
    if env := os.environ.get("BAYN_ROOT"):
        return Path(env).resolve()
    candidate = Path(__file__).resolve().parents[3]
    if (candidate / "config" / "default.yaml").exists():
        return candidate
    return Path.cwd()


ROOT = _find_root()
CONFIG_DIR = ROOT / "config"
DATA_DIR = Path(os.environ.get("BAYN_DATA_DIR", ROOT / "data"))
MODELS_DIR = Path(os.environ.get("BAYN_MODELS_DIR", ROOT / "models"))
ENV_PREFIX = "BAYN_"
_META_KEYS = {"BAYN_ROOT", "BAYN_DATA_DIR", "BAYN_MODELS_DIR", "BAYN_HARDWARE"}

# No telemetry from Hugging Face libraries (models are only downloaded by `bayn models`).
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")


# ---------------------------------------------------------------- models
class BrainFlowConfig(BaseModel):
    board: str = "MUSE_2_BOARD"
    serial_number: str = ""
    mac_address: str = ""
    timeout_s: int = 20
    stale_after_s: float = 3.0


class LSLConfig(BaseModel):
    stream_type: str = "EEG"
    stream_name: str = ""
    resolve_timeout_s: float = 10.0
    autostart_muselsl: bool = True
    stale_after_s: float = 3.0


class EEGConfig(BaseModel):
    source: Literal["sim", "muse", "lsl"] = "muse"
    channels: list[str] = ["TP9", "AF7", "AF8", "TP10"]
    fs: int = 256
    chunk_samples: int = 32
    bandpass_hz: tuple[float, float] = (1.0, 40.0)
    notch_hz: str = "auto"
    buffer_seconds: int = 60
    brainflow: BrainFlowConfig = BrainFlowConfig()
    lsl: LSLConfig = LSLConfig()

    @field_validator("notch_hz", mode="before")
    @classmethod
    def _norm_notch(cls, v: Any) -> str:
        s = "off" if v in (None, False) else str(v).strip().lower()
        if s not in {"auto", "50", "60", "off"}:
            raise ValueError("notch_hz must be auto, 50, 60 or off")
        return s


class AutoGestures(BaseModel):
    enabled: bool = False
    first_after_s: float = 15.0
    every_s: float = 45.0
    sequence: list[str] = ["clench", "triple_blink"]


class SimulatorConfig(BaseModel):
    seed: int | None = None
    mains_hz: float = 50.0
    natural_blinks: bool = True
    natural_emg: bool = True
    auto_gestures: AutoGestures = AutoGestures()


class ClenchConfig(BaseModel):
    band_hz: tuple[float, float] = (55.0, 100.0)
    window_s: float = 0.25
    hold_s: float = 1.0
    threshold_uv: float | None = None
    baseline_k: float = 4.0
    min_threshold_uv: float = 8.0


class BlinkConfig(BaseModel):
    count: int = 3                  # blinks in the trigger (user: 3, 2026-11-10)
    band_hz: tuple[float, float] = (0.3, 10.0)
    threshold_uv: float | None = None
    default_threshold_uv: float = 90.0
    min_gap_s: float = 0.2
    max_gap_s: float = 0.8
    max_width_s: float = 0.6
    settle_s: float = 0.45
    method: Literal["auto", "rule", "ml"] = "auto"   # auto: classifier if it beat the rule held-out; ml: always
    ml_threshold: float = 0.5                          # P(triple blink) needed to fire


class SeizureConfig(BaseModel):
    enabled: bool = False            # the MOCK seizure model (docs/SEIZURE_MODEL.md); `bayn demo` turns it on
    threshold: float | None = None   # None = the model's own (0.9)


class TrainingConfig(BaseModel):
    auto_inject_sim: bool = True     # simulator: perform each prompted gesture automatically
    scale: float = 1.0               # < 1 shortens the protocols (tests)


class CalibrationConfig(BaseModel):
    rest_s: float = 15.0
    reps: int = 3
    ready_s: float = 2.0
    do_s: float = 3.0
    release_s: float = 3.0
    auto_inject_sim: bool = True


class EventsConfig(BaseModel):
    gesture: Literal["clench", "triple_blink", "both", "manual"] = "clench"
    cooldown_s: float = 15.0
    manual_debounce_s: float = 2.0
    pre_s: float = 10.0
    post_s: float = 30.0
    profile: str = "default"
    clench: ClenchConfig = ClenchConfig()
    triple_blink: BlinkConfig = BlinkConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    training: TrainingConfig = TrainingConfig()
    seizure: SeizureConfig = SeizureConfig()


class AudioConfig(BaseModel):
    source: Literal["mic", "file"] = "mic"
    device: str | int | None = None
    file: str = "samples/audio/demo.wav"
    loop: bool = True
    chunk_ms: int = 100


class TranscriptionConfig(BaseModel):
    model: str = "auto"
    device: Literal["auto", "cuda", "cpu"] = "auto"
    compute_type: str = "auto"
    language: Literal["mixed", "auto", "en", "ar"] = "mixed"
    mixed_strategy: Literal["per_utterance", "arabic_lean"] = "per_utterance"
    english_confidence: float = 0.9
    initial_prompt: str = "none"
    beam_size: int | Literal["auto"] = "auto"
    cpu_threads: int | Literal["auto"] = "auto"
    single_encode: bool | Literal["auto"] = "auto"   # one encoder pass per utterance in mixed mode
                                                     # (engine_single.py); auto = on when Whisper runs on the CPU
    vad_threshold: float = 0.5
    min_silence_s: float = 0.6
    max_utterance_s: float = 15.0
    partials: bool | Literal["auto"] = "auto"
    partial_every_s: float = 1.0
    event_grace_s: float = 20.0
    event_tail_s: float = 4.0


class BusConfig(BaseModel):
    bind_host: str = "127.0.0.1"
    eeg_host: str = "127.0.0.1"
    eeg_pub_port: int = 5555
    eeg_cmd_port: int = 5556
    events_host: str = "127.0.0.1"
    events_pub_port: int = 5560
    events_cmd_port: int = 5561
    audio_host: str = "127.0.0.1"
    audio_pub_port: int = 5570
    stt_host: str = "127.0.0.1"
    stt_pub_port: int = 5575
    stt_cmd_port: int = 5576
    sum_host: str = "127.0.0.1"
    sum_pub_port: int = 5580
    sum_cmd_port: int = 5581
    showcase_host: str = "127.0.0.1"
    showcase_pub_port: int = 5590


class SummaryConfig(BaseModel):
    model: str = "auto"
    host: str = "http://127.0.0.1:11434"
    language: Literal["auto", "en", "ar", "both"] = "auto"
    temperature: float = 0.2
    max_key_points: int = 4
    keep_alive: str = "60m"
    timeout_s: float = 180.0


class UIConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8080
    plot_seconds: float = 5.0
    open_browser: bool = True


class LoggingConfig(BaseModel):
    level: str = "INFO"


class AppConfig(BaseModel):
    eeg: EEGConfig = EEGConfig()
    simulator: SimulatorConfig = SimulatorConfig()
    events: EventsConfig = EventsConfig()
    audio: AudioConfig = AudioConfig()
    transcription: TranscriptionConfig = TranscriptionConfig()
    summary: SummaryConfig = SummaryConfig()
    bus: BusConfig = BusConfig()
    ui: UIConfig = UIConfig()
    logging: LoggingConfig = LoggingConfig()


# ---------------------------------------------------------------- loading
def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (KEY=VALUE lines). Real env vars always win."""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _env_overrides(environ: dict[str, str]) -> dict:
    """Turn BAYN_A__B=x into {"a": {"b": x}}."""
    out: dict = {}
    for key, raw in environ.items():
        if not key.startswith(ENV_PREFIX) or key in _META_KEYS or "__" not in key:
            continue
        path = key[len(ENV_PREFIX):].lower().split("__")
        node = out
        for part in path[:-1]:
            node = node.setdefault(part, {})
        try:
            node[path[-1]] = yaml.safe_load(raw) if raw != "" else ""
        except yaml.YAMLError:
            node[path[-1]] = raw
    return out


def load_config(
    path: Path | None = None,
    environ: dict[str, str] | None = None,
    use_dotenv: bool = True,
) -> AppConfig:
    """Load the merged configuration. `environ` is injectable for tests."""
    if environ is None:
        if use_dotenv:
            load_dotenv()
        environ = dict(os.environ)
    merged = _deep_merge(_read_yaml(path or CONFIG_DIR / "default.yaml"), _read_yaml(CONFIG_DIR / "local.yaml"))
    merged = _deep_merge(merged, _env_overrides(environ))
    return AppConfig.model_validate(merged)
