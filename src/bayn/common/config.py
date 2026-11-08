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
ENV_PREFIX = "BAYN_"
_META_KEYS = {"BAYN_ROOT", "BAYN_DATA_DIR"}


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
    sequence: list[str] = ["clench", "double_blink"]


class SimulatorConfig(BaseModel):
    seed: int | None = None
    mains_hz: float = 50.0
    natural_blinks: bool = True
    natural_emg: bool = True
    auto_gestures: AutoGestures = AutoGestures()


class BusConfig(BaseModel):
    bind_host: str = "127.0.0.1"
    eeg_host: str = "127.0.0.1"
    eeg_pub_port: int = 5555
    eeg_cmd_port: int = 5556


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
