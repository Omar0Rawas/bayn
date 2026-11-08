# Bayn — notes for Claude sessions

Local hackathon demo: Muse 2 EEG -> on-command gesture trigger -> Whisper live transcript -> Ollama summary.
Prototype, NOT a medical device. Current status and next steps: docs/PROGRESS.md (update it at the end of every phase).

## Working agreement
- Work in the agreed phases (see docs/PROGRESS.md). **Stop after each phase** so the user can test; commit only after they confirm.
- Developed on a desktop PC (Windows 10, RTX 2080 Ti), demoed on a different laptop with unknown hardware:
  **never hardcode hardware**. GPU/CPU, Whisper model, mains frequency are auto-detected.
- After Phase 2 the **real Muse 2 is the default EEG source**; simulated stays available via `--demo` / `--source sim`.
  Don't run anything that needs the headset unless the user says it's connected.
- 100% offline after setup: no cloud APIs, no telemetry, no CDN assets in the UI.
- Demo mode uses the live microphone (WAV file source is optional).
- Languages: English, Arabic, and mixed (per-utterance language detection limited to {ar, en}).

## Architecture
- Separate processes talk over ZeroMQ (TCP) — msgpack frames `[topic, payload]`, numpy arrays as raw bytes (src/bayn/common/bus.py).
  Topics/messages: src/bayn/common/schemas.py.
- Do NOT use `zmq.asyncio` in the FastAPI server: it fails on Windows' Proactor loop. Use `ThreadedSubscriber`.
- EEG bridge publishes both `eeg.raw` (needed for clench/EMG detection at 55-100 Hz) and `eeg.filt` (1-40 Hz + notch).
- Config: config/default.yaml <- config/local.yaml <- .env <- env vars `BAYN_SECTION__KEY` (src/bayn/common/config.py).

## Hardware notes
- User's headset: Muse-3EE1. Mains on the dev PC's grid is 60 Hz (auto-detected; do not hardcode).
- Dev PC uses a USB Bluetooth dongle; its built-in adapter is disabled. Only one BLE adapter may be active.
- muselsl needs bitstring<5.

## Commands (Windows)
- `.\bayn setup | demo | run | check | test | doctor` (bayn.cmd wraps scripts/bayn.ps1 with ExecutionPolicy Bypass,
  because Windows blocks .ps1 scripts by default)
- Tests: `.\.venv\Scripts\python.exe -m pytest` (no hardware needed); levels via `-m unit|integration|system`;
  real headset: `$env:BAYN_HARDWARE=1; ... -m hardware -v -s`.
- Keep docs/TESTING.md current every phase: per-phase results, traceability matrix, defect log (D#), env issues (E#).
- Python 3.12+ (numpy 2.5 dropped 3.11). Deps pinned in requirements/*.txt; pyproject has loose ranges.
