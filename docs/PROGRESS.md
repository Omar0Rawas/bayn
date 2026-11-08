# Progress log

## Plan (approved 2026-11-08)
1. Repo, simulated EEG, UI skeleton
2. Connect the real Muse 2 (BrainFlow `MUSE_2_BOARD`, muse-lsl fallback) — becomes the default source
3. Gesture trigger + events (`Detector` interface, clench / double blink, calibration, cooldown, EEG window −10 s/+30 s, manual button + hotkey)
4. Whisper live transcription (faster-whisper, live mic, en / ar / mixed)
5. Ollama summaries (qwen2.5:3b default), transcript → summary view in the UI
6. Tune gesture triggering on the real headset
7. Docker Compose + docs polish
8. Full end-to-end test run with simulated and real data

## Phase 1 — repo, simulated EEG, UI skeleton ✅ (awaiting user test)
Built:
- Config system (YAML + env overrides), ZeroMQ bus, typed messages.
- `SimulatedEEG`: pink-noise background, alpha/theta, mains hum, DC drift, random natural blinks and weak EMG bursts as
  distractors; injectable `clench`, `double_blink`, `blink`. Deterministic with a seed.
- Streaming 1–40 Hz bandpass + notch (`auto` = 50 and 60 Hz), ring buffer, MAD-based channel quality, mains detection.
- EEG bridge process; FastAPI dashboard with live canvas plot, quality chips, simulator buttons, placeholders for later phases.
- `python -m bayn.run --demo` launcher, `bayn.doctor` machine check, `scripts/bayn.ps1`, Makefile, `scripts/claude-session.ps1`.
- 23 tests, all passing.

Decisions / gotchas:
- pyzmq asyncio sockets fail on Windows' default event loop → dashboard reads ZeroMQ in a thread (`ThreadedSubscriber`).
- uv not installed on the dev PC → plain venv + pinned requirements.
- Python ≥ 3.12 required.

Next: Phase 2 (real Muse 2) once the headset is available.
