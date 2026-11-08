# Progress log

## Plan (approved 2026-11-08)
1. Repo, simulated EEG, UI skeleton
2. Connect the real Muse 2 (BrainFlow `MUSE_2_BOARD`, muse-lsl fallback) — becomes the default source
3. Gesture trigger + events (`Detector` interface, clench / double blink, calibration, cooldown, EEG window −10 s/+30 s, manual button + hotkey)
4. Whisper live transcription (faster-whisper, live mic, en / ar / mixed)
5. Ollama summaries (qwen2.5:3b default), transcript → summary view in the UI
6. Tune gesture triggering on the real headset + guided onboarding (user request, "fun" feature): Muse-app-style
   walkthrough ("blink twice", "clench your jaw", "close your eyes and watch alpha rise") that doubles as calibration
7. Docker Compose + docs polish
8. Full end-to-end test run with simulated and real data

## Phase 1 — repo, simulated EEG, UI skeleton ✅
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

## Phase 2 — real Muse 2 ✅
Built:
- `BrainFlowSource` (default `eeg.source: muse`): connects in a worker thread, auto-reconnects when data stops for 3 s,
  reports progress ("searching for Muse-3EE1...") to the dashboard. Any BrainFlow board via `eeg.brainflow.board`.
- `LSLSource` (fallback): reads `muselsl stream`, picks channels by label, converts LSL clock to unix time.
  `run --source lsl` starts and restarts `muselsl stream` automatically.
- `SampleClock`: regular timeline from jittery device timestamps, re-anchors after gaps; bridge resets filters after gaps.
- `bayn check` (10 s connection + quality report), Bluetooth/Muse scan in `doctor`, pairing + troubleshooting in README.
- `bayn.cmd` wrapper: Windows blocks .ps1 scripts by default; the wrapper bypasses that per call.
- Brainwave panel (user request): delta/theta/alpha/beta/gamma share of power per sensor, labelled like "Alpha (AF7)",
  computed in the bridge at 2 Hz (`eeg.bands` topic, also reusable as EEG stats for Phase 5 summaries).
- Tests reorganised by level with pytest markers: 32 unit, 6 integration, 2 system (real processes end to end),
  2 hardware (real Muse, opt-in via BAYN_HARDWARE=1). Full plan, results and defect log: docs/TESTING.md.

Hands-on results (dev PC, Muse-3EE1, 2026-11-08):
- BrainFlow: connects in ~6-12 s, 256.0 Hz, no drops, all channels "good" (13-30 µV). Mains detected: **60 Hz**.
- muse-lsl: same quality, 256.0 Hz.
- Full launcher with the real headset: "searching" -> connected -> live plot within ~12 s.

Decisions / gotchas:
- Dev PC built-in Bluetooth (QCA61x4A, no antennas) could not see the Muse → USB dongle (Actions ATS2851). On Windows 10
  it showed Code 31 until its vendor driver was installed; the built-in adapter is disabled (Code 22). Laptop: run `doctor`.
- muselsl 2.5.3 crashes with bitstring 5 → pinned bitstring 4.4.0.
- muse-lsl stamps samples with unix time (not the LSL clock) → clock style detected per stream.
- Hardware tests: BrainFlow 256.0 Hz, latency median 20 ms; muse-lsl 256.0 Hz, latency median −23 ms.
- muse-lsl pulls pandas/scikit-learn/matplotlib → kept in requirements/muselsl.txt (top-level pins, cross-platform).

Next: Phase 3 (gesture trigger + events).
