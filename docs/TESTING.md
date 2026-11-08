# Testing & Validation

This document describes how Bayn is verified: which tests exist at each level (unit, integration, system, hardware-in-the-loop, manual acceptance), what each phase was tested against, the results, and the defects that testing found.

> [!IMPORTANT]
> Bayn is a hackathon prototype, **not a medical device**. "Validation" here means *engineering* verification and validation: does the software do what the design says, on real hardware, reliably enough for a live demo? **No clinical validation has been done, and none is claimed.**

**Status:** Phases 1–2 tested and passing. Phases 3–8 have test plans with acceptance criteria, filled in as each phase lands.
Last updated: 2026-11-08.

---

## 1. How to run the tests

| What | Windows (PowerShell) | macOS / Linux |
|---|---|---|
| Everything that needs no hardware | `.\bayn test` | `make test` |
| One level only | `.\.venv\Scripts\python.exe -m pytest -m unit` (or `integration`, `system`) | `.venv/bin/pytest -m unit` |
| Real headset (hardware-in-the-loop) | `$env:BAYN_HARDWARE=1; .\.venv\Scripts\python.exe -m pytest -m hardware -v -s` | `BAYN_HARDWARE=1 .venv/bin/pytest -m hardware -v -s` |
| Quick headset check (no pytest) | `.\bayn check` | `make check` |
| Machine readiness | `.\bayn doctor` | `make doctor` |

Hardware tests are skipped unless `BAYN_HARDWARE=1`. The LSL hardware test also needs `muselsl stream` running in another terminal.

---

## 2. Test levels

| Level | Purpose | Location | Count | Runtime | Runs |
|---|---|---|---|---|---|
| **Unit** | One function or class in isolation. No sockets, threads or devices. | `tests/test_filters.py`, `test_buffer_quality.py`, `test_simulator.py`, `test_bands.py`, parts of `test_bus_config.py` and `test_sources.py` | 32 | ~2 s | Every `test` run |
| **Integration** | Several components together across real boundaries: TCP sockets, threads, asyncio, BrainFlow's native library (synthetic board), LSL networking. | `test_bus_config.py`, `test_sources.py` | 6 | ~5 s | Every `test` run |
| **System** | The real `bridge` and `dashboard` *processes*, end to end, driven like a browser would (WebSocket + REST). Simulated EEG. | `tests/test_system.py` | 2 | ~10 s | Every `test` run |
| **Hardware-in-the-loop** | The real Muse 2 over Bluetooth, through both acquisition paths. | `tests/test_hardware.py` | 2 | ~45 s | On demand, with headset |
| **Manual acceptance** | A human checks the dashboard, ergonomics and demo flow against a checklist. | §4 (per phase), §6 | – | minutes | End of each phase |

### 2.1 Unit tests (32)

| Area | Test | Verifies |
|---|---|---|
| Filters | `test_passband_kept` | 5/10/20 Hz pass with gain 0.85–1.1 |
| | `test_mains_removed_both_50_and_60_in_auto` | `auto` notch attenuates **both** 50 and 60 Hz to < 3 % |
| | `test_dc_offset_and_drift_removed` | an 800 µV electrode offset causes no start-up transient (< 5 µV) |
| | `test_streaming_matches_single_pass` | chunked streaming filtering = one-shot filtering |
| | `test_notch_settings` | `auto/50/60/off`; notches above Nyquist are dropped |
| Buffer | `test_ringbuffer_wraps_in_order` | wrap-around keeps chronological order |
| | `test_ringbuffer_window_and_oversized_append` | time-window queries; appending more than capacity |
| Quality | `test_quality_levels` | flat → bad, normal → good, noisy → ok, very noisy → bad |
| | `test_quality_ignores_single_blink` | a 150 µV blink does not mark a channel bad (MAD statistic) |
| | `test_mains_detection_50_vs_60` | detects 50 Hz, 60 Hz, and reports *none* when absent |
| Simulator | `test_shape_dtype_and_plausible_amplitude` | 4 × N float32, filtered amplitude 3–40 µV (scalp-EEG range) |
| | `test_chunked_equals_continuous` | same seed → identical signal regardless of chunk size |
| | `test_alpha_peak_present` | spectral peak at 9–11 Hz (alpha rhythm) |
| | `test_clench_raises_emg_on_temporal_channels` | clench: 55–100 Hz power > 5× baseline on TP9/TP10, temporal > frontal |
| | `test_double_blink_gives_two_frontal_peaks` | exactly 2 peaks on AF7 and AF8, 0.3–0.55 s apart |
| | `test_unknown_gesture_rejected` | invalid gesture name raises |
| Brainwave bands | `test_pure_tone_lands_in_its_band[delta/theta/alpha/beta/gamma]` | a pure tone at 2.5/6/10/20/38 Hz puts > 90 % of the power in its own band |
| | `test_relative_sums_to_one_and_dc_ignored` | relative powers sum to 1; a 500 µV electrode offset doesn't count |
| | `test_mains_hum_not_counted_as_gamma` | strong 60 Hz hum adds < 5 % to gamma (gamma stops at 44 Hz) |
| | `test_simulated_eeg_alpha_on_temporal_and_clench_raises_gamma` | a jaw clench raises TP9/TP10 gamma > 3× |
| | `test_bands_message_roundtrips_over_bus` | band payload survives serialization, shape 4 × 5 |
| Bus / config | `test_pack_roundtrip_ndarray`, `test_schema_roundtrip` | msgpack + NumPy and message schemas survive serialization |
| | `test_env_overrides_typed`, `test_bad_notch_rejected` | `BAYN_*` overrides get the right types; invalid values rejected |
| Sources | `test_sample_clock_regularises_jitter` | ±20 ms packet jitter → perfectly regular timeline |
| | `test_sample_clock_reanchors_after_gap` | a 4 s dropout re-anchors the clock |
| | `test_unknown_brainflow_board_rejected` | invalid board name raises |

### 2.2 Integration tests (6)

| Test | Components | Verifies |
|---|---|---|
| `test_pubsub_over_tcp` | Publisher ↔ Subscriber (ZeroMQ TCP) | topic filtering, payload integrity |
| `test_threaded_subscriber_feeds_asyncio_default_loop` | ZeroMQ thread → asyncio queue | **regression for defect D2** (dashboard got no data on Windows) |
| `test_simulated_source_paces_in_real_time_and_records_ground_truth` | simulator + real-time pacing | 8 chunks take 0.8–1.5 s, contiguous timestamps, injection ground truth recorded |
| `test_brainflow_source_with_synthetic_board` | our BrainFlow source + BrainFlow native library | worker thread, 4 channels, contiguous unix timeline, no hardware |
| `test_lsl_source_selects_channels_by_label[lsl-clock]` | our LSL source + a real LSL outlet | channels picked by label in config order; LSL-clock → unix |
| `test_lsl_source_selects_channels_by_label[unix-clock-like-muselsl]` | same, muse-lsl timestamp style | **regression for defect D5** |

### 2.3 System tests (2)

| Test | Verifies |
|---|---|
| `test_end_to_end_simulated_pipeline` | Starts the bridge and dashboard as separate processes on free ports, connects over WebSocket like the browser, and checks: `hello` handshake; ~256 samples/s arrive in 4 channels (6–11 s of data in 9 s); per-channel quality reported for TP9/AF7/AF8/TP10; **mains detected as 60 Hz when the simulator injects 60 Hz** (detected, not assumed); a REST `double_blink` injection makes the full round trip back to the browser; brainwave band updates arrive at ~2 Hz (5 bands, relative powers sum to 1). |
| `test_bad_gesture_rejected` | The REST API rejects an invalid gesture with HTTP 400. |

### 2.4 Hardware-in-the-loop tests (2)

| Test | Pass criteria |
|---|---|
| `test_muse_via_brainflow` | connects within 60 s; effective rate 256 ± 3 Hz; channel order TP9, AF7, AF8, TP10; median latency (receive time − sample timestamp) within ±0.5 s; ≥ 3 of 4 channels not `bad` |
| `test_muse_via_lsl` | same criteria through `muselsl stream` |

---

## 3. Test environment

| | Dev PC (tested) | Demo laptop |
|---|---|---|
| OS | Windows 10 Pro 19045 | TBD: run `doctor` + hardware tests before the event |
| Python | 3.13.7 | ≥ 3.12 required |
| GPU | NVIDIA RTX 2080 Ti 11 GB | unknown, auto-detected |
| Bluetooth | USB dongle (Actions ATS2851, vendor driver); built-in QCA61x4A disabled | TBD |
| Headset | Muse 2 `Muse-3EE1` | same |
| Mains | 60 Hz (measured) | measured automatically |

---

## 4. Phase-by-phase verification

### Phase 1: Repo, simulated EEG, dashboard skeleton ✅

**Goal:** a runnable skeleton with no hardware: realistic simulated EEG, filtering, buffer, bus, live dashboard.

| Activity | Level | Result |
|---|---|---|
| Filter, buffer, quality, simulator, config, bus unit tests | Unit | ✅ pass |
| Pub/sub + asyncio bridge | Integration | ✅ pass (after fixing D2) |
| Launcher → bridge → dashboard → WebSocket, with injection | System | ✅ ~8 EEG msgs/s, all channels good, injection round trip, 50 Hz mains detected |
| Dashboard visual check (plot scrolls, markers, quality chips, reconnect) | Manual | ⏳ user |

**Acceptance criteria:** demo starts with one command, no hardware ✅ · plot updates in real time ✅ · injected gestures visible ✅ · tests pass without hardware ✅

### Phase 2: Real Muse 2 ✅

**Goal:** the real headset as the default source through BrainFlow, with muse-lsl as a fallback, plus robust connection handling.

| Activity | Level | Result |
|---|---|---|
| SampleClock, board validation | Unit | ✅ pass |
| BrainFlow source with the synthetic board; LSL source with a fake outlet (both clock styles) | Integration | ✅ pass |
| BLE scan → `Muse-3EE1` visible | Hardware | ✅ at −71 to −77 dBm via the dongle (not visible on the built-in adapter, see E1) |
| `check`: BrainFlow, 10 s | Hardware | ✅ 256.0 Hz, 0 drops, TP9 13.1 / AF7 17.5 / AF8 29.3 / TP10 13.0 µV, all `good`, mains **60 Hz** |
| `check`: muse-lsl, 10 s | Hardware | ✅ 256.0 Hz, all `good` (17–31 µV), mains 60 Hz |
| `test_muse_via_brainflow` | Hardware | ✅ 256.0 Hz, latency median 20 ms / max 49 ms |
| `test_muse_via_lsl` | Hardware | ✅ 256.0 Hz, latency median −23 ms / max 1 ms |
| Full launcher, default source = Muse | System (manual run) | ✅ "searching for Muse 2…" → connected → live data in ~12 s; inject correctly answers "not supported" |
| Brainwave band panel (δ θ α β γ per sensor, e.g. "Alpha (AF7)") on the live headset | Hardware (manual run) | ✅ 16 updates in 8 s. Eyes-open rest, mean share: TP9 δ 41 / θ 23 / α 13 / β 15 / γ 7 %; AF7 β 50 / γ 22 %, AF8 β 53 / γ 28 % (forehead muscle tension, expected while sitting at a desk) |
| Eyes-closed alpha increase on TP9/TP10 | Manual | ⏳ user |
| Reconnect after headset power-off | Manual | ⏳ user: switch the headset off for 10 s while running, then back on |
| Laptop run (`doctor`, `check`, hardware tests) | Hardware | ⏳ before the event |

**Acceptance criteria:** real Muse is the default source ✅ · both acquisition paths deliver 256 Hz with correct channel order and unix timestamps ✅ · signal quality visible ✅ · dashboard shows connection progress instead of crashing ✅ · reconnects automatically ⏳ (implemented, manual test pending)

### Phase 3: Gesture trigger + events (planned)

| Planned activity | Level | Acceptance criterion |
|---|---|---|
| `Detector` interface contract: every detector (clench, double blink, manual, seizure stub) passes one shared test | Unit | all pass |
| Clench and double-blink detectors on simulated data with injected gestures (known ground truth) | Unit | ≥ 95 % of injected gestures detected; onset error ≤ 0.5 s |
| False triggers on simulated distractors (natural blinks, short EMG bursts), 10 min of signal | Unit | ≤ 1 false trigger per 10 min |
| Cooldown | Unit | a second gesture inside the cooldown never fires |
| Calibration from rest + 3 gestures sets thresholds between the two distributions | Unit | thresholds saved per user and reloaded |
| Event window saved −10 s / +30 s | Integration | saved file holds exactly 40 s ± 1 sample, correct timestamps |
| Gesture → event → UI (visual + audio feedback), manual button + hotkey | System | event visible in the timeline < 1.5 s after the gesture |

### Phase 4: Whisper live transcription (planned)

| Planned activity | Level | Acceptance criterion |
|---|---|---|
| Audio chunking / VAD, segment timestamping | Unit | segments monotonic, timestamps in unix time |
| Transcribe known English and Arabic WAV clips | Integration | key phrases recognised; per-segment language correct for ar/en |
| GPU and CPU auto-selection | Integration | runs on CUDA when present, falls back to CPU without error |
| Live mic → dashboard transcript | System | segment appears ≤ 3 s after speech ends (GPU), ≤ 6 s (CPU) |
| Offline | System | works with the network disabled after setup |

### Phase 5: Ollama summaries (planned)

| Planned activity | Level | Acceptance criterion |
|---|---|---|
| Prompt builder: transcript window + EEG stats, en / ar / both | Unit | stable prompt structure, window boundaries correct |
| Ollama unavailable / model missing | Integration | clear error in the UI, pipeline keeps running |
| Scripted scenarios: summary covers the facts in the transcript and invents none | Manual | reviewed on ≥ 5 scenarios in English and Arabic |
| Event → summary streamed into the UI | System | first tokens < 5 s after the event window closes (GPU) |

### Phase 6: Gesture tuning on the real headset (planned)

Protocol, per tester, headset on, after calibration:

| Trial | Acceptance criterion |
|---|---|
| 20 deliberate jaw clenches, 20 deliberate double blinks | ≥ 18 / 20 detected for the chosen gesture |
| 10 min normal activity: talking, chewing, normal blinking, head movement | ≤ 1 false trigger |
| Gesture-to-event latency | ≤ 1.5 s |
| Repeat with ≥ 2 people | criteria hold for each, using their own calibration |
| Guided onboarding ("blink twice", "clench", "close your eyes", like the Muse app) | a first-time user completes it in < 2 min without help, and it produces a working calibration |

### Phase 7: Docker + docs (planned)

| Planned activity | Acceptance criterion |
|---|---|
| `docker compose --profile demo up` on a clean machine | dashboard live with simulated EEG, no manual steps |
| Host + container split (native bridge/mic, containers for the rest) | data flows across `host.docker.internal` |
| README quickstart followed literally on Windows and one of macOS/Linux | works without undocumented steps |

### Phase 8: Full end-to-end test run (planned)

| Run | Acceptance criterion |
|---|---|
| Simulated: demo mode end to end (gesture → event → transcript → summary) | 3 consecutive clean runs |
| Real: Muse + live mic, scripted demo script | 3 consecutive clean runs on the **demo laptop**, offline |
| Failure drills: headset off mid-demo, Ollama stopped, mic unplugged | each failure is visible in the UI and the system recovers or degrades gracefully |

---

## 5. Requirements traceability

| Requirement | Verified by | Status |
|---|---|---|
| 100 % local, no cloud, no telemetry | design review; UI loads no external assets; offline system run planned in Phases 4 and 8 | partial ✅ |
| Easy setup on other machines | `setup` + `doctor`; Phase 7 clean-machine test | partial ✅ |
| Muse 2 via BrainFlow `MUSE_2_BOARD` | `test_muse_via_brainflow`, `check` | ✅ |
| muse-lsl fallback | `test_muse_via_lsl`, LSL integration tests | ✅ |
| Simulated EEG with clench / blink injection | simulator unit tests, system test | ✅ |
| 1–40 Hz filter + notch | filter unit tests | ✅ |
| Notch must not assume a mains frequency | `test_mains_removed_both_50_and_60_in_auto`, mains detection unit + system tests, hardware run (60 Hz) | ✅ |
| Rolling buffer | buffer unit tests | ✅ |
| Local stream publishing | pub/sub integration test, system test | ✅ |
| Signal-quality display | quality unit tests, system test, hardware `check` | ✅ |
| Brainwave bands δ θ α β γ per sensor, Muse-app style (user request) | band unit tests, system test, live headset run | ✅ |
| Guided onboarding / calibration (user request) | Phase 6 plan | ⏳ |
| Gesture trigger, calibration, cooldown, `Detector` interface | Phase 3 plan | ⏳ |
| Whisper live transcription, en / ar / mixed | Phase 4 plan | ⏳ |
| Local LLM summaries | Phase 5 plan | ⏳ |
| Docker Compose | Phase 7 plan | ⏳ |
| One-command demo without hardware | system test (bridge + dashboard); full pipeline in Phase 8 | partial ✅ |

---

## 6. Defects found by testing

| ID | Found by | Symptom | Root cause | Fix | Regression test |
|---|---|---|---|---|---|
| D1 | Unit | Same seed gave different signals for different chunk sizes | Random numbers drawn channel-major | Draw time-major | `test_chunked_equals_continuous` |
| D2 | System (smoke run) | Dashboard showed heartbeats but **no EEG** on Windows | `zmq.asyncio` needs a selector event loop; uvicorn on Windows uses Proactor, and the relay task died silently | ZeroMQ read in a thread → asyncio queue; background-task crashes now logged | `test_threaded_subscriber_feeds_asyncio_default_loop` |
| D3 | Code review | `setup` could abort when the `py` launcher lacks a version | PowerShell 5.1 turns native stderr into a terminating error under `Stop` | `try/catch` around the probe | manual |
| D4 | Hardware | muse-lsl crashed on the first packet | `muselsl` 2.5.3 is incompatible with `bitstring` 5 | Pin `bitstring==4.4.0` | `test_muse_via_lsl` |
| D5 | Hardware | muse-lsl timestamps ~56 years in the future | muse-lsl stamps with unix time; code assumed LSL clock and added an offset | Detect the clock per stream | `test_lsl_source_selects_channels_by_label[unix-clock-like-muselsl]` |
| D6 | Hardware | `ctypes.ArgumentError` printed when stopping during a BLE search | Worker thread still in a blocking connect at interpreter exit | Silent exit when stopping; `stop()` waits for the connect to finish | manual |
| D7 | Hardware (test bug) | BrainFlow latency looked like 7.9 s | Test read the clock after the ~8 s BLE release | Measure latency at receive time | `test_muse_via_brainflow` |
| D8 | Manual (user) | `.\scripts\bayn.ps1` refused to run: "running scripts is disabled on this system" | Windows' default PowerShell execution policy blocks `.ps1` files | `bayn.cmd` wrapper runs it with `-ExecutionPolicy Bypass` for that call only | manual |

### Environment issues (not software defects)

| ID | Issue | Resolution |
|---|---|---|
| E1 | Dev PC's built-in Bluetooth (QCA61x4A Wi-Fi/BT card, no antennas attached) saw 1 BLE device and no Muse | USB Bluetooth dongle |
| E2 | Dongle (Actions ATS2851) showed Device Manager Code 31 on Windows 10 | Vendor driver; built-in adapter disabled (only one active radio). Documented in the README |
| E3 | Muse powers itself off when idle and unconnected, so a test run 25 min later found nothing | Turn it on right before testing; `doctor` shows whether it is advertising |

---

## 7. Known limitations / not yet tested

- Automatic reconnect after a real headset drop-out: implemented and unit-tested via the clock and gap logic, **not yet exercised with the real headset** (manual test pending).
- Signal-quality thresholds are initial values. They get re-tuned with real data in Phase 6.
- Hardware tests have only run on the dev PC. They must be repeated on the demo laptop.
- The dashboard UI has no automated browser tests. It is checked manually.
