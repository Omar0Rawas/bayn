# Testing & Validation

This document describes how Bayn is verified: which tests exist at each level (unit, integration, system, hardware-in-the-loop, manual acceptance), what each phase was tested against, the results, and the defects that testing found.

> [!IMPORTANT]
> Bayn is a hackathon prototype, **not a medical device**. "Validation" here means *engineering* verification and validation: does the software do what the design says, on real hardware, reliably enough for a live demo? **No clinical validation has been done, and none is claimed.**

**Status:** Phases 1–4 tested. On the real headset the jaw-clench trigger works; the rule-based double blink does not yet (finding F1, ML fix planned for Phase 6). Phase 4 transcription is verified with recorded audio on GPU and CPU; the live-mic check is pending with the user. Phases 5–8 have test plans with acceptance criteria, filled in as each phase lands.
Last updated: 2026-11-09.

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
| **Unit** | One function or class in isolation. No sockets, threads, devices or ML models. Detector and event-manager tests stream minutes of simulated EEG through the logic offline, in seconds. The segmenter uses a simple energy VAD stand-in. | `tests/test_filters.py`, `test_buffer_quality.py`, `test_simulator.py`, `test_bands.py`, `test_detectors.py`, `test_events_manager.py`, `test_transcription_unit.py`, `test_transcription_eval.py`, parts of `test_bus_config.py` and `test_sources.py` | 86 | ~25 s | Every `test` run |
| **Integration** | Several components together across real boundaries: TCP sockets, threads, asyncio, BrainFlow's native library (synthetic board), LSL networking, real Silero VAD + Whisper models. | `test_bus_config.py`, `test_sources.py`, `test_transcription_models.py`, `test_transcription_eval.py` | 10 | ~20 s | Every `test` run |
| **System** | The real `bridge`, `events`, `audio`, `transcriber` and `dashboard` *processes*, end to end, driven like a browser would (WebSocket + REST). Simulated EEG, WAV audio. | `tests/test_system.py` | 6 | ~60 s | Every `test` run |
| **Hardware-in-the-loop** | The real Muse 2 over Bluetooth, through both acquisition paths. | `tests/test_hardware.py` | 2 | ~45 s | On demand, with headset |
| **Manual acceptance** | A human checks the dashboard, ergonomics and demo flow against a checklist. | §4 (per phase), §6 | – | minutes | End of each phase |

### 2.1 Unit tests (86)

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
| Detectors: interface | `test_detector_contract[clench/double_blink/manual/seizure_stub]` | every `Detector` (including the seizure stub) honours the same contract: list of `Detection`, `status()` keys, `reset()` |
| | `test_seizure_stub_never_fires`, `test_factory_follows_gesture_setting` | stub is inert; `events.gesture` selects detectors; invalid gesture rejected |
| Detectors: clench | `test_clench_detection_rate_and_onset_accuracy` | 30 injected clenches over 3 seeds (50 and 60 Hz mains): ≥ 95 % detected, 0 false, onset error ≤ 0.5 s |
| | `test_clench_needs_hold_time`, `test_clench_fires_once_per_clench` | bursts shorter than `hold_s` never fire; one clench = one detection |
| Detectors: double blink | `test_double_blink_detection_rate` | 30 injected double blinks over 3 seeds, natural blinks + muscle bursts on: ≥ 90 % detected, 0 false |
| | `test_single_blinks_do_not_fire`, `test_rebound_is_not_a_third_blink` | single blinks ignored; regression for D9 |
| False triggers | `test_no_false_triggers_in_10_minutes_of_distractors[clench/double_blink]` | 10 min of natural blinks and short muscle bursts, no deliberate gesture: ≤ 1 trigger |
| | `test_gestures_dont_cross_trigger` | clenches never fire the blink detector and vice versa |
| Calibration | `test_calibration_separates_rest_and_gesture[clench/double_blink]` | rest level < threshold < gesture level, quality `good` |
| | `test_calibration_fails_without_gesture`, `test_profile_threshold_applied` | no gesture → `failed`/`weak`, nothing saved; profile thresholds load |
| Event manager | `test_gesture_creates_event_and_saves_exact_window` | event within 0.5 s of the true onset; saved window = 40 s ± 1 sample (−10/+30 s); `.npz`, `.csv` and `meta.json` written; before/after stats present; SQLite row `saved` |
| | `test_cooldown_suppresses_second_gesture` | a second clench 6 s later (cooldown 15 s) is ignored and counted |
| | `test_manual_trigger_debounce_and_cooldown_interaction` | button/hotkey debounced at 2 s; manual trigger is not blocked by the gesture cooldown |
| | `test_manual_event_finalizes_by_wall_clock_without_eeg` | an event without EEG still completes (marked partial) |
| | `test_event_right_after_connect_is_marked_partial` | a short pre-window is flagged as partial |
| | `test_manual_only_mode_ignores_gestures`, `test_set_gesture_at_runtime` | gesture choice works, including switching at runtime |
| | `test_guided_calibration_end_to_end[clench/double_blink]` | rest → 3 prompted reps (simulator performs them) → thresholds saved to the profile → a restarted manager loads them; no events fire while calibrating |
| | `test_calibration_cancel`, `test_calibration_needs_a_gesture` | cancel works; manual-only mode can't calibrate |
| | `test_store_rejects_path_traversal` | event IDs can't escape the data folder |
| Transcription | `test_utterances_split_on_silence_with_correct_timestamps`, `test_short_pause_does_not_split`, `test_long_monologue_is_cut` | VAD segmentation: boundaries, 0.2 s padding, unix start times, forced cut at a pause, live partials |
| | `test_silence_is_trimmed_and_processed_until_advances`, `test_processed_until_waits_for_speech_in_progress`, `test_gap_resets_and_flush_finishes_speech` | buffer stays small in silence; event windows wait for speech still in progress; stream gaps and end of stream |
| | `test_wav_roundtrip_and_resampling`, `test_stitch_and_file_source`, `test_audio_chunk_int16_roundtrip_and_levels` | WAV I/O, 44.1 kHz stereo → 16 kHz mono, looping file source with contiguous timestamps, int16 bus format, dBFS levels |
| | `test_mixed_mode_only_picks_arabic_or_english`, `test_hallucination_filter`, `test_model_auto_choice` | language choice per mode; known Whisper hallucinations dropped; GPU → large-v3-turbo, CPU → small |
| | `test_store_window_and_format` | transcript time-window query and relative-time formatting |
| Transcription evaluation | `test_normalization_ignores_diacritics_hamza_and_punctuation`, `test_wer_cer_basics`, `test_script_mix_shows_code_switching` | WER/CER metric and Arabic normalisation; share of Latin vs Arabic script |
| | `test_scores_of_the_live_user_test_are_reproduced[english/arabic/mixed]` | the live-mic scores in this document are recomputed exactly from the stored scenario |
| | `test_arabic_lean_keeps_mixed_utterances_in_arabic`, `test_initial_prompt_modes` | optional code-switching strategies (off by default) |
| | `test_event_transcript_finishes_the_sentence_in_progress` | regression for D15 |
| | `test_classroom_scenarios_are_complete` | scenario fixture is well-formed (also used for summaries) |
| Bus / config | `test_pack_roundtrip_ndarray`, `test_schema_roundtrip` | msgpack + NumPy and message schemas survive serialization |
| | `test_env_overrides_typed`, `test_bad_notch_rejected` | `BAYN_*` overrides get the right types; invalid values rejected |
| Sources | `test_sample_clock_regularises_jitter` | ±20 ms packet jitter → perfectly regular timeline |
| | `test_sample_clock_reanchors_after_gap` | a 4 s dropout re-anchors the clock |
| | `test_unknown_brainflow_board_rejected` | invalid board name raises |

### 2.2 Integration tests (10)

| Test | Components | Verifies |
|---|---|---|
| `test_pubsub_over_tcp` | Publisher ↔ Subscriber (ZeroMQ TCP) | topic filtering, payload integrity |
| `test_threaded_subscriber_feeds_asyncio_default_loop` | ZeroMQ thread → asyncio queue | **regression for defect D2** (dashboard got no data on Windows) |
| `test_simulated_source_paces_in_real_time_and_records_ground_truth` | simulator + real-time pacing | 8 chunks take 0.8–1.5 s, contiguous timestamps, injection ground truth recorded |
| `test_brainflow_source_with_synthetic_board` | our BrainFlow source + BrainFlow native library | worker thread, 4 channels, contiguous unix timeline, no hardware |
| `test_lsl_source_selects_channels_by_label[lsl-clock]` | our LSL source + a real LSL outlet | channels picked by label in config order; LSL-clock → unix |
| `test_lsl_source_selects_channels_by_label[unix-clock-like-muselsl]` | same, muse-lsl timestamp style | **regression for defect D5** |
| `test_english_clip_transcribed` | faster-whisper (tiny, CPU) + our engine | "write down the time … episode" recognised, language `en` |
| `test_arabic_clip_detected_as_arabic` | same, Arabic clip, `mixed` mode | language `ar`, Arabic script in the output |
| `test_silero_vad_segments_the_mixed_demo` | Silero VAD + segmenter + Whisper on the 55 s demo | 7–20 utterances; both `en` and `ar` detected |

| `test_evaluation_tool_scores_strategies_on_a_clip` | `bayn eval-stt` logic: VAD + Whisper + metrics | every requested strategy scored on the same utterances |

The four Whisper tests are skipped if the `tiny` model isn't downloaded (`bayn models`).

### 2.3 System tests (6)

| Test | Verifies |
|---|---|
| `test_end_to_end_simulated_pipeline` | Starts the bridge and dashboard as separate processes on free ports, connects over WebSocket like the browser, and checks: `hello` handshake; ~256 samples/s arrive in 4 channels (6–11 s of data in 9 s); per-channel quality reported for TP9/AF7/AF8/TP10; **mains detected as 60 Hz when the simulator injects 60 Hz** (detected, not assumed); a REST `double_blink` injection makes the full round trip back to the browser; brainwave band updates arrive at ~2 Hz (5 bands, relative powers sum to 1). |
| `test_gesture_event_saved_and_downloadable` | Bridge + events + dashboard processes: a clench injected through the REST API fires an `event` in the browser stream < 2.5 s later (1 s hold included); `event_done` arrives with exactly the configured window (−2/+3 s in the test); the CSV downloads over HTTP; `/api/events` lists it. |
| `test_manual_trigger_and_hotkey` | `POST /api/events/trigger {source: hotkey}` → a `manual` event with source `hotkey` reaches the browser. |
| `test_guided_calibration_with_simulator` | `POST /api/events/calibration` → prompts stream to the browser; the simulator performs each prompted clench; calibration ends `done` with quality `good`. |
| `test_live_transcript_and_event_transcript` | All 5 processes. The English test clip plays in a loop as the "microphone", with the tiny model on CPU. A transcript segment reaches the browser (`en`, contains "time", unix timestamps, latency < 5 s). A manual event fired during speech gets `transcript_window` with ≥ 1 segment. `transcript.txt` downloads. The event row in `/api/events` records the transcript. Status reports model/device and `audio_ok`. |
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
| Microphone | TC-Helicon GoXLR Mini "Chat Mic" (default input) | TBD: `bayn mics` |
| Whisper | large-v3-turbo on CUDA (cuDNN 9, CTranslate2 4.8.5); small + tiny also on disk | auto: GPU model if NVIDIA, else small |
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

### Phase 3: Gesture trigger + events ✅ (simulated, clench on real headset) · ❌ double blink on real headset (F1)

**Goal:** a deliberate gesture (or button/hotkey) fires a timestamped event with visual and audio feedback. The EEG window −10/+30 s is saved, the pipeline is notified, and everything goes through a `Detector` interface a seizure model can plug into.

**Simulated-data evaluation** (development sweep, then locked into the unit tests). Gestures injected at known times into simulated EEG with natural blinks and short muscle bursts as distractors:

| Detector | Detected | False triggers | Signal time | Onset error |
|---|---|---|---|---|
| Jaw clench (55–100 Hz RMS, hold 1 s, adaptive threshold) | **40 / 40** | **0** | 20.7 min | +0.11 s (max +0.13 s) |
| Double blink, first version (0.5–10 Hz, abs. value, 110 µV) | 13 / 40 | 0 | 20.7 min | – |
| Double blink, final (0.3–10 Hz, signed peaks, 90 µV) | **58 / 60 (97 %)** | **0** | 62 min | +0.12 s |

| Activity | Level | Acceptance criterion | Result |
|---|---|---|---|
| `Detector` contract: clench, double blink, manual, seizure stub | Unit | all pass the same test | ✅ |
| Detection rate on simulated data | Unit | ≥ 95 % (clench), ≥ 90 % (double blink, sim); onset ≤ 0.5 s | ✅ 100 % / 97 %, ≤ 0.13 s |
| False triggers, 10 min of distractors | Unit | ≤ 1 per 10 min | ✅ 0 |
| Cooldown + manual debounce | Unit | gesture inside cooldown never fires | ✅ |
| Calibration: rest + 3 reps → threshold between the distributions, saved and reloaded | Unit + System | quality `good`, profile persists across restart | ✅ |
| EEG window −10/+30 s | Unit | 40 s ± 1 sample, `.npz` + `.csv` + `meta.json` | ✅ |
| Gesture → event in the browser, manual button + hotkey, calibration over REST | System | event < 1.5 s after the gesture ends its 1 s hold | ✅ < 2.5 s from injection (= 1 s hold + ≤ 0.2 s processing) |
| Demo mode (`.\bayn demo`): scripted clench at 15 s → event → window saved | System (manual run) | works with no hardware | ✅ event at 14.9 s, window saved, no errors in any process log |
| Visual banner + plot marker + beep, event timeline, downloads | Manual | seen and heard in the browser | ✅ user, 2026-11-09 |
| Real headset (Muse-3EE1): **jaw clench**, before and after calibration | Hardware (manual) | fires on a deliberate clench, not otherwise | ✅ user: "fine before and after calibration" |
| Real headset: **double blink** | Hardware (manual) | fires on a double blink, not on other activity | ❌ user: **false triggers from head movements**; **natural-strength double blinks missed**, only very deliberate ones detected → finding F1, addressed in Phase 6 |

### Phase 4: Whisper live transcription ✅ (file audio, GPU + CPU) · ⏳ live mic by user

**Goal:** transcribe the microphone live (English, Arabic, or both mixed), timestamp every segment in the EEG clock, run on GPU when there is one and on CPU otherwise, and attach the transcript of each event's window to the event.

**Benchmark** on the dev PC. Test material: [samples/audio/demo.wav](../samples/audio/demo.wav), 55 s, 7 sentences alternating English and Arabic, generated with Windows' offline TTS (English voice + Arabic "Naayf"). Silero VAD segmentation, `mixed` language mode.

| Engine | Language detection | Accuracy | Processing per utterance |
|---|---|---|---|
| **large-v3-turbo, CUDA float16, beam 5** (RTX 2080 Ti) | 14/14 utterances correct (all ar ↔ en switches), confidence 1.00 | English word-perfect; Arabic word-perfect on the short clip, 1 letter off in 2 of 7 sentences on the demo | **0.46 s** mean, 0.54 s max |
| **small, CPU int8, beam 1** (laptop fallback) | 14/14 correct | English word-perfect; Arabic readable with small spelling slips (e.g. "سقوت" for "سقوط") | **2.41 s** mean, 2.55 s max |

**End to end** (`bayn demo --audio file`, all 5 processes, GPU): text appears **1.34 s** on average after the end of speech (max 1.62 s), including the 0.6 s pause that ends an utterance. There were 20 live partial updates. The scripted clench at 15 s fired an event, and its −10/+15 s window got **7 segments (ar + en)** attached with times relative to the event. No process errors.

| Activity | Level | Acceptance criterion | Result |
|---|---|---|---|
| VAD segmentation: split on pauses, padding, forced cut of long monologues, gap reset, flush, `processed_until` | Unit | correct boundaries and unix timestamps | ✅ (`energy_vad` stand-in) |
| Language choice: `mixed` only picks ar/en; `en`/`ar` forced; `auto` free | Unit | e.g. "Persian 60 %, Arabic 30 %" → Arabic | ✅ |
| Hallucination filter ("thanks for watching", "اشترك في القناة", no-speech) | Unit | filtered | ✅ |
| WAV I/O, resampling 44.1 kHz stereo → 16 kHz mono, file source looping, int16 bus format, mic levels | Unit | exact | ✅ |
| Transcript store: time-window query, relative-time formatting | Unit | overlap semantics | ✅ |
| Real Whisper (tiny, CPU) on English and Arabic clips; Silero VAD on the mixed demo | Integration | key phrase found; ar/en detected; both languages in demo | ✅ |
| GPU and CPU auto-selection, CPU fallback | Integration (benchmark) | CUDA when present; `device=cpu` works | ✅ both measured above; fallback path exercised by the warm-up design (forced-failure test still to do) |
| Audio → VAD → Whisper → browser; event gets its transcript attached and downloadable | System | segment in browser, latency < 5 s (CPU tiny), `transcript.txt` served, event row updated | ✅ |
| Live mic → dashboard transcript | System | ≤ 3 s after speech ends (GPU), ≤ 6 s (CPU) | ✅ 1.34 s via file audio (GPU) · ✅ user's live mic (below) |
| **Live mic, user reading 3 scripts** (classroom scenario, manual trigger, GPU, `mixed`) | Hardware (manual) | English WER ≤ 10 %; Arabic and mixed readable | English **1.2 % WER** ✅ · Arabic (Hejazi) **28.8 % WER / 12.0 % CER**, readable ✅ · Mixed **39.7 % WER / 32.8 % CER** ❌ → finding F2 |

Scores are computed by `bayn.transcription.metrics` (normalised: diacritics, hamza forms and punctuation ignored). They are reproduced exactly by `test_scores_of_the_live_user_test_are_reproduced` from [tests/data/scenarios/classroom.json](../tests/data/scenarios/classroom.json), which stores the scripts and the live transcripts and is also a Phase 5 summarization scenario.

**Strategy evaluation tool** (`bayn eval-stt`, same audio, every strategy). Check on the TTS demo clip (sentence-level switching, GPU): every strategy scores 4.9 % WER (mostly "10.15" vs "ten fifteen"); the prompt gives CER 3.1 % vs 3.9 %. No regression. The full comparison on real recordings is in "Later: transcription benchmark" below.
| Offline | System | no network after setup | ✅ by design: models load from a local folder with `local_files_only`, HF telemetry off · network-unplugged run in Phase 8 |

### Phase 5: Ollama summaries (planned)

Realistic test material: the **classroom scenario** in [tests/data/scenarios/classroom.json](../tests/data/scenarios/classroom.json). It has three real live transcripts (English, Arabic, mixed) with their true scripts, *including* recognition errors and unrelated chatter before the reading. A good summary has to cope with both.

| Planned activity | Level | Acceptance criterion |
|---|---|---|
| Classroom scenario × 3 languages: summary vs the true script | Manual + scripted checks | covers the key facts (cup by the window, water level fell, sun → vapor = evaporation, question about cooling vapor); invents nothing; ignores the pre-reading chatter; doesn't repeat transcription errors as facts |
| Prompt builder: transcript window + EEG stats, en / ar / both | Unit | stable prompt structure, window boundaries correct |
| Ollama unavailable / model missing | Integration | clear error in the UI, pipeline keeps running |
| Scripted scenarios: summary covers the facts in the transcript and invents none | Manual | reviewed on ≥ 5 scenarios in English and Arabic |
| Event → summary streamed into the UI | System | first tokens < 5 s after the event window closes (GPU) |

### Phase 6: Gesture tuning on the real headset + ML blink detector (planned)

**ML double-blink detector** (answers finding F1). One amplitude threshold can't separate natural blinks from head-movement artefacts, because their amplitudes overlap. The plan:

1. **Guided labelled recording, about 3 min:** blink twice, blink once, turn head, nod, look left/right, talk, rest. Each prompt labels its time window. This is also the Muse-app-style onboarding.
2. **Candidates:** every forehead peak above a *low* threshold. Features: shape/width, peak count and gap, AF7/AF8 symmetry, TP9/TP10 involvement, slow-drift and muscle-band energy.
3. **Model:** a scikit-learn random forest with classes double blink / single blink / movement / other, saved per user. Runs offline on CPU.
4. **Two variants, compared:** **EEG only**, which mirrors what a real future device may offer, and **EEG + Muse accelerometer**.
5. **`MLDoubleBlinkDetector` behind the `Detector` interface**, with the rule-based detector as fallback. This is also the dry run for plugging in a seizure model.

| Planned activity | Acceptance criterion |
|---|---|
| Labelled recording session completes and saves labelled windows | ≥ 20 double blinks, ≥ 20 single blinks, ≥ 20 movements per session |
| Held-out evaluation (session-wise split): rule vs ML (EEG) vs ML (EEG + accel) | report precision, recall, false triggers/min, confusion matrix |
| ML (EEG only) vs rule-based on the user's data | higher recall at ≤ the rule's false-trigger rate |
| Simulator gains head-movement artefacts | the rule-based detector's false triggers are reproducible in tests; ML fixes them |

Protocol, per tester, headset on, after calibration:

| Trial | Acceptance criterion |
|---|---|
| 20 deliberate jaw clenches, 20 deliberate double blinks | ≥ 18 / 20 detected for the chosen gesture |
| 10 min normal activity: talking, chewing, normal blinking, head movement | ≤ 1 false trigger |
| Gesture-to-event latency | ≤ 1.5 s |
| Repeat with ≥ 2 people | criteria hold for each, using their own calibration |
| Guided onboarding ("blink twice", "clench", "close your eyes", like the Muse app) | a first-time user completes it in < 2 min without help, and it produces a working calibration |

### Phase 7: Setup & packaging polish (planned)

Docker was dropped (2026-11-09): the EEG bridge (Bluetooth), the microphone and the GPU all need native processes, so containers would add complexity without simplifying setup.

| Planned activity | Acceptance criterion |
|---|---|
| `bayn setup` on a clean machine (Windows; one of macOS/Linux) | deps, CUDA libs (if NVIDIA), Whisper + Ollama models, `doctor` all green, no manual steps |
| Offline-readiness check | everything needed for the demo is on disk; network-unplugged demo works |
| Laptop checklist (Bluetooth, mic, GPU/CPU, ports, models) | `doctor` covers each item |
| README quickstart followed literally | works without undocumented steps |

### Phase 8: Full end-to-end test run (planned)

| Run | Acceptance criterion |
|---|---|
| Simulated: demo mode end to end (gesture → event → transcript → summary) | 3 consecutive clean runs |
| Real: Muse + live mic, scripted demo script | 3 consecutive clean runs on the **demo laptop**, offline |
| Failure drills: headset off mid-demo, Ollama stopped, mic unplugged | each failure is visible in the UI and the system recovers or degrades gracefully |

### Later: transcription benchmark on audio files (planned, user request 2026-11-09)

Goal: compare **all models and strategies on identical audio** to understand the limits, instead of judging from one-off live runs.

| Planned activity | Acceptance criterion / output |
|---|---|
| New, more realistic **classroom scripts** from the user: mostly Arabic (MSA + Hejazi) with occasional English key terms ("pie", "Rutherford"…), plus English-only and Arabic-only | scripts + references stored like `classroom.json` |
| Record each script once to WAV (`bayn record`), so every model hears **exactly the same audio** | recordings kept locally in `data/recordings/` (never committed) |
| Run `bayn eval-stt` for every model (tiny, base, small, medium, large-v3, large-v3-turbo) × device (GPU / CPU) × strategy (`per_utterance`, `arabic_lean`, prompt, free detection) | table of WER, CER, latency per utterance, real-time factor |
| **Key-term recall**: share of English key terms in Arabic speech that come out correctly (Latin script, right spelling) | the main metric for code-switching; target ≥ 90 % for the chosen default |
| Pick defaults per machine class (GPU laptop / CPU-only laptop) from the numbers | defaults + limitations documented in README |

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
| Gesture trigger: jaw clench / double blink, configurable | detector unit tests, system test, demo run, user test | ✅ sim · ✅ clench on real headset · ❌ double blink on real headset (F1 → Phase 6) |
| Per-user calibration | calibration unit + system tests, user test | ✅ sim · ✅ clench on real headset |
| Cooldown against false triggers | `test_cooldown_suppresses_second_gesture`, false-trigger tests | ✅ |
| Visual + audio feedback | manual (banner, plot marker, Web Audio beep) | ⏳ user |
| Timestamped event + EEG window −10/+30 s saved | event manager unit tests, system test | ✅ |
| Event starts the pipeline (transcript + summary) | `events.done` message published (system test); consumers arrive in Phases 4–5 | partial ✅ |
| UI button + hotkey backup trigger | `test_manual_trigger_and_hotkey` | ✅ |
| `Detector` interface ready for a seizure detector | `test_detector_contract[seizure_stub]` | ✅ |
| Live mic transcription, every segment timestamped | system test (file audio), GPU demo run | ✅ · ⏳ real mic by user |
| English, Arabic and both mixed | language-choice unit test, model integration tests, demo benchmark (14/14 switches) | ✅ |
| Whisper on CPU, GPU when available | benchmark on CUDA and CPU, `doctor` | ✅ |
| Transcript around the event | `test_live_transcript_and_event_transcript`, demo run | ✅ |
| Local LLM summaries | Phase 5 plan | ⏳ |
| Easy setup on other machines (Docker dropped → native setup) | Phase 7 plan | ⏳ |
| One-command demo without hardware | system tests (5 processes), `bayn demo --audio file` run | ✅ (summary in Phase 5) |

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
| D9 | Unit (detector sweep) | Double blink detected only 13/40 | Three causes: gave up on the 1st blink after 0.45 s although gaps go to 0.8 s; the band-pass rebound after each blink counted as a 3rd blink (absolute value); 110 µV threshold above many filtered deliberate blinks | Wait `max_gap_s` for a 2nd blink; signed peaks with pair polarity; merge peaks closer than `min_gap_s`; 0.3 Hz low cut; 90 µV default → 58/60 | `test_double_blink_detection_rate`, `test_rebound_is_not_a_third_blink` |
| D10 | System | Event fired right after start saved a short window without saying so | Pre-event data did not exist yet | Window flagged `partial` when shorter than expected | `test_event_right_after_connect_is_marked_partial` |
| D11 | Manual run | Demo mode's scripted double blinks never fired an event | Demo alternates clench / double blink, but the default trigger gesture is clench only | Demo mode listens to both gestures | demo run |
| D12 | Integration (benchmark) | `TypeError: 'int' object is not callable` in the segmenter | Attribute `self.step` (sample count) shadowed the method `step()` | Renamed to `step_n` | all segmenter tests |
| D13 | Manual run | Sample-audio script failed to parse; Arabic text garbled | Windows PowerShell 5.1 reads UTF-8 `.ps1` files without BOM as ANSI | Arabic sentences moved to a UTF-8 JSON data file read with `-Encoding UTF8`; scripts kept ASCII | sample-audio run |
| D15 | Live mic (user) | Event transcript lost the end of the last sentence ("يبرد فوق في السما") | A pause > 0.6 s just before +30 s split the sentence; the second half started after the window end | Window transcript includes speech continuing with < 1.5 s pause, up to `event_tail_s` (4 s) | `test_event_transcript_finishes_the_sentence_in_progress` |
| D14 | Unit (test bug) | Timestamp continuity check failed for the file source | 1e-9 s tolerance is below double precision at unix ~1.8e9 s (~2e-7 s) | 1e-6 s tolerance | `test_stitch_and_file_source` |

### Findings (performance limitations, not bugs)

| ID | Found by | Finding | Likely cause | Plan |
|---|---|---|---|---|
| F2 | Live mic, user test 2026-11-09 | Mixed Arabic/English **inside** a sentence: 39.7 % WER. English-decoded utterances **translate or transliterate the Arabic** ("فاكرين إيش صار لها" → "Do you think what happened?", شباب → "shabab"); Arabic-decoded ones mostly keep the English words | `mixed` forces one language per utterance; Whisper handles intra-sentence code-switching poorly when forced to English | **Current limitation, accepted for now.** The test script was very informal and heavily mixed. The realistic case is Arabic speech with **occasional English key terms** (e.g. "pie", "Rutherford"), which **must not be lost**. To be measured in the transcription benchmark (§4, "Later"). Candidate fixes are already in the code but off by default: `mixed_strategy: arabic_lean`, `initial_prompt: auto` |
| F1 | Hardware, user test 2026-11-09 | Rule-based double blink: head movements cause false triggers; natural-strength double blinks are missed | Movement artefacts and blinks overlap in amplitude on AF7/AF8; calibration "rest" includes natural blinks, which pushes the threshold up | Phase 6: ML detector (EEG only and EEG + accelerometer), evaluated against the rule on labelled real data. Until then, **jaw clench** is the recommended trigger |

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
- Gesture detection performance is so far measured on **simulated** gestures only. The simulator's clench and blink shapes are educated guesses; the real-headset numbers come from the Phase 6 protocol.
- Audio feedback needs one click on the page first (browser autoplay rule). The 🔇/🔔 button does that.
- **Code-switching inside a sentence (F2)** is unreliable: Arabic words can be translated or transliterated when a sentence is decoded as English. Sentence-level switching works well (14/14 on the demo clip). English key terms inside Arabic speech are not yet measured; see the planned benchmark.
- Arabic dialect (Hejazi) words are sometimes misspelled or merged (≈ 12 % CER). Modern Standard Arabic is better.
- Hardware tests have only run on the dev PC. They must be repeated on the demo laptop.
- The dashboard UI has no automated browser tests. It is checked manually.
