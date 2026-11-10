# Testing & Validation

This document describes how Bayn is verified: which tests exist at each level (unit, integration, system, hardware-in-the-loop, manual acceptance), what each phase was tested against, the results, and the defects that testing found.

> [!IMPORTANT]
> Bayn is a hackathon prototype, **not a medical device**. "Validation" here means *engineering* verification and validation: does the software do what the design says, on real hardware, reliably enough for a live demo? **No clinical validation has been done, and none is claimed.**

**Status:** Phases 1–5 tested, including the user's live-mic check of Phase 5 summaries in the dashboard. The **model benchmark** (between Phases 5 and 6) compared 6 Whisper models × 11 summary LLMs on the user's 12 recorded corpus scripts, on the dev PC's GPU and on an emulated 4-thread CPU laptop ([BENCHMARK.md](BENCHMARK.md)). The default summary model is now Gemma 4 E2B. **The best demo configuration will be re-tested on the demo laptop**, with its own microphone. On the real headset the jaw-clench trigger works; the rule-based double blink does not yet (finding F1). **Phase 6** switched the trigger to a **triple blink** (user's choice) and built an EEG-only ML detector (verified in simulation; the user's labelled headset sessions are next). Phases 7–8 have test plans with acceptance criteria, filled in as each phase lands.
Last updated: 2026-11-10.

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
| **Unit** | One function or class in isolation. No sockets, threads, devices or ML models. Detector and event-manager tests stream minutes of simulated EEG through the logic offline, in seconds. The segmenter uses a simple energy VAD stand-in. | `tests/test_filters.py`, `test_buffer_quality.py`, `test_simulator.py`, `test_bands.py`, `test_detectors.py`, `test_events_manager.py`, `test_transcription_unit.py`, `test_transcription_eval.py`, `test_summarization.py`, `test_corpus.py`, `test_benchmark.py`, `test_blink_ml.py`, parts of `test_bus_config.py` and `test_sources.py` | 165 | ~40 s | Every `test` run |
| **Integration** | Several components together across real boundaries: TCP sockets, threads, asyncio, BrainFlow's native library (synthetic board), LSL networking, real Silero VAD + Whisper models, Ollama API (fake server; real model when running), the benchmark's LLM stage and report. | `test_bus_config.py`, `test_sources.py`, `test_transcription_models.py`, `test_transcription_eval.py`, `test_summarization.py`, `test_benchmark.py`, `test_blink_ml.py` | 22 | ~30 s | Every `test` run |
| **System** | The real `bridge`, `events`, `audio`, `transcriber`, `summarizer` and `dashboard` *processes*, end to end, driven like a browser would (WebSocket + REST). Simulated EEG, WAV audio. | `tests/test_system.py` | 7 | ~60 s | Every `test` run |
| **Hardware-in-the-loop** | The real Muse 2 over Bluetooth, through both acquisition paths. | `tests/test_hardware.py` | 2 | ~45 s | On demand, with headset |
| **Manual acceptance** | A human checks the dashboard, ergonomics and demo flow against a checklist. | §4 (per phase), §6 | – | minutes | End of each phase |

### 2.1 Unit tests (165)

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
| Detectors: triple blink | `test_triple_blink_detection_rate`, `test_single_and_double_blinks_do_not_fire`, `test_blink_count_is_configurable` | 30 injected triple blinks over 3 seeds, natural blinks + muscle bursts on: ≥ 90 % detected, 0 false; single and double blinks never fire; N is a setting (until 2026-11-10 the trigger was a double blink) |
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
| Summaries | `test_output_language_follows_the_transcript`, `test_prompt_has_rules_json_example_and_relative_times`, `test_arabic_prompt_is_written_in_arabic`, `test_schema_caps_key_points` | output language; rules, unrelated JSON example, relative times; Arabic instructions in Arabic; ≤ N key points |
| | `test_render_localised_headings`, `test_partial_render_streams_text_while_json_is_incomplete`, `test_eeg_section_is_written_by_code_not_the_llm`, `test_assembled_document_has_disclaimer` | rendering incl. half-finished JSON while streaming; code-written EEG section; disclaimer |
| | `test_quality_check_rewards_facts_and_catches_flips_and_wrong_language`, `test_facts_only_required_where_the_transcript_has_them`, `test_auto_model_depends_on_gpu_memory` | quality checks; `auto` model by GPU memory |
| Bus / config | `test_pack_roundtrip_ndarray`, `test_schema_roundtrip` | msgpack + NumPy and message schemas survive serialization |
| | `test_env_overrides_typed`, `test_bad_notch_rejected` | `BAYN_*` overrides get the right types; invalid values rejected |
| Sources | `test_sample_clock_regularises_jitter` | ±20 ms packet jitter → perfectly regular timeline |
| | `test_sample_clock_reanchors_after_gap` | a 4 s dropout re-anchors the clock |
| | `test_unknown_brainflow_board_rejected` | invalid board name raises |
| Triple blink / ML (Phase 6) | `test_blink_ml.py` (unit part), `test_detectors.py` | simulator artefacts (nod, head turn, glance, talking) and the TP9/TP10 blink coupling; the rule false-triggers on nods (F1 reproduced); protocol prompt counts and length; session save/load; finder groups a double blink into one episode; features finite and **identical offline and live**; labels follow the prompts; rule fallback; the event manager records a session without firing events |

### 2.2 Integration tests (22)

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

| `test_client_streams_tokens_and_reports_speed`, `test_client_error_and_unreachable`, `test_stalled_model_times_out_instead_of_hanging` | Ollama client ↔ fake Ollama server | streaming, JSON schema sent, error lines, unreachable host, stall timeout (D17) |
| `test_service_summarises_window_streams_and_saves`, `test_service_handles_no_speech_and_llm_failure` | summarizer service ↔ fake Ollama + event store | en + ar streamed and saved, event row updated; no-speech path; failure + retry |
| `test_real_model_summarises_the_english_classroom_scenario` | real Ollama model (skipped if not running) | ≥ 60 % facts, nothing invented, right language + format |
| `test_ml_beats_the_rule_on_held_out_sessions`, `test_live_ml_detector_on_a_fresh_stream`, `test_auto_uses_ml_only_if_it_beat_the_rule`, `test_incompatible_model_is_rejected` | simulated recording sessions → features → random forest → live `MLBlinkDetector` | ML catches ≥ 85 % of triple blinks with no more false triggers/min than the rule; on a fresh stream full of nods and double blinks, no more false triggers than the rule; Auto only switches when the held-out test favours ML; models for other channel setups rejected (Phase 6) |
| `test_evaluation_tool_scores_strategies_on_a_clip` | `bayn eval-stt` logic: VAD + Whisper + metrics | every requested strategy scored on the same utterances |

The four Whisper tests are skipped if the `tiny` model isn't downloaded (`bayn models`).

### 2.3 System tests (7)

| Test | Verifies |
|---|---|
| `test_end_to_end_simulated_pipeline` | Starts the bridge and dashboard as separate processes on free ports, connects over WebSocket like the browser, and checks: `hello` handshake; ~256 samples/s arrive in 4 channels (6–11 s of data in 9 s); per-channel quality reported for TP9/AF7/AF8/TP10; **mains detected as 60 Hz when the simulator injects 60 Hz** (detected, not assumed); a REST `double_blink` injection makes the full round trip back to the browser; brainwave band updates arrive at ~2 Hz (5 bands, relative powers sum to 1). |
| `test_gesture_event_saved_and_downloadable` | Bridge + events + dashboard processes: a clench injected through the REST API fires an `event` in the browser stream < 2.5 s later (1 s hold included); `event_done` arrives with exactly the configured window (−2/+3 s in the test); the CSV downloads over HTTP; `/api/events` lists it. |
| `test_manual_trigger_and_hotkey` | `POST /api/events/trigger {source: hotkey}` → a `manual` event with source `hotkey` reaches the browser. |
| `test_guided_calibration_with_simulator` | `POST /api/events/calibration` → prompts stream to the browser; the simulator performs each prompted clench; calibration ends `done` with quality `good`. |
| `test_live_transcript_and_event_transcript` | All 6 processes (summarizer against a fake Ollama). The English test clip plays in a loop as the "microphone", with the tiny model on CPU. A transcript segment reaches the browser (`en`, contains "time", unix timestamps, latency < 5 s). A manual event fired during speech gets `transcript_window` with ≥ 1 segment. `transcript.txt` downloads. The event row in `/api/events` records the transcript. Status reports model/device and `audio_ok`. **Then the summary**: the prompt carries the transcript, text streams while writing, the final markdown has the summary and the code-written EEG section, and `summary.md` downloads. |
| `test_blink_training_session_with_simulator` | Phase 6: `POST /api/events/training` starts a labelled recording (shortened ×0.1). Prompts for every gesture class stream to the browser, the simulator performs them, the trigger state shows `training`, the session is saved to disk, and training runs in the background and reports `trained` or "not enough examples". The blink method can be switched; a bad action gets HTTP 400. |
| `test_bad_gesture_rejected` | The REST API rejects an invalid gesture with HTTP 400. |

### 2.4 Hardware-in-the-loop tests (2)

| Test | Pass criteria |
|---|---|
| `test_muse_via_brainflow` | connects within 60 s; effective rate 256 ± 3 Hz; channel order TP9, AF7, AF8, TP10; median latency (receive time − sample timestamp) within ±0.5 s; ≥ 3 of 4 channels not `bad` |
| `test_muse_via_lsl` | same criteria through `muselsl stream` |

---

## 3. Test environment

| | Dev PC (tested) | Demo laptop (ASUS Vivobook 15 X1504VA, scanned 2026-11-10) |
|---|---|---|
| CPU / RAM | AMD Ryzen 7 5800X3D (8 cores / 16 threads), desktop | Intel Core i7-1355U (2 P + 8 E cores, 12 threads, 15 W); **7.6 GB RAM, one 8 GB DDR4-3200 module (single channel)**; ~2.2 GB available with VS Code open |
| Power | desktop | plugged in, Balanced plan with the "Best performance" power mode (AC and battery); battery saver off |
| Disk | models on H: | one drive C:, 328 GB free |
| Microphone | TC-Helicon GoXLR Mini "Chat Mic" (default input) | **wireless lavalier, USB 2.4 GHz receiver "USBAudio1.0"** (not Bluetooth), `audio.device` in config/local.yaml; 16 kHz, noise-gated, SNR ~72 dB, peak -1.8 dBFS (little headroom) |
| Whisper | large-v3-turbo on CUDA (cuDNN 9, CTranslate2 4.8.5); small + tiny also on disk | CPU only; chosen by the laptop benchmark (step 5) |
| OS | Windows 10 Pro 19045 | Windows 11 Home 26200 |
| Python | 3.13.7 | 3.13.7 |
| GPU | NVIDIA RTX 2080 Ti 11 GB | none (Intel UHD integrated) → CPU-only path |
| Bluetooth | USB dongle (Actions ATS2851, vendor driver); built-in QCA61x4A disabled | built-in Intel Wireless Bluetooth, the only adapter; working (`doctor`: 6 BLE devices visible). Paired: soundcore Q20i headphones |
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

### Phase 5: Ollama summaries ✅ (dev PC, GPU) · ✅ user test (live mic + dashboard, 2026-11-09: "very satisfactory")

**Goal:** when an event's window closes, a small local LLM summarises the transcript of the window. The dashboard shows how the captured text turns into the summary, in English and/or Arabic, without inventing content.

**Test material:** the **classroom scenario** ([tests/data/scenarios/classroom.json](../tests/data/scenarios/classroom.json)): three real live transcripts (English, Arabic, mixed) *including* recognition errors and unrelated chatter before the reading. Automatic checks per summary (`bayn eval-summary`):
- **facts:** key facts present, counting only facts the live transcript actually contains
- **invented:** recognition errors stated as facts ("happy water" from "فرحت"), the unrelated chatter, medical words, **meaning flips** ("water level rose")
- **language:** written in the requested language
- **format:** all sections present

**Model comparison** (RTX 2080 Ti; 3 transcripts × English/Arabic output × 2 runs = 12 summaries per model):

| Model, prompt | Facts (en / ar) | Invented | Right language | Pass | Avg time |
|---|---|---|---|---|---|
| qwen2.5:3b, English-only prompt (first probe) | – | "happy water", invented "make a noise"; **Arabic requested → English written** | ❌ | – | 2 s (first load 64 s) |
| qwen2.5:3b, localised markdown prompt | 80 % / 56 % | 0/12 | 12/12 | 3/12 | 2.6 s |
| **qwen2.5:7b, localised markdown prompt** | 88 % / 64 % | 0/12 | 12/12 | 5/12 | 3.9 s |
| llama3.2:3b, localised markdown prompt | 70 % / 68 % | **4/12** ("Black Clover", "فرح") | 10/12 | 0/12 | 26.6 s (repetition loops) |
| qwen2.5:3b, **structured JSON** | 65 % / 35 % | 1/12 | 12/12 | 2/12 | 1.4 s |
| **qwen2.5:7b, structured JSON (chosen)** | **96 % / 68 %** | **0/12** | **12/12** | **8/12** | 3.3 s |

**Decisions:**
- **Structured output.** Ollama forces the answer into `{summary, key_points ≤ 4, unclear}` and code renders localised headings. This fixed the Arabic format (the model skipped sections) and stopped "line-by-line translation instead of summarising". It also caught a **meaning flip**: one free-text Arabic summary turned "the water level went down" into "ارتفع مستوى الماء" ("rose").
- **`model: auto`** (at Phase 5): qwen2.5:7b with an NVIDIA GPU ≥ 10 GB, else qwen2.5:3b. **Superseded by the model benchmark** (below): Gemma 4 E2B with ≥ 6 GB VRAM or ≥ 12 GB RAM, else Qwen 3 4B.
- **The EEG section is written by code** from the measured statistics, labelled "signal description, not a medical assessment". The LLM never sees EEG numbers, so it can't invent medical interpretations.

**End to end** (`bayn demo --audio file`, real Whisper large-v3-turbo + qwen2.5:7b on GPU). The scripted clench fired an event, and its window (10 segments, ar + en) was summarised **2.4 s after the window closed**. The first text appeared after 0.5 s, with 21 live updates at 58 tokens/s. The English summary merges Arabic and English speech correctly but adds one small embellishment ("or dropping of his head", not in the transcript; F3).

| Activity | Level | Acceptance criterion | Result |
|---|---|---|---|
| Prompts: language detection, rules, JSON example (unrelated topic), relative times, Arabic prompt in Arabic, schema caps key points | Unit | structure as designed | ✅ |
| Rendering: localised headings, **partial JSON rendered while streaming**, EEG section by code, disclaimer | Unit | correct for en/ar | ✅ |
| Quality checks: facts, flips, wrong language, facts limited to what the transcript contains | Unit | catches each failure type | ✅ |
| `auto` model choice (updated after the benchmark): GPU ≥ 6 GB, or no NVIDIA GPU and RAM ≥ 7 GB (demo laptop, 2026-11-10), or a small GPU and RAM ≥ 12 GB → gemma4:e2b, else qwen3:4b, fall back to what's pulled | Unit | as designed | ✅ |
| Ollama client against a fake server: streaming, JSON schema sent, error lines, unreachable server, **stalled model times out** | Integration | errors surface as `OllamaError` | ✅ |
| Service: both languages streamed and saved (`summary.md/json`), event row updated, no-speech without an LLM call, LLM failure reported after one retry | Integration | as designed | ✅ |
| Real model (Ollama running): English classroom scenario | Integration | ≥ 60 % facts, nothing invented, right language and format | ✅ qwen2.5:7b · ✅ gemma4:e2b (2026-11-10) |
| Speech → transcript → event → **summary streamed to the browser**, prompt visible, `summary.md` downloadable | System | all processes, fake Ollama for determinism | ✅ |
| Event → summary on real models | System (manual run) | first text < 5 s after the window closes | ✅ 0.5 s (total 2.4 s) |
| Dashboard: ① heard → ② context (EEG + prompt) → ③ summary writing live, EN/AR tabs, regenerate | Manual | readable, understandable | ✅ user, live mic ("very satisfactory") |

### Model benchmark (between Phases 5 and 6) ✅ dev PC · ⏳ demo laptop

**Goal** (user request 2026-11-09/10): find the strongest Whisper × summary-LLM combination for English, Arabic and mixed speech, and make sure it still works with minimal resources. Models are compared on **identical audio** instead of one-off live runs.

**Material:** the [recording corpus](CORPUS.md). The user recorded all 12 scripts (4 topics × English / Arabic / mixed, ~35 s each) with a good desk microphone; recordings stay local in `data/recordings/`. Every script has key terms and summary facts, and the incident topic lists forbidden diagnosis words.

**Method** (`bayn benchmark`, [src/bayn/benchmark.py](../src/bayn/benchmark.py), report [BENCHMARK.md](BENCHMARK.md)):
1. **Whisper:** tiny, base, small, medium, large-v3-turbo and large-v3 transcribe every recording with the live settings (`mixed` language mode, same VAD segmentation). Two profiles: `gpu` (RTX 2080 Ti, float16, beam 5) and `laptop` (CPU only, **4 threads**, int8, beam 1), which emulates a modest laptop. Each model runs in its own process, so memory is measured cleanly.
2. **LLM screening:** 11 candidates (Qwen 2.5 1.5/3/7B, Qwen 3 1.7/4/8B, Qwen 3.5 4B, Gemma 3 1/4B, Gemma 4 E2B/E4B) summarise the perfect script text in English and in Arabic (24 summaries each).
3. **Combinations:** the 6 finalists summarise what 8 Whisper configurations actually produced (1,152 summaries).
4. **Minimal laptop speed:** the finalists on the CPU with 4 threads (`num_gpu=0`). Real GPU / RAM use is measured from outside (driver and process memory), because Ollama's own report is wrong for Gemma 4 (E7).
5. Scores are recomputed from the saved text whenever a metric changes, so a scoring fix never requires re-running models.

**Results (dev PC):**

| Whisper | WER en | CER ar | CER mixed | Eng. terms kept (mixed) | RTF GPU | RTF 4 CPU threads | Memory |
|---|---|---|---|---|---|---|---|
| large-v3 | 5.7 % | 7.0 % | 20.0 % | 25 % | 0.24 | 1.65 ❌ | 3.9 GB VRAM |
| **large-v3-turbo** (GPU default) | 5.4 % | 7.2 % | 22.3 % | 15 % | 0.07 | 1.28 ❌ | 2.3 GB VRAM |
| medium | 16.9 % | 9.7 % | 22.6 % | 15 % | 0.19 | 0.95 ❌ | 2.1 GB VRAM |
| **small** (CPU default) | 4.5 % | 12.3 % | 30.3 % | 5 % | 0.08 | **0.36** | 0.6 GB RAM |
| base / tiny | 27 / 11 % | 24 / 31 % | 37 / 43 % | ≤ 25 % | < 0.1 | ≤ 0.12 | ≤ 0.4 GB |

| Summary LLM | Facts, perfect transcript | Facts, real transcripts (turbo / small-CPU) | Mixed speech (turbo) | GPU memory | 4 CPU threads: time, RAM |
|---|---|---|---|---|---|
| **gemma4:e2b** (new default) | 74 % | 81 % / 77 % | 82 % | 3.0 GB | **8 s**, 4.7 GB |
| gemma4:e4b | 78 % | 83 % / 78 % | 82 % | 4.8 GB | 17 s, 6.7 GB |
| **qwen3:4b** (new 8 GB-RAM default) | 75 % | 70 % / 61 % | 74 % | 3.1 GB | 19 s, 3.1 GB |
| qwen2.5:7b (old GPU default) | 73 % | 68 % / 56 % | 66 % | 4.8 GB | 32 s, 4.9 GB |
| qwen2.5:3b (old CPU default) | 55 % | 52 % / 44 % | 48 % | 2.1 GB | 11 s, 2.1 GB |
| gemma3:1b | 62 % | 53 % / 48 % | 36 % | 0.9 GB | 8 s, 0.9 GB |

No model put a diagnosis into an incident summary, except one gemma3:1b summary. Each case ran once, so differences of a few points are noise.

**Decisions:**
- **Summary model `auto`:** `gemma4:e2b` with an NVIDIA GPU ≥ 6 GB, Apple Silicon, no NVIDIA GPU and ≥ 7 GB RAM (added after the demo-laptop benchmark, 2026-11-10: 1.7 GB resident, 19 s per summary, 70 % vs 45 % facts via Whisper small against qwen3:4b; docs/BENCHMARK_LAPTOP.md), or a smaller NVIDIA GPU and ≥ 12 GB RAM. It is the best measured on real transcripts, especially mixed speech, where it understands English terms that Whisper wrote in Arabic letters. It is also lighter on the GPU than qwen2.5:7b and the fastest 4B-class model on a CPU. Otherwise `qwen3:4b` (< 7 GB RAM, or a small GPU with < 12 GB RAM; not measured there).
- **Whisper `auto` unchanged:** turbo on a GPU, small on the CPU. Small is the only accurate model that keeps up live on 4 CPU threads; medium (RTF 0.95) may keep up on a laptop with more cores, to be checked there.
- **To re-test on the demo laptop** (user, 2026-11-10): the same benchmark with the laptop's own microphone and the `cpu` profile (all cores). The demo configuration is chosen from those numbers.

| Activity | Level | Acceptance criterion | Result |
|---|---|---|---|
| Numbers count the same in words and digits ("ten twenty" = "10:20" = "عشرة وعشرين"), alternative spellings ("Sara\|Sarah") | Unit | WER 0 between equivalent forms | ✅ (D19) |
| Benchmark stages against a fake Ollama: CPU-only options, caching/resume, report, per-machine recommendation | Integration | as designed | ✅ |
| `think: false` sent to reasoning models, retried without for servers that reject it | Integration | as designed | ✅ |
| Full benchmark on the dev PC | Manual | report generated | ✅ [BENCHMARK.md](BENCHMARK.md) |
| Same benchmark on the demo laptop with its microphone (`--profiles cpu`, plus `gpu` if it has an NVIDIA GPU) | Manual | demo configuration chosen from its numbers | ⏳ laptop phase |

### Phase 6: Triple-blink trigger + ML detector (EEG only) ✅ · decision: ML not necessary · ✅ user demo test

**Goal** (answers finding F1): a blink trigger that ignores head movement, nods, sideways glances and talking, yet still catches natural-strength deliberate blinks. One amplitude threshold can't do that, because these artefacts are as large as blinks.

**Scope changes (user, 2026-11-10):**
- **The trigger is THREE blinks**, not two. Double and single blinks are now near misses that must not fire. The number is a setting (`events.triple_blink.count`).
- **EEG only.** No accelerometer variant, because the end device won't have one.
- **The Muse-style onboarding was dropped.**
- **The synchronised TP9/TP10 peak the user noticed when blinking** is measured and used as a feature.

**How it works** ([blinks.py](../src/bayn/events/detectors/blinks.py), [blink_ml.py](../src/bayn/events/blink_ml.py), [training.py](../src/bayn/events/training.py)):
1. **Threshold rule (fallback):** exactly N blink peaks on AF7+AF8 (0.3–10 Hz), each 0.2–0.8 s after the previous one.
   - Later blinks of a fast series ride on the previous blink's high-pass rebound, so they count by their **rise from the dip before them** (D24).
   - A much weaker stray natural blink next to the series isn't counted.
2. **Labelled recording sessions** in the dashboard (Event trigger → Blink detector → *Train…*). Each prompt labels its time window; recordings are saved to `data/training/<profile>-<source>/` and accumulate.
   - **Standard** (~5.2 min): 12 triple blinks plus the things that must not fire: 8 double blinks, 6 single blinks, 8 head turns, 8 nods ("yes, yes, yes"), 8 sideways glances and 2 talking periods, in random order.
   - **Noisy** (~4.4 min): talking, looking around, free head movement and natural sitting, with 16 triple-blink prompts *during* the activity. This is the robustness test.
3. **Candidate finder:** a deliberately sensitive streaming peak finder on the forehead signal. Its threshold is 3 × the robust noise level, kept between 30 and 100 µV. Nearby peaks form an *episode*.
4. **31 EEG features per episode:**
   - Forehead: peak count and strong-peak count, both gaps between the three largest peaks, their relative sizes and widths, rebound.
   - AF7/AF8: agreement (blinks) versus opposition (eye movements).
   - **TP9/TP10 vs forehead:** correlation, size at the forehead peak, lag, TP9–TP10 agreement.
   - Slow electrode drift (head movement).
   - Muscle energy versus just before the episode (talking, neck).
5. **Random forest per user** (scikit-learn, CPU, ~1 MB) with classes triple blink / blinks (one or two) / movement / other. It fires when P(triple blink) ≥ 0.5.
6. **`MLBlinkDetector`** sits behind the `Detector` interface.
   - `method: auto` uses the model only if it beat the rule on held-out data (≤ 5 points lower recall, no more false triggers/min); `ml` forces it; otherwise the rule runs.
   - The finder and features run identically offline and live (tested).
7. **Evaluation against the rule** on held-out data: leave one session out, or 4 time blocks within a single session. Metrics are triple-blink prompts caught and false triggers per minute, broken down by what the person was doing.

**Results so far:**

| Data | Rule: caught · false triggers | ML: caught · false triggers |
|---|---|---|
| Simulator, 2 standard + 2 noisy sessions (19 min), leave one session out | 54/56 (96 %) · **1.08/min** (nodding three times 9, head movement 6) | 53/56 (95 %) · **0.34/min** |
| Simulator, only 1 standard + 1 noisy session (9.6 min) | 26/28 (93 %) · 0.41/min | 23/28 (82 %) · 0.95/min: too little data, so Auto keeps the rule |
| Simulator, fresh stream: 8 triple blinks among 7 nods, 3 double blinks, head turns, glances, talking | 8/8 · 2 nods fired | 8/8 · 1 false (a natural blink right before a double blink made three) |
| **Real Muse, user's standard + noisy session (9.7 min), leave one session out**, blink signal / threshold / synthetic data learned from the training session only | forehead average: 13/28 (46 %) · 1.22/min. **Learned blink signal: 22/28 (79 %)** · 2.16/min (fires on double blinks) | real data only: 21/28 (75 %) · 1.76/min. **Real + synthetic: 23/28 (82 %) · 0.81/min** (4 of 6 false: nodding) |

**What the user's first real sessions showed** (plots of every prompt per sensor):
- **On this Muse, blinks are largest on TP9/TP10** (±300 µV, sharp dip-then-peak per blink, the same on both). AF7 shows a smaller mirror image, and AF8 is noisy and moves with TP. The Muse reference sits on the forehead (Fpz), right above the eyes, so a blink appears inverted on the ear sensors. That's the "synchronised TP9/TP10 peak" the user saw. The AF7/AF8 average used until then was close to the *weakest* view of the blinks (D25).
- **Nods and head turns look different:** large slow swings on AF7/AF8 (AF8 up to ±500 µV), small on TP.
- **Fix 1: a learned blink signal.** From the person's labelled blink prompts, the dominant spatial pattern is learned (first principal component, median over prompts), giving TP9 −0.62, AF7 +0.20, AF8 −0.35, TP10 −0.67 for this user. The rule, the candidate finder and the features all use it, and the rule's threshold is learned from the same data. A calibration done on another signal is ignored (calibration results record which signal they belong to).
- **Fix 2: synthetic data from the person's own recordings** (user's idea). Their real individual blinks are cut out and recombined as 1–4 blinks with varied strength (×0.75–1.3), speed (×0.85–1.2) and gaps (their real triple-blink rhythm ×0.8–1.3), placed on real stretches of their recordings, nods and head turns included. They are used only for training; every number above is on held-out *real* data. This halved the false triggers (1.76 → 0.81/min).
- **Boundaries session** (user's idea, `boundaries` protocol, ~6.4 min): the person shows the edges, both what should count (soft / fast / slow triple blinks, while looking down or talking) and the closest look-alikes (double and four blinks, one hard blink, squeezing the eyes, natural blinks, nodding three times, looking up/down, raising eyebrows). The report breaks misses down by variant and false triggers by activity.

#### Decision test: is a machine-learning model necessary? ([BLINK_DECISION.md](BLINK_DECISION.md))

Once the blink signal was right, the user asked whether ML is needed at all (2026-11-10). The test was **pre-registered before the deciding sessions were recorded.** The criteria are constants in `bayn.events.blink_ml`, and the report is generated (`.\bayn blink decide`).
- **Contenders:**
  - The rule on the learned blink signal.
  - The classifier trained on real + synthetic data.
  - Both learn only from the sessions not being tested (leave one session out, real recordings only).
- **Decision rule:** ML is necessary only if it is *clearly* better, meaning either:
  - ≥ 10 points more triple blinks caught with no more false triggers, or
  - ≥ 0.5/min fewer false triggers while catching ≤ 5 points fewer.
  - Otherwise the simpler rule wins. Auto applies the same rule live.
- **Final once there are:**
  - a standard, a noisy and a boundaries session;
  - sessions done seated *and* standing (user: the demo may be given standing).
  - Walking in place (spoken prompts) is reported per condition, not required.
- **Re-recording:** bad sessions may be deleted and re-recorded *before* running the test, never because of its result.

| Status | Rule (learned signal) | ML (real + synthetic) | Verdict |
|---|---|---|---|
| Preliminary (standard + noisy, seated, 9.7 min) | 24/28 caught (86 %, CI 69–94 %) · 0.41 false/min (all nods) | 23/28 (82 %, CI 64–92 %) · 0.81/min | ML not necessary so far |
| **FINAL** (6 sessions, 31 min: 2 standard, 2 noisy, 1 boundaries seated, 1 standard standing; 96 triple-blink prompts) | **64/96 (67 %) · 0.30 false/min** | 61/96 (64 %) · 0.73/min | **ML is not necessary.** Same recall, less than half the false triggers. Auto runs the rule on the learned blink signal |

**Why recall fell to 67 % in the final data** (diagnosed after the decision; the verdict is unchanged):
- **16 of 32 misses:** slow and soft triple blinks had gaps of 0.85–1.0 s, but the rule allows at most 0.8 s, a value guessed in the simulator.
- **7 misses:** later blinks of a series measure larger (they rise from the previous blink's dip), so the first blink was dropped as "too small".
- **4 misses:** a fourth peak (possibly real fourth blinks).
- **3 misses:** fast blinks merged.
- The classifier's peak finder has the same gap limit.
- **Possible follow-up, Rule v2 (not built):** learn the gap limits from the person's own triple blinks, measure every blink of a series the same way, and evaluate held-out. It would be reported separately and wouldn't change the recorded verdict.

**User demo test (2026-11-10, live, Auto = rule on the learned signal):** "triple blink is satisfactory for a demo… if the blinks are intentional enough and not too soft they are caught almost every time". It is not scored, because some triple blinks were made during the 15 s cooldown.

**Exploratory nod check** (forehead/ear ratio): it lost blinks held-out (71 % vs 86 %). Blinks done while talking or moving have forehead/ear ratios up to 3.3, overlapping nods (2.9 and 5.8), so a cut-off learned in one condition doesn't transfer. It is reported but not used.

**Lesson (simulator):** the classifier needs roughly **2 standard + 2 noisy sessions** (~20 min) before it beats the rule. That's why Auto checks the held-out result before switching.

**First look at real data** (the 15 event windows from the user's Phase 3 headset test, unlabelled):
- Real blinks are large and wide: about 300–450 µV and ~170 ms on the forehead, versus 115 ms in the simulator.
- **TP9/TP10 move with the forehead peak, weakly and positively** (median correlation +0.1 to +0.3), so the simulator now couples blinks that way. The labelled sessions measure it per class (`python -m bayn.events.blink_ml inspect`).
- Real forehead noise is 22–87 µV robust sigma (simulator ~10), which led to defect D23.

| Activity | Level | Acceptance criterion | Result |
|---|---|---|---|
| Rule: triple blinks caught; single and double blinks never fire; count configurable; band-pass rebound is not an extra blink | Unit | ≥ 90 % caught, 0 false | ✅ 29/30 |
| Simulator artefacts: nod (2–3 times), head turn, sideways glance, talking; triple/double blink series; TP9/TP10 blink coupling | Unit | each has its signature (glance: AF7/AF8 opposite; nod: wide, large on TP) | ✅ |
| The rule false-triggers on simulated nodding (reproduces F1) | Unit | ≥ 2 of 16 nods fire | ✅ |
| Protocols: prompt counts, ~4.5–5.5 min, random order per session; session save/load | Unit | as designed | ✅ |
| Finder groups a triple blink into one episode; features finite; **offline features = live features** | Unit | exact match | ✅ |
| Labels follow the prompts (each triple-blink prompt yields its episode; double/single blinks become "blinks") | Unit | ≥ all but one | ✅ |
| Train on 4 simulated sessions, leave one session out | Integration | ML recall ≥ 85 %, false triggers/min ≤ the rule | ✅ 95 %, 0.34 vs 1.08/min |
| Live ML detector on a fresh simulated stream full of near misses | Integration | all but ≤ 2 triple blinks caught, ≤ 2 false, ≤ the rule's false, ≤ 1 near miss fired | ✅ |
| Auto uses ML only if it beat the rule held-out; `ml` forces it; no model → rule; incompatible model rejected | Unit / Integration | as designed | ✅ |
| Event manager records a session (no events fire meanwhile), simulator performs the prompts, asks for training | Unit | as designed | ✅ |
| Dashboard → recording session → saved → trained in the background (real processes) | System | session saved; trained or "not enough examples" reported | ✅ |
| Learned blink signal recovers a TP-dominant pattern; synthetic data built only from the person's blinks; report compares 4 variants; a model moves the rule onto the learned signal, and a calibration on another signal is ignored | Unit / Integration | as designed | ✅ |
| **Real headset: decision test** (pre-registered, [BLINK_DECISION.md](BLINK_DECISION.md)) | Manual (user) + held-out | answers "is ML necessary?" | ✅ FINAL: no (rule 67 % / 0.30 per min vs ML 64 % / 0.73) |
| **Live demo run by the user, intentional triple blinks** | Manual (user) | satisfactory for a demo | ✅ "caught almost every time" when intentional and not too soft (not scored: cooldown overlaps) |
| **10 min of normal activity** (talking, head movement, natural blinking) with the trigger armed | Manual (user) | ≤ 1 false trigger | ⏳ |
| Gesture-to-event latency | Manual | ≤ 1.5 s after the last blink | ⏳ (by design ~0.5–0.7 s after the third blink) |

### Seizure-detector socket with a mock model (2026-11-10) ✅ software, simulator

**Goal** (user): show how a seizure model plugs into the `Detector` socket. The user's requirement: train on *real*
public data, with a clear disclaimer that it isn't validated on Muse data.
- **Data search:** CHB-MIT (PhysioNet, ODC-By 1.0, scalp EEG at 256 Hz like the Muse, 198 annotated seizures) was
  chosen over Siena (CC-BY 4.0, 512 Hz, 47 seizures) and the Bonn sets (single channel, no licence stated).
- **Model:** a logistic regression on 14 features; channels are approximated Muse-like derivations from the
  bipolar montage. The model card is [SEIZURE_MODEL.md](SEIZURE_MODEL.md).

| Activity | Level | Acceptance criterion | Result |
|---|---|---|---|
| EDF reader, Muse-like channel mapping, annotation parser (both summary formats) | Unit | exact values | ✅ |
| Streaming features equal offline features (odd chunk sizes); no baseline in the first 30 s | Unit | identical | ✅ |
| Labels exclude the transition around seizures; alarm needs 5 consecutive windows; 120 s refractory | Unit | as designed | ✅ |
| Leave-one-patient-out on CHB-MIT (8 patients, 46.9 h, 28 seizures) | Offline evaluation | report sensitivity, delay, false alarms/h | **21/28 (75 %), median 20 s, 3.6/h** at threshold 0.9 (random forest: 8/28) |
| Shipped model: 14 coefficients matching the features, Muse channels, disclaimer text | Unit | as designed | ✅ |
| Detector on simulated EEG fires on the replayed CHB-MIT clip (patient chb23, not used for training) | Integration | one alarm, -10…+45 s around onset | ✅ |
| Detector ignores 12 gestures and artefacts (clench, triple blink, talking, nods, head turns, glances, eyebrows, squeeze) | Integration | no alarm | ✅ |
| Socket off by default, runs beside any gesture when on; headset without the Muse channels → never fires | Unit | as designed | ✅ |
| Event manager turns the alarm into a `seizure`-kind event | Integration | label says "mock" | ✅ |
| Dashboard switch on/off, status shows the detector with its disclaimer | System | as designed | ✅ |
| On a real Muse | – | – | **not done, and not claimed** |

### Phase 7: Setup & packaging polish (planned)

Docker was dropped (2026-11-09): the EEG bridge (Bluetooth), the microphone and the GPU all need native processes, so containers would add complexity without simplifying setup.

| Planned activity | Acceptance criterion |
|---|---|
| `bayn setup` on a clean machine (Windows; one of macOS/Linux) | deps, CUDA libs (if NVIDIA), Whisper + Ollama models, `doctor` all green, no manual steps |
| Offline-readiness check | everything needed for the demo is on disk; network-unplugged demo works |
| **Laptop disk budget** | all models in `models/` (copyable), nothing large on C:, `doctor` reports sizes + free space (✅ done early, 2026-11-09) |
| Ollama without the tray app | `bayn run` starts `ollama serve` itself with the user's model folder (✅ done early) |
| Laptop checklist (Bluetooth, mic, GPU/CPU, ports, models) | `doctor` covers each item |
| README quickstart followed literally | works without undocumented steps |

**Demo laptop, clean machine (2026-11-10):** `.\bayn setup` ran unattended (venv, CPU-only deps, Whisper small + tiny, Ollama 0.40.15 via winget, qwen3:4b) and `doctor` reported "ready", but the summary model landed outside `models\` (D27, fixed). `doctor`'s only warning: Bluetooth radio switched off.

**Demo laptop, step 6 (2026-11-10):** end to end under demo conditions (plain terminal, VS Code closed, real Muse, wireless mic): user-accepted ("the blink works well enough"). 2 triple-blink events, none reported as false, in ~5 min with the headset on; summaries en 19.7 s / ar 28.0 s in the LLM (27 / 37 s after the window closed), both accurate, nothing invented; live lines 6-10 s after speech (target 2-3 s not met; accepted). Fixes during the step: D30. Audio + Muse together for 6 min: no audio warnings. Pending: Wi-Fi-off offline run.

**Demo laptop, wrap-up (2026-11-10):** the measured setup became `auto` for CPU-only machines (summary rule ≥ 7 GB RAM, `transcription.single_encode: auto` = on for CPU, off for CUDA until the dev-PC GPU regression). Full suite on the laptop: **214 passed, 3 skipped** (headset ×2, real LLM without Ollama running). New tests: `test_transcription_single.py` (×7 incl. `test_auto_is_off_on_cuda`), `test_report_on_a_machine_without_gpu_uses_the_cpu_runs`; `test_auto_model_depends_on_gpu_and_ram` extended (7.6 GB CPU-only → gemma4:e2b, 6 GB → qwen3:4b). Not yet run on the laptop: `-m hardware`, the real-LLM test, the step 7 drills.

**Demo laptop, step 5 (2026-11-10):** `benchmark stt|mem|llm --profile(s) cpu` → docs/BENCHMARK_LAPTOP.md (D28 fixed in the report generator). New: `SingleEncodeWhisperEngine` (engine_single.py, not live) + `regress_single`: 0 decode mismatches vs the original engine on base / small / medium / turbo over 37 real utterances, 26-48 % less Whisper time (details: PROGRESS.md). +4 integration tests (`tests/test_transcription_single.py` ×3, `test_report_on_a_machine_without_gpu_uses_the_cpu_runs`).

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
| Gesture trigger: jaw clench / triple blink (double blink until 2026-11-10), configurable | detector unit tests, system test, demo run, user test | ✅ sim · ✅ clench on real headset · ⏳ triple blink on real headset (F1 → Phase 6 ML, sessions pending) |
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
| Local LLM summary of the transcript around the event + basic EEG stats | summarization unit/integration tests, system test, real-model test, demo run | ✅ (EEG part written by code) |
| Show the captured text and how it turns into the summary | dashboard ① heard → ② context + prompt → ③ live summary; system test checks prompt + streaming | ✅ · ⏳ user |
| Arabic, English, and both | `summary.language` auto/en/ar/both; model comparison per output language | ✅ (Arabic weaker, F3) |
| Easy setup on other machines (Docker dropped → native setup) | Phase 7 plan | ⏳ |
| One-command demo without hardware | system tests (6 processes), `bayn demo --audio file` run with real models | ✅ |

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
| D16 | Summary evaluation (test bug) | Every Arabic summary failed "format"; Arabic and mixed could never reach full fact coverage | Heading written "الرييسيه" vs normalised "الرئيسيه"; facts required that the live transcript had lost (cut-off ending, "shubak") | Correct heading; facts carry an `in:` list of variants whose transcript contains them; `--rescore` re-checks saved summaries | `test_quality_check_*`, `test_facts_only_required_where_the_transcript_has_them` |
| D17 | Manual demo run | Summary started but never finished; the next event's summary waited behind it | Three LLMs left loaded by model comparisons (9.6 GB) + Whisper oversubscribed the 11 GB GPU; no stall timeout | 60 s token-stall timeout → error shown, queue continues; evaluation unloads its models; `auto` needs ≥ 10 GB VRAM for 7b | `test_stalled_model_times_out_instead_of_hanging`, `test_auto_model_depends_on_gpu_memory` |
| D18 | Integration | Live summary text could stay blank until the end with a fast model | The first streamed update (no summary text yet) used up the 80 ms throttle | Empty renders don't count against the throttle | `test_service_summarises_window_streams_and_saves` |
| D19 | Benchmark (reading transcripts) | Correct transcripts scored as errors: "10-20", "40 seconds", "3.14", "Sarah" counted as lost key terms | Metrics compared number words with digits literally; a name's equally valid spelling was not accepted | Numbers normalised in words and digits, English and Arabic ("ten twenty" = "10:20" = "عشرة وعشرين"); decimals only joined across "point"/"فاصلة", so a self-correction ("3.4. 3.14") stays visible; key terms accept alternatives (`Sara\|Sarah`) | `test_numbers_score_the_same_in_words_and_digits`, `test_digit_transcripts_keep_the_key_terms` |
| D20 | Benchmark (reading summaries) | English summaries flagged "wrong language" | The "Unclear" section quotes Arabic fragments verbatim, which is its purpose | Language judged on the summary and key points only | `test_quality_check_*` |
| D21 | Benchmark (reading summaries) | A summary saying "40 years" (Whisper misheard "forty seconds") was credited with the duration fact; Arabic summaries using Whisper's Arabic-letter English terms (النيوكليوس) were not credited | Fact keywords too loose (number alone) and too strict (no transliterations) | Duration needs the unit; Arabic facts accept recognisable transliteration stems | `test_reading_the_script_back_covers_every_fact_and_nothing_forbidden` |
| D26 | Building the decision test | The rule on the learned signal fired on double blinks (6 false triggers) | Learning the rule threshold needs rest periods; the noisy session has none, so a fold trained on it got no threshold and fell back to 90 µV (blinks there are ~500 µV) | Natural-sitting activity counts as rest → 86 % caught, 0.41 false/min, and no double blink fires | decision test report |
| D27 | `bayn setup` on the clean demo laptop (2026-11-10) | Setup said "ready", but `models\ollama` didn't exist; the summary model was in `~\.ollama\models`. After a reboot (or with no tray app running) Bayn's own `ollama serve` would find no model | The Ollama installer launches its tray app; its server started before setup set `OLLAMA_MODELS`, and setup's pull went through that server into the default folder | Setup sets `OLLAMA_MODELS` / `OLLAMA_NO_CLOUD` before installing Ollama and stops the tray app the installer launched. Laptop: model moved into `models\ollama` (no re-download), checked through `local_server()` | manual (laptop) |
| D28 | `benchmark report --out docs/BENCHMARK_LAPTOP.md` on the demo laptop (2026-11-10) | The laptop report said "results are from the dev PC", sections 2 and 3 were empty and section 5 found no combination, although 192 CPU summaries were measured; section 4's speed came from one arbitrary transcript source | The report used only `gpu`-profile LLM runs for quality (CPU profiles were speed-only on the dev PC) and kept the last cached file per (profile, model) | Quality comes from `gpu` runs when there are any, else from the CPU profile (`quality_profile`); CPU speed is over all sources (`cpu_speed`). Dev-PC output unchanged | `test_report_on_a_machine_without_gpu_uses_the_cpu_runs` |
| D29 | Full suite on the demo laptop (2026-11-10) | 2 system tests failed (no clench event; calibration results without `clench`) | The test stack pins its config with env vars but not the gesture, so the laptop's config/local.yaml (`events.gesture: triple_blink`) leaked in | `running_stack` sets `BAYN_EVENTS__GESTURE=clench` → 210 passed, 3 skipped (headset ×2, real LLM) | `test_gesture_event_saved_and_downloadable`, `test_guided_calibration_with_simulator` |
| D30 | Step 6 on the demo laptop (8 GB), live mic (2026-11-10) | Transcript lines 12-35 s behind speech; Whisper 8-34 s per utterance live vs 5 s offline on the same speech | Memory: ~1-2 GB available with gemma4:e2b loaded, browser and dev tools open, so Windows paged constantly (one 22 s stall at 12 MB available). Each light Bayn service also committed ~785 MB (numpy + scipy OpenBLAS per-thread buffers), system commit 22 / 24.9 GB | `bayn run` sets `OPENBLAS_NUM_THREADS=1` (~90 MB per service); opt-in `transcription.single_encode` (on for the laptop): 2.7-3.7 s per utterance after the model load; demo with only Bayn + a light browser open | `test_service_uses_the_engine_the_config_asks_for`, replay run (PROGRESS.md) |
| D25 | Real headset sessions (user: "still not very accurate") | Rule 12/28 and ML 10/28 triple blinks caught | The blink signal was the AF7/AF8 average, but on the Muse blinks are largest on TP9/TP10 (forehead reference) and AF8 is noisy, so the average was close to the weakest view of the blinks | Blink signal learned per person from their labelled blinks; synthetic training data from their own blinks → ML 23/28, 0.81 false/min held-out | `test_learned_weights_find_the_persons_blink_pattern`, `test_trained_model_moves_the_rule_onto_the_learned_signal` |
| D24 | Switching the trigger to three blinks (simulator) | The rule missed 4–5 of 30 triple blinks | Each blink's 0.3 Hz high-pass rebound pulls the baseline down; the third blink of a fast series rose 170 µV but only reached +87 µV, under the 90 µV threshold. Two blinks overlapping without returning to baseline also merged | Inside a series, a blink counts by its rise from the dip before it (and must cross zero); a peak also ends on a 40 % dip; stray weaker blinks (< 45 % of the largest) aren't counted → 29/30 | `test_triple_blink_detection_rate`, `test_single_and_double_blinks_do_not_fire` |
| D23 | Phase 6, checking the candidate finder on real Muse windows | The ML finder saw only 37 peaks in 10 min of real EEG (natural blinking gives > 100) | Its threshold was 5 × noise; real forehead noise is 22–87 µV (simulator ~10 µV), so the threshold reached 110–430 µV, above many real blinks | 3 × noise, kept between 30 and 100 µV → 5–18 candidates per 40 s window, in line with blink rates | `test_finder_groups_a_double_blink_into_one_episode`, re-run on the real windows |
| D22 | Benchmark tool | CPU-mode RAM of every model measured as 0 MB | Ollama 0.40 runs the model in a separate `llama-server.exe`, not `ollama.exe` | Memory probe counts all `*llama*` processes | manual re-measure |
| D15 | Live mic (user) | Event transcript lost the end of the last sentence ("يبرد فوق في السما") | A pause > 0.6 s just before +30 s split the sentence; the second half started after the window end | Window transcript includes speech continuing with < 1.5 s pause, up to `event_tail_s` (4 s) | `test_event_transcript_finishes_the_sentence_in_progress` |
| D14 | Unit (test bug) | Timestamp continuity check failed for the file source | 1e-9 s tolerance is below double precision at unix ~1.8e9 s (~2e-7 s) | 1e-6 s tolerance | `test_stitch_and_file_source` |

### Findings (performance limitations, not bugs)

| ID | Found by | Finding | Likely cause | Plan |
|---|---|---|---|---|
| F4 | Model benchmark 2026-11-10 | In mixed speech, Whisper **hears** the English key terms but usually writes them **in Arabic letters** (gold foil → الجولد فويل, nucleus → النيوكليوس, school nurse → سكول نورس). Only 5–25 % stay in English letters, whatever the strategy (`arabic_lean`, style prompt, forced Arabic: ≤ +1 term in 5) | An utterance is decoded as Arabic, and Whisper spells foreign words phonetically | **Accepted.** The summary LLM matters more: Gemma 4 understands the transliterations (82 % of facts on mixed speech vs 66 % for qwen2.5:7b). large-v3 keeps the most English letters (25 %); worth a look on the laptop if it is fast enough |
| F5 | Real-model test 2026-11-10 | Gemma 4 E2B mentioned the user's aside before the reading ("Black Clover") in a summary; qwen2.5:7b had left it out | The window includes 10 s before the trigger | **Not a demo goal** (user, 2026-11-10): the aside isn't representative classroom chatter. Removed from the forbidden list |
| F3 | Summary evaluation 2026-11-09 | Summaries are faithful but imperfect: Arabic output covers fewer facts (68 % vs 96 % English with qwen2.5:7b) because it is more abstract, not wrong. Small embellishments happen ("or dropping of his head"). The "Unclear" list is almost never used, even when the transcript has obvious recognition errors. qwen2.5:3b (CPU / small GPUs) covers only ~50 % | Small local models; noisy transcripts | Accepted for the prototype. More scenarios (user's new scripts) and a larger model where hardware allows. Always shown with "automatic transcript, may contain errors" |
| F2 | Live mic, user test 2026-11-09 | Mixed Arabic/English **inside** a sentence: 39.7 % WER. English-decoded utterances **translate or transliterate the Arabic** ("فاكرين إيش صار لها" → "Do you think what happened?", شباب → "shabab"); Arabic-decoded ones mostly keep the English words | `mixed` forces one language per utterance; Whisper handles intra-sentence code-switching poorly when forced to English | **Current limitation, accepted for now.** The test script was very informal and heavily mixed. The realistic case is Arabic speech with **occasional English key terms** (e.g. "pie", "Rutherford"), which **must not be lost**. To be measured in the transcription benchmark (§4, "Later"). Candidate fixes are already in the code but off by default: `mixed_strategy: arabic_lean`, `initial_prompt: auto` |
| F1 | Hardware, user test 2026-11-09 | Rule-based double blink: head movements cause false triggers; natural-strength double blinks are missed | Movement artefacts and blinks overlap in amplitude on AF7/AF8; calibration "rest" includes natural blinks, which pushes the threshold up | Phase 6: trigger changed to a triple blink; EEG-only ML detector built and verified in simulation (fewer false triggers than the rule with ~20 min of training data); real-headset sessions pending. Until it is trained on the user's data, **jaw clench** is the recommended trigger |

### Environment issues (not software defects)

| ID | Issue | Resolution |
|---|---|---|
| E1 | Dev PC's built-in Bluetooth (QCA61x4A Wi-Fi/BT card, no antennas attached) saw 1 BLE device and no Muse | USB Bluetooth dongle |
| E2 | Dongle (Actions ATS2851) showed Device Manager Code 31 on Windows 10 | Vendor driver; built-in adapter disabled (only one active radio). Documented in the README |
| E4 | Dev PC's C: drive down to 5.2 GB free: Ollama models (8 GB) + pip cache (1.5 GB) on C: | Ollama models moved to `models/ollama` (`OLLAMA_MODELS`), pip cache cleared, rejected llama3.2:3b deleted → 14.7 GB free. Setup now does this by default; `doctor` checks disk space |
| E5 | Ollama tray app stopped starting its server after the move (server log empty) | `bayn run` / `bayn models` start `ollama serve` themselves; Ollama 0.40 had cloud features on by default → `OLLAMA_NO_CLOUD=1` |
| E6 | `ollama list` with no server running started the Ollama **tray app**, which ignored the model folder (stale environment) and failed to start its server | Never call the `ollama` CLI from Bayn; `local_server()` starts `ollama serve` with the right environment for every tool (run, models, eval, benchmark) |
| E7 | Ollama's `/api/ps` reports Gemma 4 E2B/E4B as 0.2 GB, but they really use 3.0 / 4.8 GB of GPU memory | The benchmark measures memory from outside (GPU driver, process RAM) |
| E3 | Muse powers itself off when idle and unconnected, so a test run 25 min later found nothing | Turn it on right before testing; `doctor` shows whether it is advertising |
| E8 | Demo laptop (i7-1355U, 15 W): Whisper speed doesn't change with 4, 6, 10 or 12 threads; clock ~2.0-2.4 GHz under load with the MyASUS fan profile already on "Performance" | Power-limited chip: speed has to come from doing less work (one encoder pass per utterance, `engine_single.py`), not from threads |
| E9 | faster-whisper retries a greedy decode that fails its quality checks with random sampling (temperature fallback): on garbled Arabic (base / medium) the original engine gave 3 different texts in 3 runs of the same utterance | Engine comparisons re-decode differing utterances at temperature 0 (`regress_single.greedy`); live app unchanged |

---

## 7. Known limitations / not yet tested

- Automatic reconnect after a real headset drop-out: implemented and unit-tested via the clock and gap logic, **not yet exercised with the real headset** (manual test pending).
- Signal-quality thresholds are initial values. They get re-tuned with real data in Phase 6.
- Gesture detection performance is so far measured on **simulated** gestures only. The simulator's clench and blink shapes are educated guesses; the real-headset numbers come from the Phase 6 protocol.
- Audio feedback needs one click on the page first (browser autoplay rule). The 🔇/🔔 button does that.
- **Code-switching inside a sentence (F2)** is unreliable: Arabic words can be translated or transliterated when a sentence is decoded as English. Sentence-level switching works well (14/14 on the demo clip). English key terms inside Arabic speech are mostly written in Arabic letters (F4); the summary model still understands most of them.
- Model benchmark numbers come from the dev PC and one good microphone; the demo laptop and its microphone are re-tested before the hackathon.
- Arabic dialect (Hejazi) words are sometimes misspelled or merged (≈ 12 % CER). Modern Standard Arabic is better.
- Hardware tests have only run on the dev PC. They must be repeated on the demo laptop.
- The dashboard UI has no automated browser tests. It is checked manually.
