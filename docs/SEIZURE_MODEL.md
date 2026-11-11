# Mock seizure model (model card)

> [!WARNING]
> **Not validated on real Muse data. Not a medical device. Not for any clinical use.** This model only shows how a
> real seizure detector would plug into Bayn's detector socket. It was trained on public clinical scalp EEG from
> children (CHB-MIT), with four channels *approximated* to look like a Muse. It has never been tested on Muse
> recordings of seizures. Any real use would need its own data, clinical validation and regulatory clearance.

## What it is

| | |
|---|---|
| Purpose | Demonstrate the `Detector` socket: a model's alarm creates an event, the EEG window is saved, and the transcript and summary follow, exactly as for a gesture |
| Not an absence-seizure detector | Bayn targets **absence seizures** (brief staring spells, classically with generalised ~3 Hz spike-and-wave EEG). CHB-MIT does not label seizure types and was not collected for absence seizures, so this model is **not** an absence-seizure detector: a real one needs its own data, ideally recorded with the target headset |
| Training data | **CHB-MIT Scalp EEG Database** (PhysioNet): 8 patients (chb01, chb02, chb03, chb05, chb06, chb08, chb10, chb20), 46.9 h, 28 seizures |
| Channels | TP9, AF7, AF8, TP10 against a forehead reference, approximated from the bipolar montage: AF7 ≈ −(FP1−F7)/2, TP9 ≈ −(FP1−F7)−(F7−T7), mirrored on the right |
| Features | 14 per 4 s window, every 1 s: change in line length and power against the person's own baseline 30–180 s earlier, relative delta/theta/alpha/beta power, dominant frequency, rhythmicity, Hjorth parameters, synchrony between channels ([features.py](../src/bayn/seizure/features.py)) |
| Model | Logistic regression (14 coefficients, [model.json](../src/bayn/seizure/assets/model.json)). Alarm when P ≥ 0.9 for 5 windows in a row (~5 s), then quiet for 120 s |
| Demo clip | A real seizure from patient chb23 (**not** used for training), 75 s of EEG before the onset included, Muse-like channels. The simulator's "Seizure (CHB-MIT clip)" button replays it |

## How well it does (on CHB-MIT, not on Muse)

Leave one patient out: every patient is scored by a model trained on the others. A seizure counts as detected if
an alarm fires between 10 s before its onset and 30 s after its end; every other alarm is a false alarm.

| Threshold | Seizures detected | Median delay after onset | False alarms per hour |
|---|---|---|---|
| 0.5 | 22/28 (79 %) | 12.5 s | 12.78 |
| 0.7 | 19/28 (68 %) | 14.0 s | 7.63 |
| 0.8 | 21/28 (75 %) | 15.0 s | 5.73 |
| 0.9 | 21/28 (75 %) | 20.0 s | 3.58 |

Shipped threshold **0.9**, picked from this table (so these numbers are slightly optimistic). For comparison, a random forest scored 8/28 detected
and 0.36 false alarms/h at the same threshold (logistic regression: 21/28,
3.58/h). The simpler, transparent model was kept.

Per patient (threshold 0.8, for reference):

| Patient | Detected | Median delay (s) | False alarms / h |
|---|---|---|---|
| chb01 | 3/3 | 10.0 | 2.0 |
| chb02 | 2/3 | 65.0 | 12.26 |
| chb03 | 2/3 | 62.0 | 13.26 |
| chb05 | 3/3 | 25.0 | 10.75 |
| chb06 | 2/6 | 22.0 | 2.1 |
| chb08 | 3/3 | 15.0 | 6.01 |
| chb10 | 2/3 | 7.0 | 7.37 |
| chb20 | 4/4 | 13.5 | 2.25 |

These numbers describe the model on the data it was built for. They say **nothing** about Muse recordings,
adults, or seizure types not in this subset. On the Muse, eye blinks, jaw clenches and head movement look nothing
like these hospital recordings; the detector is off by default and switched on only for the demo.

## Data and licence

Guttag, J. (2010). *CHB-MIT Scalp EEG Database* (version 1.0.0). PhysioNet. https://doi.org/10.13026/C2K01R.
Shoeb, A. (2009). *Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment.* PhD thesis, MIT.
Goldberger, A. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23):e215–e220.
Licence: Open Data Commons Attribution License v1.0. The demo clip is derived from this database.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m bayn.seizure.chbmit download    # ~1.3 GB, once
.\.venv\Scripts\python.exe -m bayn.seizure.train --clip
```

Generated 2026-11-10 by `python -m bayn.seizure.train`.
