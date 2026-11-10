"""Train + evaluate the MOCK seizure model on CHB-MIT, and cut the demo clip.

    python -m bayn.seizure.chbmit download     # once, ~1.3 GB (data/datasets/chbmit, gitignored)
    python -m bayn.seizure.train               # leave-one-patient-out evaluation, then train on all and
                                               # write src/bayn/seizure/assets/model.json + docs/SEIZURE_MODEL.md
    python -m bayn.seizure.train --clip        # also cut the demo clip from a patient NOT used for training

The shipped model is a logistic regression (JSON coefficients: transparent, no pickle). It is a demonstration
of the detector socket, trained on children's clinical scalp EEG with approximated Muse-like channels; it is
NOT validated on Muse data and must not be used for any clinical purpose.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from bayn.seizure import chbmit
from bayn.seizure.features import FEATURES, STEP_S, WINDOW_S, recording_features

ASSETS = Path(__file__).parent / "assets"
MODEL_PATH = ASSETS / "model.json"
CLIP_PATH = ASSETS / "chbmit_clip.npz"
CARD_PATH = Path(__file__).resolve().parents[3] / "docs" / "SEIZURE_MODEL.md"
PRE_EXCLUDE_S, POST_EXCLUDE_S = 60.0, 120.0     # transition zones around seizures: not used for training
NEG_STRIDE = 5                                  # keep every 5th non-seizure window for training
CONSECUTIVE = 5                                 # alarm: P >= threshold for 5 windows in a row (~5 s)
REFRACTORY_S = 120.0
THRESHOLDS = (0.5, 0.7, 0.8, 0.9)
CLIP_PATIENT = "chb23"


def labels_for(ends: np.ndarray, seizures: list[tuple[float, float]]) -> np.ndarray:
    """1 = seizure, 0 = not, -1 = excluded (transition). Window = [end - 4 s, end]; centre decides."""
    centre = ends - WINDOW_S / 2
    y = np.zeros(len(ends), int)
    for a, b in seizures:
        y[(centre >= a - PRE_EXCLUDE_S) & (centre < a)] = -1
        y[(centre > b) & (centre <= b + POST_EXCLUDE_S)] = -1
    for a, b in seizures:
        y[(centre >= a) & (centre <= b)] = 1
    return y


def load_features(patients: list[str]) -> dict[str, list[dict]]:
    out = {}
    for p in patients:
        recs = []
        for name, data, fs, seizures in chbmit.load_patient(p):
            X, ends = recording_features(data, fs)
            recs.append({"file": name, "X": X, "t": ends, "y": labels_for(ends, seizures), "seizures": seizures})
        out[p] = recs
        print(f"  {p}: {len(recs)} files, {sum(len(r['y']) for r in recs) / 3600:.1f} h, "
              f"{sum(len(r['seizures']) for r in recs)} seizures", flush=True)
    return out


def train_rows(recs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for r in recs:
        pos = r["y"] == 1
        neg = np.flatnonzero(r["y"] == 0)[::NEG_STRIDE]
        X += [r["X"][pos], r["X"][neg]]
        y += [np.ones(pos.sum()), np.zeros(len(neg))]
    return np.vstack(X), np.concatenate(y)


def fit_logistic(X: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.linear_model import LogisticRegression

    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-9
    m = LogisticRegression(class_weight="balanced", C=0.5, max_iter=2000).fit((X - mean) / std, y)
    return {"type": "logistic", "mean": mean.tolist(), "std": std.tolist(), "coef": m.coef_[0].tolist(),
            "intercept": float(m.intercept_[0])}


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    if model["type"] == "logistic":
        z = ((X - np.array(model["mean"])) / np.array(model["std"])) @ np.array(model["coef"]) + model["intercept"]
        return 1 / (1 + np.exp(-z))
    return model["sk"].predict_proba(X)[:, 1]


def fit_forest(X: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.ensemble import RandomForestClassifier

    return {"type": "forest", "sk": RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                                           class_weight="balanced", random_state=0,
                                                           n_jobs=-1).fit(X, y)}


def alarms(p: np.ndarray, t: np.ndarray, threshold: float) -> list[float]:
    """Live alarm logic: P >= threshold for CONSECUTIVE windows; then quiet for REFRACTORY_S."""
    out, run, last = [], 0, -1e9
    for pi, ti in zip(p, t):
        run = run + 1 if pi >= threshold else 0
        if run >= CONSECUTIVE and ti - last >= REFRACTORY_S:
            out.append(float(ti))
            last = ti
    return out


def event_metrics(recs: list[dict], probs: list[np.ndarray], threshold: float) -> dict:
    n_sz = det = 0
    latency, false, hours = [], 0, 0.0
    for r, p in zip(recs, probs):
        hours += len(r["t"]) * STEP_S / 3600
        al = alarms(p, r["t"], threshold)
        for a, b in r["seizures"]:
            n_sz += 1
            hit = [x for x in al if a - 10 <= x <= b + 30]
            if hit:
                det += 1
                latency.append(hit[0] - a)
        false += sum(1 for x in al if not any(a - 10 <= x <= b + PRE_EXCLUDE_S for a, b in r["seizures"]))
    return {"seizures": n_sz, "detected": det, "sensitivity": round(det / n_sz, 3) if n_sz else None,
            "median_latency_s": round(float(np.median(latency)), 1) if latency else None,
            "false_alarms": false, "hours": round(hours, 1),
            "false_per_hour": round(false / hours, 2) if hours else None}


def lopo(data: dict[str, list[dict]], fitter) -> dict:
    """Leave one patient out: every patient scored by a model that never saw them."""
    per_thr = {thr: [] for thr in THRESHOLDS}
    per_patient = {}
    for test in data:
        X, y = train_rows([r for p, recs in data.items() if p != test for r in recs])
        m = fitter(X, y)
        probs = [predict(m, r["X"]) for r in data[test]]
        for thr in THRESHOLDS:
            per_thr[thr].append((data[test], probs))
        per_patient[test] = event_metrics(data[test], probs, 0.8)
    summary = {}
    for thr, parts in per_thr.items():
        recs = [r for recs, _ in parts for r in recs]
        probs = [p for _, ps in parts for p in ps]
        summary[str(thr)] = event_metrics(recs, probs, thr)
    return {"by_threshold": summary, "by_patient_at_0.8": per_patient}


def cut_clip() -> None:
    """A real seizure from a patient NOT used for training: 75 s before onset + the seizure + 15 s, as
    Muse-like channels (float16, ~200 kB). The simulator replays it ("Seizure" button)."""
    chbmit._fetch(f"{chbmit.BASE_URL}/{CLIP_PATIENT}/{CLIP_PATIENT}-summary.txt",
                  chbmit.DATASET_DIR / CLIP_PATIENT / f"{CLIP_PATIENT}-summary.txt")
    f = next(x for x in chbmit.subset(CLIP_PATIENT) if x.seizures)
    chbmit._fetch(f"{chbmit.BASE_URL}/{CLIP_PATIENT}/{f.name}", chbmit.DATASET_DIR / CLIP_PATIENT / f.name)
    edf = chbmit.read_edf(chbmit.DATASET_DIR / CLIP_PATIENT / f.name)
    m = chbmit.muse_like(edf)
    a, b = f.seizures[0]
    i0, i1 = int((a - 75) * edf.fs), int((min(b, a + 60) + 15) * edf.fs)
    ASSETS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CLIP_PATH, data=m[:, i0:i1].astype(np.float16), fs=edf.fs, onset_s=75.0,
                        seizure_s=float(min(b, a + 60) - a), source=f"CHB-MIT {CLIP_PATIENT}/{f.name} "
                        f"seizure at {a:.0f}-{b:.0f} s (ODC-By 1.0; Guttag 2010, doi:10.13026/C2K01R)")
    print(f"clip: {CLIP_PATH} ({(i1 - i0) / edf.fs:.0f} s, seizure from 75 s)")


def model_card(report: dict) -> str:
    lr = report["logistic"]["by_threshold"][str(report["threshold"])]
    rf = report["forest"]["by_threshold"][str(report["threshold"])]
    rows = "\n".join(f"| {p} | {m['detected']}/{m['seizures']} | {m['median_latency_s']} | {m['false_per_hour']} |"
                     for p, m in report["logistic"]["by_patient_at_0.8"].items())
    thr_rows = "\n".join(f"| {t} | {m['detected']}/{m['seizures']} ({(m['sensitivity'] or 0) * 100:.0f} %) | "
                         f"{m['median_latency_s']} s | {m['false_per_hour']} |"
                         for t, m in report["logistic"]["by_threshold"].items())
    return f"""# Mock seizure model (model card)

> [!WARNING]
> **Not validated on real Muse data. Not a medical device. Not for any clinical use.** This model only shows how a
> real seizure detector would plug into Bayn's detector socket. It was trained on public clinical scalp EEG from
> children (CHB-MIT), with four channels *approximated* to look like a Muse. It has never been tested on Muse
> recordings of seizures. Any real use would need its own data, clinical validation and regulatory clearance.

## What it is

| | |
|---|---|
| Purpose | Demonstrate the `Detector` socket: a model's alarm creates an event, the EEG window is saved, and the transcript and summary follow, exactly as for a gesture |
| Training data | **CHB-MIT Scalp EEG Database** (PhysioNet): {len(report['patients'])} patients ({', '.join(report['patients'])}), {report['hours']} h, {report['seizures']} seizures |
| Channels | TP9, AF7, AF8, TP10 against a forehead reference, approximated from the bipolar montage: AF7 ≈ −(FP1−F7)/2, TP9 ≈ −(FP1−F7)−(F7−T7), mirrored on the right |
| Features | 14 per 4 s window, every 1 s: change in line length and power against the person's own baseline 30–180 s earlier, relative delta/theta/alpha/beta power, dominant frequency, rhythmicity, Hjorth parameters, synchrony between channels ([features.py](../src/bayn/seizure/features.py)) |
| Model | Logistic regression (14 coefficients, [model.json](../src/bayn/seizure/assets/model.json)). Alarm when P ≥ {report['threshold']} for {CONSECUTIVE} windows in a row (~{CONSECUTIVE} s), then quiet for {REFRACTORY_S:.0f} s |
| Demo clip | A real seizure from patient {CLIP_PATIENT} (**not** used for training), 75 s of EEG before the onset included, Muse-like channels. The simulator's "Seizure (CHB-MIT clip)" button replays it |

## How well it does (on CHB-MIT, not on Muse)

Leave one patient out: every patient is scored by a model trained on the others. A seizure counts as detected if
an alarm fires between 10 s before its onset and 30 s after its end; every other alarm is a false alarm.

| Threshold | Seizures detected | Median delay after onset | False alarms per hour |
|---|---|---|---|
{thr_rows}

Shipped threshold **{report['threshold']}**, picked from this table (so these numbers are slightly optimistic). For comparison, a random forest scored {rf['detected']}/{rf['seizures']} detected
and {rf['false_per_hour']} false alarms/h at the same threshold (logistic regression: {lr['detected']}/{lr['seizures']},
{lr['false_per_hour']}/h). The simpler, transparent model was kept.

Per patient (threshold 0.8, for reference):

| Patient | Detected | Median delay (s) | False alarms / h |
|---|---|---|---|
{rows}

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
.\\.venv\\Scripts\\python.exe -m bayn.seizure.chbmit download    # ~1.3 GB, once
.\\.venv\\Scripts\\python.exe -m bayn.seizure.train --clip
```

Generated {time.strftime('%Y-%m-%d')} by `python -m bayn.seizure.train`.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", action="store_true", help="also cut the demo clip (downloads one file of chb23)")
    ap.add_argument("--threshold", type=float, default=0.9,
                    help="0.9: same seizures as 0.8 with fewer false alarms in the leave-one-patient-out run")
    args = ap.parse_args(argv)
    patients = [p for p in chbmit.PATIENTS if (chbmit.DATASET_DIR / p / f"{p}-summary.txt").exists()]
    if len(patients) < 3:
        print("Download CHB-MIT first: python -m bayn.seizure.chbmit download")
        return 1
    print("features...", flush=True)
    data = load_features(patients)
    print("leave-one-patient-out...", flush=True)
    report = {"patients": patients, "threshold": args.threshold, "features": list(FEATURES),
              "hours": round(sum(len(r["y"]) for recs in data.values() for r in recs) * STEP_S / 3600, 1),
              "seizures": sum(len(r["seizures"]) for recs in data.values() for r in recs),
              "logistic": lopo(data, fit_logistic), "forest": lopo(data, fit_forest)}
    for name in ("logistic", "forest"):
        for thr, m in report[name]["by_threshold"].items():
            print(f"  {name:8s} thr {thr}: {m['detected']}/{m['seizures']} detected, "
                  f"delay {m['median_latency_s']} s, {m['false_per_hour']} false/h")
    X, y = train_rows([r for recs in data.values() for r in recs])
    model = fit_logistic(X, y)
    model.update({"features": list(FEATURES), "threshold": args.threshold, "consecutive": CONSECUTIVE,
                  "refractory_s": REFRACTORY_S, "window_s": WINDOW_S, "step_s": STEP_S, "fs": 256.0,
                  "channels": chbmit.CHANNELS, "trained_on": f"CHB-MIT {', '.join(patients)}",
                  "evaluation": report["logistic"]["by_threshold"][str(args.threshold)],
                  "disclaimer": "MOCK model: trained on CHB-MIT (children, clinical EEG, approximated Muse-like "
                                "channels). Not validated on Muse data. Not for clinical use."})
    ASSETS.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(model, indent=1), encoding="utf-8")
    if args.clip:
        cut_clip()
    CARD_PATH.write_text(model_card(report), encoding="utf-8")
    print(f"model: {MODEL_PATH}\ncard: {CARD_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
