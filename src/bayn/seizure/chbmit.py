"""CHB-MIT Scalp EEG Database (PhysioNet): download a subset, read it, map it to Muse-like channels.

Guttag, J. (2010). CHB-MIT Scalp EEG Database (version 1.0.0). PhysioNet. https://doi.org/10.13026/C2K01R
Shoeb, A. (2009). Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment. MIT PhD thesis.
Goldberger et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. Circulation 101(23):e215-e220.
License: Open Data Commons Attribution License v1.0 (use with attribution).

The recordings use a bipolar "double banana" montage at 256 Hz (the Muse also samples at 256 Hz). The Muse
records TP9, AF7, AF8, TP10 against a forehead reference (Fpz), so we build four APPROXIMATE virtual channels
from the bipolar pairs (FP1/FP2 stand in for the forehead reference, T7/T8 for TP9/TP10):
    AF7  ~ -(FP1-F7) / 2           AF8  ~ -(FP2-F8) / 2
    TP9  ~ -(FP1-F7) - (F7-T7)     TP10 ~ -(FP2-F8) - (F8-T8)
Only used to train / evaluate the mock model and to cut a demo clip; nothing here runs at demo time.

    python -m bayn.seizure.chbmit download      # ~1.3 GB into data/datasets/chbmit (gitignored)
"""

from __future__ import annotations

import re
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bayn.common.config import DATA_DIR

BASE_URL = "https://physionet.org/files/chbmit/1.0.0"
DATASET_DIR = DATA_DIR / "datasets" / "chbmit"
CHANNELS = ["TP9", "AF7", "AF8", "TP10"]           # Muse order
# patients used (children / adolescents with intractable seizures); a few seizure files + one without
PATIENTS = ["chb01", "chb02", "chb03", "chb05", "chb06", "chb08", "chb10", "chb20"]
SEIZURE_FILES_PER_PATIENT = 3
CLEAN_FILES_PER_PATIENT = 1


# ---------------------------------------------------------------- EDF (European Data Format) reader
@dataclass
class EDF:
    labels: list[str]
    fs: float
    data: np.ndarray            # (n_signals, n_samples) physical units (µV)


def read_edf(path: Path) -> EDF:
    """Minimal EDF reader (16-bit, equal sampling rate per signal, as in CHB-MIT)."""
    raw = Path(path).read_bytes()
    n_header = int(raw[184:192].decode().strip())
    n_records = int(raw[236:244].decode().strip())
    duration = float(raw[244:252].decode().strip())
    ns = int(raw[252:256].decode().strip())
    h = raw[256:n_header]

    def field_(width: int, offset: int) -> list[str]:
        return [h[offset + i * width: offset + (i + 1) * width].decode("latin-1").strip() for i in range(ns)]

    off = 0
    labels = field_(16, off); off += 16 * ns
    off += 80 * ns + 8 * ns                                    # transducer, physical dimension
    pmin = np.array(field_(8, off), float); off += 8 * ns
    pmax = np.array(field_(8, off), float); off += 8 * ns
    dmin = np.array(field_(8, off), float); off += 8 * ns
    dmax = np.array(field_(8, off), float); off += 8 * ns
    off += 80 * ns                                             # prefiltering
    nsamp = np.array(field_(8, off), int)
    if len(set(nsamp)) != 1:
        raise ValueError("signals with different sampling rates are not supported")
    spr = int(nsamp[0])
    d = np.frombuffer(raw[n_header:n_header + n_records * ns * spr * 2], dtype="<i2")
    d = d.reshape(n_records, ns, spr).transpose(1, 0, 2).reshape(ns, -1).astype(np.float32)
    gain = (pmax - pmin) / np.where(dmax - dmin == 0, 1, dmax - dmin)
    d = (d - dmin[:, None]) * gain[:, None] + pmin[:, None]
    return EDF(labels, spr / duration, d)


def muse_like(edf: EDF) -> np.ndarray:
    """(4, n) virtual TP9, AF7, AF8, TP10 from the bipolar montage (see module doc). Raises if pairs are missing."""
    idx = {}
    for i, lab in enumerate(edf.labels):
        idx.setdefault(lab.upper().replace(" ", ""), i)       # first occurrence (some files repeat T8-P8)

    def ch(name: str) -> np.ndarray:
        if name not in idx:
            raise KeyError(f"channel {name} missing")
        return edf.data[idx[name]]

    fp1f7, f7t7, fp2f8, f8t8 = ch("FP1-F7"), ch("F7-T7"), ch("FP2-F8"), ch("F8-T8")
    return np.vstack([-(fp1f7 + f7t7), -fp1f7 / 2, -fp2f8 / 2, -(fp2f8 + f8t8)]).astype(np.float32)


# ---------------------------------------------------------------- seizure annotations (summary files)
@dataclass
class FileInfo:
    name: str
    seizures: list[tuple[float, float]] = field(default_factory=list)   # (start s, end s)


def parse_summary(text: str) -> list[FileInfo]:
    files, cur = [], None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("File Name:"):
            cur = FileInfo(line.split(":", 1)[1].strip())
            files.append(cur)
            start = None
        elif cur is not None and re.match(r"Seizure( \d+)? Start Time:", line):
            start = float(re.findall(r"(\d+)\s*seconds", line)[0])
        elif cur is not None and re.match(r"Seizure( \d+)? End Time:", line):
            cur.seizures.append((start, float(re.findall(r"(\d+)\s*seconds", line)[0])))
    return files


def _fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "bayn-research-download"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)


def subset(patient: str) -> list[FileInfo]:
    """The files we use for one patient: the first few with seizures, plus clean ones."""
    files = parse_summary((DATASET_DIR / patient / f"{patient}-summary.txt").read_text(encoding="latin-1"))
    with_sz = [f for f in files if f.seizures][:SEIZURE_FILES_PER_PATIENT]
    clean = [f for f in files if not f.seizures][:CLEAN_FILES_PER_PATIENT]
    return with_sz + clean


def download(patients: list[str] = PATIENTS) -> None:
    for p in patients:
        _fetch(f"{BASE_URL}/{p}/{p}-summary.txt", DATASET_DIR / p / f"{p}-summary.txt")
        for f in subset(p):
            print(f"  {p}/{f.name} ({len(f.seizures)} seizure(s))", flush=True)
            _fetch(f"{BASE_URL}/{p}/{f.name}", DATASET_DIR / p / f.name)
    print(f"done -> {DATASET_DIR}")


def load_patient(patient: str) -> list[tuple[str, np.ndarray, float, list[tuple[float, float]]]]:
    """[(file, muse-like data (4, n), fs, seizures)] for the downloaded subset of one patient."""
    out = []
    for f in subset(patient):
        path = DATASET_DIR / patient / f.name
        if not path.exists():
            continue
        edf = read_edf(path)
        try:
            out.append((f.name, muse_like(edf), edf.fs, f.seizures))
        except KeyError as e:                     # montage change in a few files
            print(f"  skip {f.name}: {e}")
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["download"]:
        download()
    else:
        print(__doc__)
