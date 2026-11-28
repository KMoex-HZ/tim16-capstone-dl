"""Pembanding non-DL tim: fitur statistik + FFT (RMS, puncak, kurtosis, energi pita) + Random Forest.

    python -m src.pembanding.baseline_nondl --seed 1

Memakai split dan praproses dari src.protokol (kanal tunggal bus_mV), lalu menulis
results/baseline/metrics.json lewat src.common.evaluate (jalur yang sama dengan model anggota).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier

from . import manifest
from src.protokol import data

AKAR = manifest.AKAR
FS = data.LAJU_HZ
PITA = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 70), (70, 100)]   # Hz


def fitur(X: np.ndarray) -> np.ndarray:
    """X: [n, 1, 400] -> [n, jumlah fitur]."""
    x = X[:, 0, :].astype(np.float64)
    rms = np.sqrt((x ** 2).mean(1))
    puncak = np.abs(x).max(1)
    f = [x.mean(1), x.std(1), rms, puncak, np.ptp(x, axis=1), puncak / (rms + 1e-9),
         kurtosis(x, axis=1), skew(x, axis=1)]
    sp = np.abs(np.fft.rfft(x - x.mean(1, keepdims=True), axis=1)) ** 2
    fr = np.fft.rfftfreq(x.shape[1], 1 / FS)
    tot = sp.sum(1) + 1e-12
    for lo, hi in PITA:
        f.append(sp[:, (fr >= lo) & (fr < hi)].sum(1) / tot)        # energi pita relatif
    f.append(np.log10(tot))                                           # energi total (log)
    f.append(fr[sp.argmax(1)])                                        # frekuensi dominan
    f.append((sp * fr).sum(1) / tot)                                  # sentroid spektral
    return np.nan_to_num(np.stack(f, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()

    t0 = time.time()
    Xtr, ytr, Xva, yva, _, _ = data.muat()
    rf = RandomForestClassifier(n_estimators=300, random_state=a.seed, n_jobs=-1)
    rf.fit(fitur(Xtr), ytr)
    pred = rf.predict(fitur(Xva)).astype(np.int64)
    durasi = time.time() - t0

    ringkas = data.hitung_f1_alat(yva, pred)
    print(f"F1 per alat (validasi): {ringkas['val_f1_alat']} | Rinci: {ringkas['val_f1_rinci']}")

    out = AKAR / "results" / "baseline"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", pred)
    subprocess.run([sys.executable, "-m", "src.common.evaluate", "--nim", "baseline", "--model", "baseline",
                    "--seed", str(a.seed), "--pred-val", str(out / "pred_val.npy"),
                    "--durasi", f"{durasi:.1f}", "--perangkat", a.perangkat], check=True, cwd=AKAR)


if __name__ == "__main__":
    main()
