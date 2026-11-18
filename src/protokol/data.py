"""Pemuat data kasus tim 16 (NILM mini) — MILIK TIM, dibekukan setelah disetujui pembimbing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import manifest, split

TUGAS = "klasifikasi"
UKURAN_JENDELA = 400       # 2 detik pada 200 Hz
BUANG_AWAL_DETIK = 1.0     # 1 detik pertama tiap kondisi dibuang
NAMA_ALAT = ("kipas", "pompa", "lampu")


def ke_biner(y_kelas: np.ndarray) -> np.ndarray:
    """Konversi kelas 0..7 ke matriks biner (N, 3) untuk [kipas, pompa, lampu]."""
    y = np.asarray(y_kelas, dtype=np.int64).ravel()
    return np.stack([(y >> 2) & 1, (y >> 1) & 1, y & 1], axis=1)


def dari_biner(y_biner: np.ndarray) -> np.ndarray:
    """Konversi prediksi biner (N, 3) kembali ke label bulat 0..7."""
    b = (np.asarray(y_biner) > 0).astype(np.int64)
    return b[:, 0] * 4 + b[:, 1] * 2 + b[:, 2]


def _potong_jendela(
    df_label: pd.DataFrame,
    t_daya: np.ndarray,
    val_daya: np.ndarray,
    ulangan_set: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    X_list, y_list = [], []
    for _, row in df_label.iterrows():
        if int(row["ulangan"]) not in ulangan_set:
            continue
        t0 = float(row["t_mulai_host"]) + BUANG_AWAL_DETIK
        t1 = float(row["t_selesai_host"])
        i0 = int(np.searchsorted(t_daya, t0, side="left"))
        i1 = int(np.searchsorted(t_daya, t1, side="right"))
        n_jendela = (i1 - i0) // UKURAN_JENDELA
        if n_jendela <= 0:
            continue

        segmen = val_daya[i0 : i0 + n_jendela * UKURAN_JENDELA].reshape(
            n_jendela, UKURAN_JENDELA, 2
        )
        k = 1 if float(row["kipas"]) > 0 else 0
        p = 1 if float(row["pompa"]) > 0 else 0
        l = 1 if float(row["lampu"]) > 0 else 0
        kelas = k * 4 + p * 2 + l

        X_list.append(segmen)
        y_list.append(np.full(n_jendela, kelas, dtype=np.int64))

    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def muat() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    manifest.baca()
    df_label = pd.read_csv(manifest.AKAR / "data" / "label.csv")
    df_daya = (
        pd.read_csv(manifest.AKAR / "data" / "daya.csv")
        .sort_values("t_host")
        .reset_index(drop=True)
    )

    t_daya = df_daya["t_host"].to_numpy(dtype=np.float64)
    val_daya = df_daya[["arus_mA", "bus_mV"]].to_numpy(dtype=np.float32)

    # Split per grup ulangan (15 ulangan -> 9 train, 3 val, 3 test) memakai src.common.split
    ulangan_unik = np.array(sorted(df_label["ulangan"].unique()), dtype=np.int64)
    idx_sisa, idx_test = split.indeks(len(ulangan_unik), porsi_val=0.2, seed=split.SEED_SPLIT)
    idx_tr, idx_va = split.indeks(len(idx_sisa), porsi_val=0.25, seed=split.SEED_SPLIT)

    u_train = set(ulangan_unik[idx_sisa[idx_tr]].tolist())
    u_val = set(ulangan_unik[idx_sisa[idx_va]].tolist())
    u_test = set(ulangan_unik[idx_test].tolist())

    X_train, y_train = _potong_jendela(df_label, t_daya, val_daya, u_train)
    X_val, y_val = _potong_jendela(df_label, t_daya, val_daya, u_val)
    X_test, y_test = _potong_jendela(df_label, t_daya, val_daya, u_test)

    # Normalisasi Z-score dihitung hanya dari X_train
    mu = X_train.mean(axis=(0, 1), keepdims=True)
    sd = X_train.std(axis=(0, 1), keepdims=True) + 1e-6
    X_train = (X_train - mu) / sd
    X_val = (X_val - mu) / sd
    X_test = (X_test - mu) / sd

    return X_train, y_train, X_val, y_val, X_test, y_test