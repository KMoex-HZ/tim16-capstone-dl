"""Protokol pemuatan data Tim 16 (Rogue Volt): NILM mini dari tegangan rel 5 V (bus_mV)."""
from __future__ import annotations

import bisect
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np

from src.common import manifest, split

TUGAS = "klasifikasi"
TARAF_KIPAS = [0, 128, 255]
SESI_AKTIF = ["sesi-20261117-153501", "sesi-20261117-182740"]

LAJU_HZ = 200.0
JENDELA_DETIK = 2.0
LANGKAH_DETIK = 1.0
TRANSISI_AWAL_DETIK = 1.0
CELAH_MAKS_DETIK = 0.05
TITIK_CEK_AKTUATOR = (1.0, 15.0, 29.0)

FOLDER_CACHE = manifest.AKAR / "data" / "cache"


def _baca_csv(folder: Path, nama_berkas: str) -> list[dict]:
    path = folder / nama_berkas
    if not path.exists():
        return []
    with path.open(encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def deteksi_lampu_terbalik(baris_label: list[dict]) -> bool:
    """Cek apakah perintah lampu terbalik terhadap cahaya fisik (lux) tanpa nilai basi."""
    urut = sorted(
        (b for b in baris_label if "lampu" in b and b.get("lux") not in (None, "")),
        key=lambda b: int(b["no"]),
    )
    nilai_lux = [float(b["lux"]) for b in urut]
    indeks_basi = set()
    i = 0
    while i < len(nilai_lux):
        j = i
        while j + 1 < len(nilai_lux) and nilai_lux[j + 1] == nilai_lux[i]:
            j += 1
        if j - i >= 2:
            indeks_basi.update(range(i, j + 1))
        i = j + 1

    lux_perintah_nyala = [
        x for idx, (b, x) in enumerate(zip(urut, nilai_lux))
        if idx not in indeks_basi and float(b["lampu"]) > 0
    ]
    lux_perintah_mati = [
        x for idx, (b, x) in enumerate(zip(urut, nilai_lux))
        if idx not in indeks_basi and float(b["lampu"]) == 0
    ]
    if not lux_perintah_nyala or not lux_perintah_mati:
        return False
    return median(lux_perintah_nyala) < median(lux_perintah_mati)


def koreksi_label_lampu(baris_label: list[dict]) -> list[dict]:
    """Simpan perintah asli di 'lampu_perintah' dan ganti 'lampu' menjadi keadaan fisik."""
    terbalik = deteksi_lampu_terbalik(baris_label)
    hasil = []
    for b in baris_label:
        salinan = dict(b)
        salinan["t_mulai_host"] = float(salinan["t_mulai_host"])
        salinan["t_selesai_host"] = float(salinan["t_selesai_host"])
        if "lampu" in salinan:
            salinan["lampu_perintah"] = salinan["lampu"]
            if terbalik:
                salinan["lampu"] = "0" if float(salinan["lampu"]) > 0 else "255"
        hasil.append(salinan)
    return hasil


def saring_kondisi_aktuator(folder: Path, baris_label: list[dict]) -> set[int]:
    """Kembalikan nomor kondisi yang status aktuatornya tidak sesuai perintah."""
    log_aktuator = defaultdict(list)
    for r in _baca_csv(folder, "aktuator.csv"):
        log_aktuator[r["nama"]].append((float(r["t_host"]), float(r["nilai"])))

    kondisi_tidak_sah = set()
    for b in baris_label:
        no_kondisi = int(b["no"])
        for alat in ("kipas", "pompa", "lampu"):
            waktu_log = [t for t, _ in log_aktuator[alat]]
            target_perintah = float(b["lampu_perintah"] if alat == "lampu" else b[alat])
            for jeda in TITIK_CEK_AKTUATOR:
                idx = bisect.bisect_right(waktu_log, b["t_mulai_host"] + jeda) - 1
                if idx < 0:
                    kondisi_tidak_sah.add(no_kondisi)
                    break
                nilai_aktual = log_aktuator[alat][idx][1]
                if (nilai_aktual > 0) != (target_perintah > 0):
                    kondisi_tidak_sah.add(no_kondisi)
                    break
                if alat != "lampu" and nilai_aktual != target_perintah:
                    kondisi_tidak_sah.add(no_kondisi)
                    break
            if no_kondisi in kondisi_tidak_sah:
                break
    return kondisi_tidak_sah


def _baca_deret_tegangan(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    """Baca sinyal bus_mV dan petakan jam ESP32 (t_esp_ms) ke waktu host."""
    t_host, t_esp, tegangan = [], [], []
    with (folder / "daya.csv").open(encoding="utf-8", errors="replace") as f:
        pembaca = csv.reader(f)
        kepala = next(pembaca)
        i_kanal = kepala.index("kanal")
        i_bus = kepala.index("bus_mV")
        for r in pembaca:
            if len(r) != len(kepala) or r[i_kanal] != "kipas":
                continue
            try:
                th = float(r[0])
                te = float(r[1]) / 1000.0
                v = float(r[i_bus])
            except ValueError:
                continue
            if np.isnan(v):
                continue
            t_host.append(th)
            t_esp.append(te)
            tegangan.append(v)

    t_host_arr = np.asarray(t_host, dtype=np.float64)
    t_esp_arr = np.asarray(t_esp, dtype=np.float64)
    v_arr = np.asarray(tegangan, dtype=np.float32)

    urut = np.argsort(t_esp_arr, kind="stable")
    t_host_arr, t_esp_arr, v_arr = t_host_arr[urut], t_esp_arr[urut], v_arr[urut]
    offset_host = float(np.percentile(t_host_arr - t_esp_arr, 1)) if len(t_host_arr) else 0.0
    return t_esp_arr + offset_host, v_arr


def _potong_jendela(t_deret: np.ndarray, v_deret: np.ndarray, t0: float, t1: float) -> np.ndarray | None:
    i0, i1 = np.searchsorted(t_deret, [t0, t1])
    t_potong = t_deret[max(i0 - 1, 0) : min(i1 + 1, len(t_deret))]
    if len(t_potong) < 2 or t_potong[0] > t0 + CELAH_MAKS_DETIK or t_potong[-1] < t1 - CELAH_MAKS_DETIK:
        return None
    if np.diff(t_potong).max() > CELAH_MAKS_DETIK:
        return None

    n_sampel = int(round((t1 - t0) * LAJU_HZ))
    grid = t0 + np.arange(n_sampel) / LAJU_HZ
    v_potong = v_deret[max(i0 - 1, 0) : min(i1 + 1, len(t_deret))]
    hasil = np.interp(grid, t_potong, v_potong).astype(np.float32)
    return hasil[None, :]  # bentuk [1, 400]


def gabung_kode_label(baris: dict) -> int:
    """Kode 0..11 = kelas_kipas (0..2) * 4 + pompa (0/1) * 2 + lampu (0/1)."""
    kelas_kipas = TARAF_KIPAS.index(int(float(baris["kipas"])))
    pompa = int(float(baris["pompa"]) > 0)
    lampu = int(float(baris["lampu"]) > 0)
    return kelas_kipas * 4 + pompa * 2 + lampu


def urai_kode_label(kode: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kembalikan (kelas_kipas [0..2], pompa [0/1], lampu [0/1]) dari kode 0..11."""
    k = np.asarray(kode, dtype=np.int64)
    kipas = k // 4
    pompa = (k >> 1) & 1
    lampu = k & 1
    return kipas, pompa, lampu


def _f1_biner(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    return 0.0 if tp == 0 else round(2 * tp / (2 * tp + fp + fn), 4)


def hitung_f1_alat(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Hitung F1 per alat (kipas macro-F1 3 kelas, pompa F1, lampu F1) dan rata-ratanya."""
    kt, pt, lt = urai_kode_label(y_true)
    kp, pp, lp = urai_kode_label(y_pred)
    f1_kipas = round(float(np.mean([_f1_biner(kt == c, kp == c) for c in range(3)])), 4)
    f1_pompa = _f1_biner(pt == 1, pp == 1)
    f1_lampu = _f1_biner(lt == 1, lp == 1)
    rerata = round(float(np.mean([f1_kipas, f1_pompa, f1_lampu])), 4)
    return {
        "val_f1_alat": rerata,
        "val_f1_rinci": {"kipas": f1_kipas, "pompa": f1_pompa, "lampu": f1_lampu},
    }


def _kunci_cache() -> str:
    h = hashlib.sha256(b"tim16-protokol-v2-busmV")
    for s in SESI_AKTIF:
        folder = manifest.AKAR / "data" / s
        for berkas in ("label.csv", "aktuator.csv"):
            h.update((folder / berkas).read_bytes())
        h.update(str((folder / "daya.csv").stat().st_size).encode())
    h.update(json.dumps([LAJU_HZ, JENDELA_DETIK, LANGKAH_DETIK, TRANSISI_AWAL_DETIK]).encode())
    return h.hexdigest()[:16]


def _kumpulkan_jendela() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    FOLDER_CACHE.mkdir(parents=True, exist_ok=True)
    berkas_cache = FOLDER_CACHE / f"tim16_{_kunci_cache()}.npz"
    if berkas_cache.exists():
        with np.load(berkas_cache) as z:
            return z["X"], z["y"], z["ulangan"]

    X_list, y_list, ulangan_list = [], [], []
    for s in SESI_AKTIF:
        folder = manifest.AKAR / "data" / s
        if not folder.exists():
            raise FileNotFoundError(f"Folder sesi belum ditemukan: {folder}")
        baris_label = koreksi_label_lampu(_baca_csv(folder, "label.csv"))
        kondisi_batal = saring_kondisi_aktuator(folder, baris_label)
        t_deret, v_deret = _baca_deret_tegangan(folder)

        for b in baris_label:
            if int(b["no"]) in kondisi_batal:
                continue
            t = b["t_mulai_host"] + TRANSISI_AWAL_DETIK
            while t + JENDELA_DETIK <= b["t_selesai_host"] + 1e-6:
                jendela = _potong_jendela(t_deret, v_deret, t, t + JENDELA_DETIK)
                if jendela is not None:
                    X_list.append(jendela)
                    y_list.append(gabung_kode_label(b))
                    ulangan_list.append(int(b["ulangan"]))
                t += LANGKAH_DETIK

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    ulangan = np.asarray(ulangan_list, dtype=np.int64)
    np.savez(berkas_cache, X=X, y=y, ulangan=ulangan)
    return X, y, ulangan


def muat(standar: bool = True):
    """Kembalikan (X_train, y_train, X_val, y_val, X_test, y_test) dengan split per ulangan."""
    X, y, ulangan = _kumpulkan_jendela()
    daftar_ulangan = sorted(set(ulangan.tolist()))

    idx_sisa, idx_uji = split.indeks(len(daftar_ulangan), porsi_val=1 / 3)
    ulangan_uji = [daftar_ulangan[int(i)] for i in idx_uji]
    sisa_ulangan = [daftar_ulangan[int(i)] for i in idx_sisa]

    _, idx_val = split.indeks(len(sisa_ulangan), porsi_val=0.2)
    ulangan_val = [sisa_ulangan[int(i)] for i in idx_val]

    mask_uji = np.isin(ulangan, ulangan_uji)
    mask_val = np.isin(ulangan, ulangan_val)
    mask_latih = ~mask_uji & ~mask_val

    if standar:
        mu = X[mask_latih].mean(axis=(0, 2), keepdims=True)
        sd = X[mask_latih].std(axis=(0, 2), keepdims=True) + 1e-6
        X = ((X - mu) / sd).astype(np.float32)

    return (
        X[mask_latih],
        y[mask_latih],
        X[mask_val],
        y[mask_val],
        X[mask_uji],
        y[mask_uji],
    )


if __name__ == "__main__":
    X_tr, y_tr, X_va, y_va, X_te, y_te = muat()
    print(f"Latih: {X_tr.shape} | Validasi: {X_va.shape} | Uji: {X_te.shape} | Kelas unik: {len(set(y_tr.tolist()))}")