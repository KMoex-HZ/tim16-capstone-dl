"""Pengujian protokol Tim 16: uji simulasi pemotong jendela, koreksi lampu, dan anti-kebocoran ulangan."""
from __future__ import annotations

import csv
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from src.protokol import data


def _buat_sesi_simulasi(folder: Path, n_ulangan: int = 15) -> Path:
    """Bangkitkan folder sesi simulasi (label.csv, aktuator.csv, daya.csv @ 200 Hz)."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(16)

    t_sekarang = 1000.0
    t_esp_awal = 0.0
    durasi_kondisi = 6.0  # 1 s transisi dibuang + 5 s -> 4 jendela (2 s, langkah 1 s)
    jeda = 1.0

    baris_label = []
    baris_aktuator = []
    baris_daya = []

    kombinasi = list(product([0, 128, 255], [0, 255], [0, 255]))
    no = 0
    for ulangan in range(1, n_ulangan + 1):
        for kipas, pompa, lampu in kombinasi:
            no += 1
            t_mulai = t_sekarang
            t_selesai = t_mulai + durasi_kondisi

            # Simulasi perintah lampu terbalik terhadap lux (seperti sesi meja)
            lux = 60.0 if lampu > 0 else 230.0
            baris_label.append({
                "no": no,
                "ulangan": ulangan,
                "kipas": kipas,
                "pompa": pompa,
                "lampu": lampu,
                "lux": lux,
                "t_mulai_host": f"{t_mulai:.4f}",
                "t_selesai_host": f"{t_selesai:.4f}",
            })
            baris_aktuator.extend([
                {"t_host": f"{t_mulai:.4f}", "nama": "kipas", "nilai": kipas},
                {"t_host": f"{t_mulai:.4f}", "nama": "pompa", "nilai": pompa},
                {"t_host": f"{t_mulai:.4f}", "nama": "lampu", "nilai": 1 if lampu > 0 else 0},
            ])

            # Sinyal bus_mV turun proporsional terhadap beban yang aktif
            n_sampel = int(durasi_kondisi * data.LAJU_HZ) + 20
            beban = (kipas / 255.0) * 120.0 + (pompa > 0) * 180.0 + (lampu == 0) * 90.0
            for k in range(n_sampel):
                dt = k / data.LAJU_HZ
                th = t_mulai + dt
                te_ms = (th - 1000.0 + t_esp_awal) * 1000.0
                v = 5000.0 - beban + float(rng.normal(0.0, 2.0))
                baris_daya.append([f"{th:.4f}", f"{te_ms:.2f}", "kipas", "0.0", f"{v:.2f}"])

            t_sekarang = t_selesai + jeda

    with (folder / "label.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(baris_label[0].keys()))
        w.writeheader()
        w.writerows(baris_label)

    with (folder / "aktuator.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["t_host", "nama", "nilai"])
        w.writeheader()
        w.writerows(baris_aktuator)

    with (folder / "daya.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_host", "t_esp_ms", "kanal", "arus_mA", "bus_mV"])
        w.writerows(baris_daya)

    return folder


def test_simulasi_muat_dan_anti_kebocoran_ulangan(tmp_path: Path):
    """Uji end-to-end pada folder simulasi dan pastikan ulangan latih/val/uji tidak tumpang tindih."""
    folder_sim = _buat_sesi_simulasi(tmp_path / "sesi-simulasi", n_ulangan=15)

    X_tr, y_tr, X_va, y_va, X_te, y_te = data.muat(folder_sesi=[folder_sim])
    info = data.ringkasan_split(folder_sesi=[folder_sim])

    assert X_tr.ndim == 3 and X_tr.shape[1:] == (1, 400)
    assert X_va.shape[1:] == (1, 400) and X_te.shape[1:] == (1, 400)
    assert len(np.unique(y_tr)) == 12

    # Pastikan tidak ada ulangan yang muncul di latih/val dan uji sekaligus
    assert set(info["ulangan_latih"] if "ulangan_latih" in info else info["latih"]).isdisjoint(info["uji"])
    assert set(info["val"]).isdisjoint(info["uji"])
    assert set(info["latih"]).isdisjoint(info["val"])

    # Normalisasi dihitung dari data latih saja (mean ~ 0, std ~ 1 pada X_train)
    assert abs(float(X_tr.mean())) < 1e-4
    assert abs(float(X_tr.std()) - 1.0) < 1e-2


def test_periksa_tidak_bocor_gagal_jika_ada_irisan():
    data.periksa_tidak_bocor([1, 2, 3], [4], [5, 6])
    with pytest.raises(AssertionError):
        data.periksa_tidak_bocor([1, 2, 3], [4], [3, 6])


def test_deteksi_lampu_terbalik_dan_lux_basi():
    baris_terbalik = [
        {"no": "1", "lampu": "255", "lux": "60", "t_mulai_host": "0", "t_selesai_host": "30"},
        {"no": "2", "lampu": "0", "lux": "230", "t_mulai_host": "32", "t_selesai_host": "62"},
        {"no": "3", "lampu": "255", "lux": "240", "t_mulai_host": "64", "t_selesai_host": "94"},
        {"no": "4", "lampu": "0", "lux": "240", "t_mulai_host": "96", "t_selesai_host": "126"},
        {"no": "5", "lampu": "255", "lux": "240", "t_mulai_host": "128", "t_selesai_host": "158"},
    ]
    hasil = data.koreksi_label_lampu(baris_terbalik)
    assert hasil[0]["lampu"] == "0" and hasil[0]["lampu_perintah"] == "255"


def test_kode_label_dan_f1_alat():
    kode = data.gabung_kode_label({"kipas": "128", "pompa": "255", "lampu": "0"})
    assert kode == 6
    k, p, l = data.urai_kode_label(np.array([kode]))
    assert (int(k[0]), int(p[0]), int(l[0])) == (1, 1, 0)
    skor = data.hitung_f1_alat(np.array([0, 6, 11]), np.array([0, 6, 11]))
    assert skor["val_f1_alat"] == 1.0


data_meja_ada = all((data.manifest.AKAR / "data" / s).exists() for s in data.SESI_AKTIF)


@pytest.mark.skipif(not data_meja_ada, reason="Folder sesi meja 17 Nov belum ada di data/")
def test_data_primer_meja_valid_dan_tidak_bocor():
    X_tr, y_tr, X_va, y_va, X_te, y_te = data.muat()
    info = data.ringkasan_split()
    assert X_tr.shape == (1755, 1, 400)
    assert X_va.shape == (470, 1, 400)
    assert X_te.shape == (1130, 1, 400)
    assert len(np.unique(y_tr)) == 12
    assert set(info["latih"]).isdisjoint(info["uji"])
    assert set(info["val"]).isdisjoint(info["uji"])
    assert set(info["latih"]).isdisjoint(info["val"])