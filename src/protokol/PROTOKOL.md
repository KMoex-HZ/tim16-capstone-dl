# Protokol evaluasi tim — 16 · Rogue Volt

Diisi Penjaga Data & Evaluasi pada Sprint 0/1, digabung ke `main` lewat *pull request* yang
disetujui pembimbing, lalu **dibekukan**. Perubahan setelah itu hanya lewat *pull request*
baru yang disetujui pembimbing.

| Butir | Isi |
|---|---|
| Kasus (dari kartu Kasus tim) | Pengenalan alat menyala dari daya total satu titik (NILM mini) |
| Sumber dataset | Dua sesi rekaman primer Meja 5 Volt Tim 16 (17 Nov 2026: `data/sesi-20261117-153501` & `data/sesi-20261117-182740`, rincian di `data/MANIFEST.json`) |
| Jenis tugas (`TUGAS`) | `klasifikasi` |
| Target / label | Kombinasi taraf kipas (`0`/`128`/`255`, 3 kelas) serta status fisik pompa dan lampu (dikoreksi dari sensor `lux`), dikodekan 0..11 (`kelas_kipas*4 + pompa*2 + lampu`) |
| Ukuran data latih / validasi / uji | 1.755 jendela (8 ulangan) / 470 jendela (2 ulangan) / 1.130 jendela (5 ulangan) |
| Cara split | `split.indeks` seed 2026 per grup `ulangan` (bukan acak per jendela) + verifikasi `periksa_tidak_bocor()` |
| Metrik utama untuk komparasi | Rata-rata F1 per alat (`val_f1_alat`: macro-F1 kipas, F1 pompa, F1 lampu) serta `val_macro_f1` & `val_acc` penilai starter |
| Praproses yang sama untuk semua model | Buang 1 detik pertama tiap kondisi, potong jendela 2 detik bergeser 1 detik (400 titik @ 200 Hz menurut jam ESP32) pada kanal tunggal `[bus_mV]` (tegangan rel 5 V; `arus_mA` kipas tidak dipakai), tolak jendela bercelah > 50 ms & kondisi tidak sesuai `aktuator.csv`, normalisasi Z-score dari `X_train` saja |
| Anggaran komputasi per run | ≤ 20 menit CPU laptop |
| Evaluasi data uji | sekali, oleh Penjaga Data, pada sprint terakhir (`--final`) |
| `data_hash` saat diperbarui | `253cd7a1402639fef1ec82033dfc8a58` |

## Penyimpangan dari Rencana Awal (Temuan Sesi Meja 17 Nov 2026)
1. **Kanal masukan (`bus_mV` sebagai pengganti `kanal = total`):** Pada saat pengambilan data sesi meja 17 Nov 2026 (`sesi-20261117-153501` & `sesi-20261117-182740`), sensor INA219 arus total (`0x44`) belum terpasang sehingga `daya.csv` hanya merekam kanal `kipas` (`INA219 0x40`). Arus kipas (`arus_mA`) sengaja **tidak dipakai** sebagai masukan karena akan membocorkan status kipas secara langsung; sebagai pengganti titik tunggal NILM digunakan **tegangan rel 5 V (`bus_mV`)** yang turun sebanding dengan beban total pada catu bersama.
2. **Label 12 kelas (`kipas 0/128/255`, `pompa`, `lampu`):** Mengikuti `rencana.yaml` aktual sesi meja (180 kondisi = 12 kombinasi × 15 ulangan) dan hasil pemeriksaan mutu (`data/CATATAN-SESI.md`). Perintah lampu yang tercatat terbalik di `label.csv` dikoreksi ke kondisi fisik menggunakan pembacaan sensor `lux`.

## Pemeriksaan Simulasi dan Anti-Kebocoran
- Disediakan pada `tests/test_protokol.py` (`pytest -v`): menguji `muat(folder_sesi=[...])` pada sesi simulasi sintetik serta memverifikasi dengan `periksa_tidak_bocor()` bahwa tidak ada `ulangan` yang muncul di split latih/validasi dan uji sekaligus.

Disetujui pembimbing pada: *(menunggu tinjauan PR revisi protokol)*