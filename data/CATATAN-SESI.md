# Catatan Pemeriksaan Mutu Sesi — Tim 16 (Rogue Volt)

## 1. Sesi yang Dipakai & Selisih terhadap Rencana Awal
- **Tanggal pengambilan:** 17 November 2026 (Meja 5 Volt Ruang Cerdas).
- **Sumber sesi (disimpan terpisah per arsip di `data/`):**
  1. `sesi-20261117-153501` — kondisi 1–47 (47 kondisi, sesi awal sebelum dihentikan).
  2. `sesi-20261117-182740` — kondisi 48–180 (133 kondisi, lanjutan otomatis `#174` hingga selesai).
- **Jumlah kondisi & selisih dari kartu kasus awal:**
  - Total kondisi terekam: **180 kondisi** (`12 kombinasi × 15 ulangan` dari faktor `kipas` `[0, 128, 255]`, `pompa` `[0, 255]`, `lampu` `[0, 255]`).
  - **Selisih (-60 kondisi dari 240 kondisi pada deskripsi awal):** sesuai `rencana.yaml` aktual di kedua arsip sesi meja, faktor `buzzer` tidak dijalankan pada sesi meja 17 Nov sehingga rancangan menjadi 180 kondisi (12 kombinasi), bukan 240 kondisi (16 kombinasi).

## 2. Ringkasan Pemeriksaan Mutu Sensor & Perangkat Keras
- **Kanal daya (`daya.csv`, 200 Hz):** Hanya modul `INA219` kanal `kipas` (`0x40`) yang terpasang saat pengambilan data; modul `INA219` arus total (`0x44`) belum terpasang.
  - **Pemeriksaan respons arus kipas (verifikasi alat menyala):**
    - `kipas PWM = 0`: median arus **-0,1 mA** (`n = 44` kondisi)
    - `kipas PWM = 128`: median arus **110,8 mA** (`n = 43` kondisi)
    - `kipas PWM = 255`: median arus **246,4 mA** (`n = 46` kondisi)
  - **`arus_mA` tidak dijadikan fitur model** agar status kipas tidak bocor langsung; sebagai sinyal titik tunggal NILM digunakan **tegangan rel 5 V (`bus_mV`)**.
- **Pemeriksaan logika lampu (`label.csv` vs `lux`):**
  - Pada program ESP32 sesi meja, perintah lampu tercatat terbalik terhadap keadaan fisik bohlam (`deteksi_lampu_terbalik = True` pada kedua sesi).
  - Pada `sesi-20261117-182740`, median cahaya `lux` saat perintah `0` adalah **230,0** dan saat perintah `255` turun menjadi **62,5** (pada `sesi-20261117-153501`, setelah menyaring nilai `lux` basi `240.0`, arah juga terdeteksi terbalik). Protokol membalik label `lampu` ke keadaan fisik sebenarnya.
- **Sinkronisasi waktu & celah Wi-Fi:** Jendela dipotong menggunakan jam ESP32 (`t_esp_ms`) yang dipetakan ke waktu host (`t_host`), dan jendela yang memiliki celah sampel `> 50 ms` ditolak.
- **Digital (`digital.csv`):** 0 baris (sensor PIR/pintu tidak terpasang di meja).

## 3. Kondisi yang Dikeluarkan beserta Alasannya
Dari **180 kondisi** rancangan, sebanyak **46 kondisi dikeluarkan** (tersisa **134 kondisi sah**) karena status aktual pada `aktuator.csv` (titik cek `1 s`, `15 s`, `29 s`) tidak sesuai perintah pada `label.csv`:
- **`sesi-20261117-153501` (30 kondisi dikeluarkan):** nomor `13, 14, 15, 16, 19, 20, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47` (aktuator tidak sesuai perintah dan/atau perekaman terputus menjelang penghentian sesi pertama).
- **`sesi-20261117-182740` (16 kondisi dikeluarkan):** nomor `106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121` (ketidaksesuaian status aktuator pada log `aktuator.csv` terhadap perintah kondisi).

## 4. Pemotongan Jendela & Ukuran Data Valid
- **Jendela ditolak (celah `> 50 ms` / sambungan terputus):** **397 jendela** (**336** pada `sesi-20261117-153501` dan **61** pada `sesi-20261117-182740`).
- **Jendela valid diterima:** **3.355 jendela** (**140** dari sesi pertama + **3.215** dari sesi kedua), masing-masing berukuran `[1, 400]` (2 detik @ 200 Hz, 1 detik awal dibuang):
  - **Latih:** **1.755 jendela** (8 ulangan: `[1, 4, 5, 6, 7, 8, 13, 15]`)
  - **Validasi:** **470 jendela** (2 ulangan: `[11, 14]`)
  - **Uji:** **1.130 jendela** (5 ulangan: `[2, 3, 9, 10, 12]`)

## 5. Konfirmasi `MANIFEST.json` & Keseragaman `data_hash` Anggota
- Seluruh 8 berkas primer pada `data/MANIFEST.json` diverifikasi dengan SHA-256 (sidik berkas mentah `manifest.data_hash()` = `099240e04d7219b951882fea8b01d727`).
- Seluruh anggota dan pembanding menghasilkan `data_hash` evaluasi yang **identik** pada `results/*/metrics.json`:

| Jalur / Anggota | `model` | `n_val` | `data_hash` |
|---|---|---|---|
| Baseline | `baseline` | 470 | `253cd7a1402639fef1ec82033dfc8a58` |
| CNN (`123450019`) | `cnn` | 470 | `253cd7a1402639fef1ec82033dfc8a58` |
| RNN (`123450035`) | `rnn` | 470 | `253cd7a1402639fef1ec82033dfc8a58` |
| Transformer (`123450071`) | `trf` | 470 | `253cd7a1402639fef1ec82033dfc8a58` |
| Autoencoder (`123450079`) | `ae` | 470 | `253cd7a1402639fef1ec82033dfc8a58` |