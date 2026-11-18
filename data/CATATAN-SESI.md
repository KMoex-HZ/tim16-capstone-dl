# Catatan Pemeriksaan Mutu Sesi — Tim 16 (Rogue Volt)

## 1. Sesi yang Dipakai
- **Tanggal pengambilan:** 17 November 2026 (Meja 5 Volt Ruang Cerdas).
- **Sumber sesi (disimpan terpisah per arsip di `data/`):**
  1. `sesi-20261117-153501` (kondisi 1–47, durasi 25 menit, sesi awal sebelum dihentikan).
  2. `sesi-20261117-182740` (`otomatislanjutan #174`, kondisi 48–180, durasi 1 jam 10 menit, sesi lanjutan hingga selesai).
- **Total rancangan kondisi:** 180 kondisi (12 kombinasi dari faktor `kipas` `[0, 128, 255]`, `pompa` `[0, 255]`, `lampu` `[0, 255]` × 15 ulangan).

## 2. Pemeriksaan Mutu Sensor & Anomali Perangkat Keras
- **Daya (`daya.csv`, 200 Hz):** Hanya kanal `kipas` (`INA219 0x40`) yang terekam pada meja saat sesi berlangsung; sensor `INA219` arus total belum terpasang. Oleh karena itu, **`arus_mA` tidak dipakai** sebagai fitur agar tidak membocorkan status kipas secara langsung, dan protokol memakai **tegangan rel 5 V (`bus_mV`)** sebagai sinyal titik tunggal NILM.
- **Koreksi logika lampu (`label.csv` vs `lux`):** Pada program ESP32 sesi meja, perintah lampu tercatat terbalik terhadap keadaan bohlam fisik (terverifikasi dari nilai median sensor cahaya `lux` yang turun saat perintah lampu `255` dan naik saat `0`). Protokol membalik label `lampu` ke keadaan fisik sebenarnya.
- **Sinkronisasi waktu & celah Wi-Fi:** Pemotongan jendela memakai jam ESP32 (`t_esp_ms`) yang dipetakan ke waktu host (`t_host`), serta menolak jendela yang memiliki celah sampel > 50 ms akibat pengiriman Wi-Fi tersendat.
- **Digital (`digital.csv`):** Kosong (0 baris, sensor PIR/pintu tidak terpasang di meja).

## 3. Penyaringan Kondisi & Ukuran Data Valid
- **Penyaringan `aktuator.csv`:** Kondisi yang status aktuator fisiknya tidak sesuai perintah eksperimen (termasuk kondisi 26–47 saat perekaman daya terputus) dikeluarkan.
- **Pembuangan transisi & jendela tidak utuh:** 1 detik pertama tiap kondisi dibuang, lalu dipotong menjadi jendela 2 detik (400 titik @ 200 Hz) dengan langkah 1 detik; jendela dengan celah > 50 ms ditolak.
- **Total data valid:** **3.355 jendela** utuh (**1.755 latih** / 8 ulangan, **470 validasi** / 2 ulangan, **1.130 uji** / 5 ulangan) mencakup seluruh 12 kombinasi kelas.