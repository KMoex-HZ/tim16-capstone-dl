# Catatan Pemeriksaan Mutu Sesi — Tim 16 (Rogue Volt)

## 1. Sesi yang Dipakai
- **Tanggal pengambilan:** 17 November 2026 (Meja 5 Volt Ruang Cerdas).
- **Sumber sesi (digabung):**
  1. Sesi `15.35` (kondisi 1–47, durasi 25 menit, status dibatalkan).
  2. Sesi lanjutan `18.27` (`otomatislanjutan #174`, kondisi 48–180, durasi 1 jam 10 menit, status selesai).
- **Total kondisi:** 180 kondisi (12 kombinasi dari faktor `kipas` [0, 128, 255], `pompa` [0, 255], `lampu` [0, 255] × 15 ulangan).

## 2. Pemeriksaan Mutu Sensor
- **Daya (`daya.csv`, 200 Hz):** Kanal yang terpasang dan terekam pada meja saat sesi berlangsung adalah `kipas` (`INA219 0x40`), berisi pembacaan `arus_mA` dan `bus_mV` (923.371 baris). Arus meningkat konsisten saat kipas menyala (~114 mA pada PWM 128 dan ~245 mA pada PWM 255) dan tegangan `bus_mV` ikut merespons beban pompa/lampu.
- **Lingkungan (`lingkungan.csv`) & Aktuator (`aktuator.csv`):** Tercatat lengkap sesuai stempel waktu `t_host`.
- **Digital (`digital.csv`):** Kosong (0 baris, sensor PIR/pintu tidak terpasang di meja).

## 3. Kondisi yang Dikeluarkan & Alasannya
- **Kondisi 26–47 (22 kondisi, pukul 15:48:21 – 15:59:33):** Dikeluarkan karena sampel `daya.csv` bernilai 0 (perekaman daya terputus menjelang sesi pertama dibatalkan).
- **Kondisi 106 (1 kondisi, pukul 18:58:36):** Dikeluarkan karena hanya memiliki 406 sampel; setelah pembuangan 1 detik pertama (200 sampel), sisa sampel kurang dari 1 jendela utuh (400 titik / 2 detik).
- **Total data valid:** 157 kondisi valid menghasilkan **2.016 jendela** (1.176 latih, 452 validasi, 388 uji).
- **`data_hash`:** `d65948c2d79caabac07c85504b355eda`