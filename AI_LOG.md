# AI_LOG — Tim 16 — Rogue Volt

Satu baris per pemakaian AI yang bermakna (ChatGPT, Claude, Copilot, Bravais, dan sejenisnya).
Ringkas dan jujur; transkrip penuh tidak diminta. Yang dinilai bukan "pakai AI atau tidak",
melainkan "mengerti atau tidak" — isi kolom terakhir dengan apa yang Anda ubah dari keluarannya.

| Tanggal | Siapa (NIM) | Alat | Untuk apa | Apa yang diubah dari keluarannya |
|---|---|---|---|---|
| 2026-11-17 | 123450071 | Gemini | skrip pemotong jendela sinyal 2 detik (`src/protokol/data.py`) & validasi `MANIFEST.json` | menyesuaikan batas indeks kondisi terputus (26–47), membuang 1 detik awal transisi, dan memetakan 8 kelas kombinasi biner (kipas, pompa, lampu) |
| 2026-11-18 | 123450071 | Gemini | kerangka Patch Time-Series Transformer Encoder (`models/123450071/run.py`) | mengatur ukuran *patch* 10 titik (50 ms), dimensi ringan (`d_model=64`, 2 layer) agar tidak *overfit* pada 1.176 jendela latih, serta menyambungkan ke `src.common.evaluate` |