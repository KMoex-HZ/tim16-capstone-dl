# Protokol evaluasi tim — 16 · Rogue Volt

Diisi Penjaga Data & Evaluasi pada Sprint 0/1, digabung ke `main` lewat *pull request* yang
disetujui pembimbing, lalu **dibekukan**. Perubahan setelah itu hanya lewat *pull request*
baru yang disetujui pembimbing.

| Butir | Isi |
|---|---|
| Kasus (dari kartu Kasus tim) | Pengenalan alat menyala dari daya total satu titik (NILM mini) |
| Sumber dataset | Rekaman primer Meja 5 Volt Tim 16 (17 Nov 2026: `data/daya.csv` & `data/label.csv`) |
| Jenis tugas (`TUGAS`) | `klasifikasi` |
| Target / label | Kombinasi status 3 alat aktif (`kipas`, `pompa`, `lampu` > 0) dikodekan 0..7 (`kipas*4 + pompa*2 + lampu`) |
| Ukuran data latih / validasi / uji | 1.176 jendela (9 ulangan) / 452 jendela (3 ulangan) / 388 jendela (3 ulangan) |
| Cara split | `split.indeks` seed 2026 per grup `ulangan` (bukan acak per jendela) |
| Metrik utama untuk komparasi | `macro_f1` (dan F1 per alat via `ke_biner`), rerata ± simpangan dari 3 seed |
| Praproses yang sama untuk semua model | Buang 1 detik pertama tiap kondisi, potong jendela 2 detik (400 titik @ 200 Hz) pada kanal `[arus_mA, bus_mV]`, normalisasi Z-score dari `X_train` saja |
| Anggaran komputasi per run | ≤ 20 menit CPU laptop |
| Evaluasi data uji | sekali, oleh Penjaga Data, pada sprint terakhir (`--final`) |
| `data_hash` saat dibekukan | `d65948c2d79caabac07c85504b355eda` |

Disetujui pembimbing pada: *(tautan pull request)*