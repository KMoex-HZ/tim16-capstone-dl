# Catatan model 123450019 (CNN 1D — `cnn`)

## Alasan optimizer

Menggunakan **AdamW** (`lr=1e-3`, `weight_decay=1e-2`) dipadukan dengan penjadwalan **CosineAnnealingLR**, disamakan dengan jalur Transformer agar tabel komparasi mengukur perbedaan arsitektur, bukan perbedaan optimizer. Pilihan ini juga cocok untuk arsitektur ber-BatchNorm: skala gradien antar blok konvolusi berbeda jauh karena jumlah kanal naik 1 → 32 → 64 → 128, sehingga pembaruan adaptif per parameter lebih stabil daripada SGD dengan satu laju belajar tunggal. *Decoupled weight decay* pada AdamW juga tidak ikut mengecilkan parameter skala BatchNorm seperti yang terjadi pada L2 biasa.

## Regularisasi

1. **Dropout** (`p=0.25`) pada kepala, setelah `AdaptiveAvgPool1d(1)` dan `LayerNorm(128)`, sebelum lapisan `Linear(128, 5)`.
2. **Weight Decay** (`1e-2`) pada optimizer AdamW.
3. **BatchNorm** setelah setiap `Conv1d` (4 blok), yang di samping menstabilkan latihan juga memberi efek regularisasi dari derau statistik per batch.
4. **Global average pooling** sebagai pengganti `Flatten`. Meratakan (128, 25) menjadi 128 memangkas kepala dari 16.005 parameter menjadi 645, sekaligus membuat keluaran tidak bergantung pada posisi kejadian di dalam jendela.
5. **Gradient Clipping** (`max_norm=1.0`) serta penyimpanan bobot terbaik berdasarkan rata-rata F1 per alat pada data validasi (`val_f1_alat`) untuk mencegah *overfitting* pada sinyal tegangan rel `bus_mV`.

## Pencarian hyperparameter (tabel)

| Eksperimen | kernel | lebar | dropout | lr | Hasil Validasi (`--seed 1`) |
|---|---|---|---|---|---|
| Konfigurasi Sprint 1 (protokol revisi `bus_mV`, 12 kode) | 9-7-5-3 | 32 | 0.25 | 1e-3 | `val_f1_alat`: `0.8865` (kipas `0.9918`, pompa `0.8418`, lampu `0.8260`), `val_acc`: `0.770213`, `val_macro_f1`: `0.626979` |

Parameter model 106.341. Medan reseptif 68 cuplikan (0,34 detik dari jendela 2 detik).

## Ablation (>= 2)

*(Akan dilengkapi pada Sprint 2: perbandingan aktivasi `GELU` vs `ReLU`, serta `AdaptiveAvgPool` vs gabungan `AvgPool + MaxPool` pada kepala)*

Dasar pemilihan kedua ablation: aktivasi `GELU` dipakai agar selaras dengan jalur Transformer dan belum diuji terhadap `ReLU` yang dipakai di materi kuliah; sementara rata-rata global diduga menghapus komponen DC yang menandai pompa dan lampu, dua alat dengan F1 terendah di Sprint 1.

## Hasil 3 seed konfigurasi akhir

*(Akan dilengkapi pada Sprint 2 untuk `--seed 1`, `--seed 2`, `--seed 3`)*

Catatan untuk Sprint 2: `--seed 1` yang sama menghasilkan `val_macro_f1` `0.5415` dan `0.6116` pada dua mesin berbeda untuk jalur `trf`, sehingga rata-rata ± simpangan tiga seed wajib dilaporkan sebelum menyimpulkan jalur mana yang unggul.