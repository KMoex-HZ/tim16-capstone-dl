# Catatan model 123450071 (Transformer Encoder — `trf`)

## Alasan optimizer
Menggunakan **AdamW** (`lr=1e-3`, `weight_decay=1e-2`) dipadukan dengan penjadwalan **CosineAnnealingLR**, karena arsitektur Transformer memiliki mekanisme *self-attention* yang sensitif terhadap skala gradien awal sehingga membutuhkan pembaruan adaptif per parameter serta peluruhan bobot (*decoupled weight decay*) agar lebih stabil dibanding SGD polos.

## Regularisasi
1. **Dropout (`p=0.2`)** pada lapisan proyeksi *patch* masuk (`[1, 400]` dibagi 40 *patch* berukuran 10) dan di dalam blok `TransformerEncoderLayer`.
2. **Weight Decay (`1e-2`)** pada optimizer AdamW.
3. **Gradient Clipping (`max_norm=1.0`)** serta penyimpanan bobot terbaik berdasarkan rata-rata F1 per alat pada data validasi (`val_f1_alat`) untuk mencegah *overfitting* pada sinyal tegangan rel `bus_mV`.

## Pencarian hyperparameter (tabel)
| Eksperimen | `patch_size` | `d_model` | `nhead` | `num_layers` | `dropout` | Hasil Validasi (`--seed 1`) |
|---|---|---|---|---|---|---|
| Konfigurasi Sprint 1 (protokol revisi `bus_mV`, 12 kode) | 10 | 64 | 4 | 2 | 0.2 | `val_f1_alat`: `0.8902` (kipas `0.9836`, pompa `0.8685`, lampu `0.8185`), `val_acc`: `0.765957`, `val_macro_f1`: `0.5415` |

## Ablation (>= 2)
*(Akan dilengkapi pada Sprint 2: perbandingan `patch_size=10` vs `patch_size=20`, serta `num_layers=2` vs `num_layers=1`)*

## Hasil 3 seed konfigurasi akhir
*(Akan dilengkapi pada Sprint 2 untuk `--seed 1`, `--seed 2`, `--seed 3`)*