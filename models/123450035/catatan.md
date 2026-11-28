# Catatan model 123450035 (jalur RNN: GRU atas deret tegangan 5 V)

## Alasan optimizer
Adam (lr 3e-3) karena adaptif dan stabil untuk RNN tanpa penyetelan awal. Gradient clipping 1,0 untuk menjaga stabilitas GRU.

## Regularisasi
Dropout 0,2 sebelum layer klasifikasi. Penyetelan lebih lanjut di Sprint 2.

## Pencarian hyperparameter (tabel)
Sprint 2.

## Ablation (>= 2)
Sprint 2.

## Hasil 3 seed konfigurasi akhir
Sprint 2. Sprint 1: seed 1 saja, val_acc 0,7404, macro-F1 0,6518, F1 alat kipas 0,9516 / pompa 0,7953 / lampu 0,8152.
