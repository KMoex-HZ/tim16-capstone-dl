# CNN dari nol (Modul 05) - 123450019
# CNN 1D atas Deret Tegangan Rel (Modul 06) - 123450019

Jalur model **CNN** (`cnn`) untuk klasifikasi multi-alat NILM mini pada data tegangan rel meja (400 titik / 2 detik). Empat blok `Conv1d` (kernel 9, 7, 5, 3) dengan BatchNorm dan MaxPool, ditutup *global average pooling* dan satu lapisan linear menjadi 5 logit (3 taraf kipas + pompa + lampu).

## Cara Menjalankan

```bash
# Latih dan evaluasi validasi (Sprint 1)
python models/123450019/run.py --seed 1

# Evaluasi final data uji (Sprint 2)
python models/123450019/run.py --seed 1 --final
```