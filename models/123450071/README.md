# Transformer Encoder atas Patch (Modul 09) - 123450071

Jalur model **Transformer (`trf`)** untuk klasifikasi multi-alat NILM mini pada data arus dan tegangan meja (400 titik / 2 detik).

## Cara Menjalankan
```bash
# Latih dan evaluasi validasi (Sprint 1)
python models/123450071/run.py --seed 1

# Evaluasi final data uji (Sprint 2)
python models/123450071/run.py --seed 1 --final