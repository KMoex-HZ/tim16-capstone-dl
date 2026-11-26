# RNN (GRU) atas Deret Tegangan (Modul 07 - 08) - 123450035

Jalur model **RNN (`rnn`)** berupa GRU 1 lapis untuk klasifikasi multi-alat NILM mini pada data tegangan rel 5 V meja (400 titik / 2 detik, dibagi 40 langkah waktu x 10 fitur). Keluaran 12 kelas (kipas x pompa x lampu).

## Cara Menjalankan
```bash
# Latih dan evaluasi validasi (Sprint 1)
python models/123450035/run.py --seed 1

# Evaluasi final data uji (Sprint 2)
python models/123450035/run.py --seed 1 --final
```

Detail optimizer, regularisasi, dan hasil ada di `catatan.md`.
