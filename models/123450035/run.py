"""run.py jalur RNN (GRU atas deret tegangan 5 V). Tim 16.

    python models/123450035/run.py --seed 1 [--final]
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402  (protokol tim yang dibekukan)

NIM = Path(__file__).resolve().parent.name     # nama folder = NIM
JALUR = "rnn"                                   # cnn | tl | rnn | trf | ae

N_LANGKAH = 40      # 400 titik -> 40 langkah waktu x 10 fitur
EPOCH = 30
BATCH = 64
LR = 3e-3
HIDDEN = 64
DROPOUT = 0.2
CLIP = 1.0


def latih_dan_prediksi(seed: int, final: bool):
    """Kembalikan (pred_val, pred_test|None)."""
    import torch
    from torch import nn

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    Xtr, ytr, Xva, yva, Xte, _ = data.muat()

    def ke_seq(X):
        t = torch.tensor(np.asarray(X, dtype=np.float32))   # (N, 1, 400)
        return t.reshape(len(t), N_LANGKAH, -1)              # (N, 40, 10)

    n_kelas = 12   # kode label 0..11 (kipas x pompa x lampu)
    X = ke_seq(Xtr)
    y = torch.tensor(np.asarray(ytr), dtype=torch.long)
    Xv = ke_seq(Xva)
    yv = torch.tensor(np.asarray(yva), dtype=torch.long)

    class ModelRNN(nn.Module):
        def __init__(self, n_fitur: int, n_kelas: int):
            super().__init__()
            self.rnn = nn.GRU(n_fitur, HIDDEN, num_layers=1, batch_first=True)
            self.drop = nn.Dropout(DROPOUT)
            self.fc = nn.Linear(HIDDEN, n_kelas)

        def forward(self, x):
            _, h = self.rnn(x)
            return self.fc(self.drop(h[-1]))

    model = ModelRNN(X.shape[2], n_kelas)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(seed)

    for ep in range(EPOCH):
        model.train()
        perm = torch.randperm(len(X), generator=g)
        total, benar = 0.0, 0
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            logit = model(X[idx])
            loss = nn.functional.cross_entropy(logit, y[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            total += loss.item() * len(idx)
            benar += int((logit.argmax(1) == y[idx]).sum())
        model.eval()
        with torch.no_grad():
            acc_val = float((model(Xv).argmax(1) == yv).float().mean())
        print(f"epoch {ep + 1:02d}/{EPOCH} loss={total / len(X):.4f} "
              f"acc_latih={benar / len(X):.3f} acc_val={acc_val:.3f}", flush=True)

    model.eval()
    with torch.no_grad():
        pred_val = model(Xv).argmax(1).numpy()
        pred_test = model(ke_seq(Xte)).argmax(1).numpy() if final and Xte is not None else None
    return pred_val, pred_test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true", help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()
    t0 = time.time()
    pred_val, pred_test = latih_dan_prediksi(a.seed, a.final)
    durasi = time.time() - t0
    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", np.asarray(pred_val))
    perintah = [sys.executable, "-m", "src.common.evaluate", "--nim", NIM, "--model", JALUR, "--seed", str(a.seed),
                "--pred-val", str(out / "pred_val.npy"), "--durasi", f"{durasi:.1f}", "--perangkat", a.perangkat]
    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test))
        perintah += ["--pred-test", str(out / "pred_test.npy")]
    subprocess.run(perintah, check=True, cwd=AKAR)

    # F1 per alat (kipas, pompa, lampu) pada validasi -> metrik utama tim
    _, _, _, y_val, _, _ = data.muat()
    f1 = data.hitung_f1_alat(np.asarray(y_val), np.asarray(pred_val))
    (out / "f1_alat_val.json").write_text(json.dumps(f1, indent=2) + "\n", encoding="utf-8")
    print("F1 per alat (val):", json.dumps(f1))


if __name__ == "__main__":
    main()
