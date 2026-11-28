"""Jalur Autoencoder (ae) - 123450079 (Hafsa Fazila Arradhi).

Tahap 1: autoencoder konvolusi 1D dilatih tanpa label (rekonstruksi bus_mV).
Tahap 2: encoder dilatih lanjut (fine-tuning) bersama kepala linear untuk klasifikasi 12 kelas,
loss = klasifikasi + rekonstruksi; epoch terbaik dipilih dari val_f1_alat.

    python models/123450079/run.py --seed 1 [--final]
"""
from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402

NIM = Path(__file__).resolve().parent.name     # 123450079
JALUR = "ae"


def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr, ytr, Xva, yva, Xte, _ = data.muat()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)

    class AE(nn.Module):
        def __init__(self, z: int = 32):
            super().__init__()
            # (1, 400) -> (16, 200) -> (32, 100) -> (64, 50) -> z
            self.enc = nn.Sequential(
                nn.Conv1d(1, 16, 7, 2, 3), nn.ReLU(),
                nn.Conv1d(16, 32, 7, 2, 3), nn.ReLU(),
                nn.Conv1d(32, 64, 7, 2, 3), nn.ReLU(),
                nn.Flatten(), nn.Linear(64 * 50, z),
            )
            self.fc = nn.Linear(z, 64 * 50)
            self.dec = nn.Sequential(
                nn.ConvTranspose1d(64, 32, 8, 2, 3), nn.ReLU(),
                nn.ConvTranspose1d(32, 16, 8, 2, 3), nn.ReLU(),
                nn.ConvTranspose1d(16, 1, 8, 2, 3),
            )

        def forward(self, x):
            z = self.enc(x)
            return self.dec(F.relu(self.fc(z)).view(-1, 64, 50)), z

    def dekode_kode(logits: torch.Tensor) -> np.ndarray:
        kipas = logits[:, :3].argmax(dim=1)
        pompa = (logits[:, 3] > 0).long()
        lampu = (logits[:, 4] > 0).long()
        return (kipas * 4 + pompa * 2 + lampu).cpu().numpy().astype(np.int64)

    # ---- tahap 1: pelatihan autoencoder (tanpa label) ----
    ae = AE().to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    epoch_ae, batch = 40, 64
    mse_terbaik, bobot_ae = 1e9, copy.deepcopy(ae.state_dict())
    for _ in range(epoch_ae):
        ae.train()
        for i in torch.randperm(len(Xtr_t), device=device).split(batch):
            opt.zero_grad()
            rec, _ = ae(Xtr_t[i])
            F.mse_loss(rec, Xtr_t[i]).backward()
            opt.step()
        ae.eval()
        with torch.no_grad():
            mse_val = F.mse_loss(ae(Xva_t)[0], Xva_t).item()
        if mse_val < mse_terbaik:
            mse_terbaik, bobot_ae = mse_val, copy.deepcopy(ae.state_dict())
    ae.load_state_dict(bobot_ae)
    ae.eval()
    print(f"MSE rekonstruksi validasi terbaik: {mse_terbaik:.4f}")

    # ---- tahap 2: encoder ikut dilatih bersama kepala linear (loss klasifikasi + rekonstruksi) ----
    kepala = nn.Linear(32, 5).to(device)              # 3 taraf kipas + pompa + lampu
    opt2 = torch.optim.Adam(list(ae.parameters()) + list(kepala.parameters()), lr=1e-3, weight_decay=1e-4)
    y_kipas = torch.div(ytr_t, 4, rounding_mode="floor")
    y_biner = torch.stack([((ytr_t >> 1) & 1).float(), (ytr_t & 1).float()], dim=1)
    skor_terbaik = -1.0
    bobot_terbaik = (copy.deepcopy(ae.state_dict()), copy.deepcopy(kepala.state_dict()))
    for _ in range(60):
        ae.train()
        kepala.train()
        for i in torch.randperm(len(Xtr_t), device=device).split(batch):
            opt2.zero_grad()
            rec, z = ae(Xtr_t[i])
            out = kepala(z)
            loss = (F.cross_entropy(out[:, :3], y_kipas[i])
                    + F.binary_cross_entropy_with_logits(out[:, 3:], y_biner[i])
                    + F.mse_loss(rec, Xtr_t[i]))
            loss.backward()
            opt2.step()
        ae.eval()
        kepala.eval()
        with torch.no_grad():
            skor = data.hitung_f1_alat(yva, dekode_kode(kepala(ae.enc(Xva_t))))["val_f1_alat"]
        if skor >= skor_terbaik:
            skor_terbaik = skor
            bobot_terbaik = (copy.deepcopy(ae.state_dict()), copy.deepcopy(kepala.state_dict()))
    ae.load_state_dict(bobot_terbaik[0])
    kepala.load_state_dict(bobot_terbaik[1])
    ae.eval()
    kepala.eval()

    with torch.no_grad():
        pred_val = dekode_kode(kepala(ae.enc(Xva_t)))
        ringkasan = data.hitung_f1_alat(yva, pred_val)
        print(f"F1 per alat (validasi): {ringkasan['val_f1_alat']} | Rinci: {ringkasan['val_f1_rinci']}")
        pred_test = None
        if final and Xte is not None:
            pred_test = dekode_kode(kepala(ae.enc(torch.tensor(Xte, dtype=torch.float32, device=device))))
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


if __name__ == "__main__":
    main()
