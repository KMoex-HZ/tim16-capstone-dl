"""Jalur Transformer Encoder (trf) — 123450071 (Khairunnisa Maharani).

    python models/123450071/run.py --seed 1 [--final]
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

NIM = Path(__file__).resolve().parent.name     # 123450071
JALUR = "trf"                                   # jalur Transformer


def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Melatih Transformer Encoder 1D pada deret tegangan rel bus_mV (1, 400)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr, ytr, Xva, yva, Xte, _ = data.muat()

    class TimeSeriesTransformer(nn.Module):
        def __init__(
            self,
            seq_len: int = 400,
            in_ch: int = 1,
            patch_size: int = 10,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 128,
            dropout: float = 0.2,
            num_logits: int = 5,  # 3 taraf kipas + 1 pompa + 1 lampu
        ):
            super().__init__()
            assert seq_len % patch_size == 0
            self.patch_size = patch_size
            num_patches = seq_len // patch_size

            self.patch_proj = nn.Linear(patch_size * in_ch, d_model)
            self.pos_embed = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)
            self.drop_in = nn.Dropout(dropout)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, num_logits)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, 1, 400) -> (B, 400, 1) -> 40 patch berukuran (10 * 1)
            b, c, t = x.shape
            x = x.transpose(1, 2).reshape(b, t // self.patch_size, self.patch_size * c)
            x = self.drop_in(self.patch_proj(x) + self.pos_embed)
            x = self.encoder(x)
            x = self.norm(x.mean(dim=1))
            return self.head(x)

    def dekode_kode(logits: torch.Tensor) -> np.ndarray:
        kipas = logits[:, :3].argmax(dim=1)
        pompa = (logits[:, 3] > 0).long()
        lampu = (logits[:, 4] > 0).long()
        return (kipas * 4 + pompa * 2 + lampu).cpu().numpy().astype(np.int64)

    model = TimeSeriesTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    epochs = 40
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ds_train = TensorDataset(
        torch.tensor(Xtr, dtype=torch.float32),
        torch.tensor(ytr, dtype=torch.long),
    )
    loader = DataLoader(ds_train, batch_size=64, shuffle=True)
    va_tensor = torch.tensor(Xva, dtype=torch.float32, device=device)

    skor_terbaik = -1.0
    bobot_terbaik = copy.deepcopy(model.state_dict())

    for _ in range(epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            y_kipas = torch.div(by, 4, rounding_mode="floor")
            y_biner = torch.stack([((by >> 1) & 1).float(), (by & 1).float()], dim=1)

            optimizer.zero_grad()
            out = model(bx)
            loss = F.cross_entropy(out[:, :3], y_kipas) + F.binary_cross_entropy_with_logits(out[:, 3:], y_biner)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            pred_va_ep = dekode_kode(model(va_tensor))
            skor_ep = data.hitung_f1_alat(yva, pred_va_ep)["val_f1_alat"]
            if skor_ep >= skor_terbaik:
                skor_terbaik = skor_ep
                bobot_terbaik = copy.deepcopy(model.state_dict())

    model.load_state_dict(bobot_terbaik)
    model.eval()
    with torch.no_grad():
        pred_val = dekode_kode(model(va_tensor))
        ringkasan = data.hitung_f1_alat(yva, pred_val)
        print(f"F1 per alat (validasi): {ringkasan['val_f1_alat']} | Rinci: {ringkasan['val_f1_rinci']}")

        pred_test = None
        if final and Xte is not None:
            te_tensor = torch.tensor(Xte, dtype=torch.float32, device=device)
            pred_test = dekode_kode(model(te_tensor))

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

    perintah = [
        sys.executable,
        "-m",
        "src.common.evaluate",
        "--nim",
        NIM,
        "--model",
        JALUR,
        "--seed",
        str(a.seed),
        "--pred-val",
        str(out / "pred_val.npy"),
        "--durasi",
        f"{durasi:.1f}",
        "--perangkat",
        a.perangkat,
    ]
    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test))
        perintah += ["--pred-test", str(out / "pred_test.npy")]

    subprocess.run(perintah, check=True, cwd=AKAR)


if __name__ == "__main__":
    main()