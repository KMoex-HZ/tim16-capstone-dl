"""Jalur Transformer Encoder (trf) — 123450071 (Khairunnisa Maharani).

    python models/123450071/run.py --seed 1 [--final]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402  (protokol tim yang dibekukan)

NIM = Path(__file__).resolve().parent.name     # 123450071
JALUR = "trf"                                   # jalur Transformer


def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Melatih Transformer Encoder 1D pada deret waktu arus & tegangan (400, 2)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr, ytr, Xva, _, Xte, _ = data.muat()
    n_kelas = int(max(ytr.max(), 7)) + 1

    class TimeSeriesTransformer(nn.Module):
        def __init__(
            self,
            seq_len: int = 400,
            in_ch: int = 2,
            patch_size: int = 10,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 128,
            dropout: float = 0.2,
            num_classes: int = 8,
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
            self.head = nn.Linear(d_model, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, 400, 2) -> bagi menjadi 40 patch berukuran (10 * 2)
            b, t, c = x.shape
            x = x.reshape(b, t // self.patch_size, self.patch_size * c)
            x = self.drop_in(self.patch_proj(x) + self.pos_embed)
            x = self.encoder(x)
            x = self.norm(x.mean(dim=1))  # global average pooling antar-token
            return self.head(x)

    model = TimeSeriesTransformer(num_classes=n_kelas).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    epochs = 35
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    ds_train = TensorDataset(
        torch.tensor(Xtr, dtype=torch.float32),
        torch.tensor(ytr, dtype=torch.long),
    )
    loader = DataLoader(ds_train, batch_size=64, shuffle=True)

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        va_tensor = torch.tensor(Xva, dtype=torch.float32, device=device)
        pred_val = model(va_tensor).argmax(dim=1).cpu().numpy().astype(np.int64)

        pred_test = None
        if final and Xte is not None:
            te_tensor = torch.tensor(Xte, dtype=torch.float32, device=device)
            pred_test = model(te_tensor).argmax(dim=1).cpu().numpy().astype(np.int64)

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