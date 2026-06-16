"""Train the MLP regression head on pre-computed CLAP embeddings.

Workflow
--------
1. Run `training/generate_embeddings.py` once to build the .npz corpus.
2. Run this script (many times / hyperparameter sweeps) on the cached data.

Usage
-----
    python -m training.train --synth fm --data data/fm.npz \\
        --epochs 200 --lr 3e-4 --batch 256 --out checkpoints/fm_head.pt

The checkpoint contains the full ClapMlpHead state dict so it can be
loaded directly with `torch.load` for inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


def _train(
    synth_type: str,
    data_path: str | Path,
    out_path: str | Path,
    epochs: int = 200,
    lr: float = 3e-4,
    batch_size: int = 256,
    val_split: float = 0.1,
    hidden: tuple[int, ...] = (256, 128),
    dropout: float = 0.3,
    seed: int = 42,
) -> None:
    torch.manual_seed(seed)
    device = (
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()         else
        "cpu"
    )
    print(f"Device: {device}")

    data = np.load(data_path)
    X = torch.from_numpy(data["embeddings"]).float()
    Y = torch.from_numpy(data["params"]).float()
    assert X.shape[0] == Y.shape[0], "embedding / param count mismatch"
    print(f"Loaded {X.shape[0]:,} examples  "
          f"(emb {X.shape[1]}-dim → {Y.shape[1]} params)")

    dataset = TensorDataset(X, Y)
    n_val   = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size * 2, shuffle=False,
                              num_workers=0)

    from models.clap_head import ClapMlpHead
    model = ClapMlpHead(synth_type=synth_type, hidden=hidden, dropout=dropout)
    model.to(device)

    # Only train the MLP head; CLAP is frozen and not on device.
    opt   = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        # --- val ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += loss_fn(model(xb), yb).item() * len(xb)
        val_loss /= n_val
        sched.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3}/{epochs}  "
                  f"train={train_loss:.5f}  val={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "synth_type": synth_type,
                "hidden":     hidden,
                "dropout":    dropout,
                "epoch":      epoch,
                "val_loss":   best_val,
                "state_dict": model.state_dict(),
            }, out_path)

    print(f"\nBest val MSE: {best_val:.5f}")
    print(f"Checkpoint → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train MLP regression head on pre-computed embeddings.")
    ap.add_argument("--synth",   required=True, choices=["fm", "fm2op", "wavetable", "subtractive"])
    ap.add_argument("--data",    required=True, help="Path to .npz from generate_embeddings.py")
    ap.add_argument("--out",     default="checkpoints/model.pt")
    ap.add_argument("--epochs",  type=int,   default=200)
    ap.add_argument("--lr",      type=float, default=3e-4)
    ap.add_argument("--batch",   type=int,   default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed",    type=int,   default=42)
    args = ap.parse_args()

    _train(
        synth_type=args.synth,
        data_path=args.data,
        out_path=args.out,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        dropout=args.dropout,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
