"""Minimal 1D training example for FFNO, R2-FFNO and Decomposed-FNO.

Learns a synthetic map f(u) = d^2 u / dx^2 on a 1D grid.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import FFNO, R2FFNO, DecomposedFNO

OUT_DIR = Path(__file__).resolve().parent / "outputs"


def standardize(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Zero-mean / unit-std per sample along the spatial axis ([B, C, N])."""
    mean = t.mean(dim=-1, keepdim=True)
    std = t.std(dim=-1, keepdim=True).clamp_min(eps)
    return (t - mean) / std


def make_batch(batch: int, n: int, device: torch.device):
    """Random Fourier series u(x) and target u_xx (both standardized)."""
    x = torch.linspace(0, 1, n, device=device).view(1, 1, n)
    k = torch.arange(1, 9, device=device).view(1, -1, 1)
    amp = torch.randn(batch, k.shape[1], 1, device=device)
    phase = 2 * torch.pi * torch.rand(batch, k.shape[1], 1, device=device)
    u = (amp * torch.sin(2 * torch.pi * k * x + phase)).sum(dim=1, keepdim=True)
    u_xx = (
        amp * (-((2 * torch.pi * k) ** 2)) * torch.sin(2 * torch.pi * k * x + phase)
    ).sum(dim=1, keepdim=True)
    return standardize(u), standardize(u_xx)


def plot_prediction_vs_truth(
    model_name: str,
    x: torch.Tensor,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    mse: float,
    out_path: Path,
):
    """Save prediction vs ground truth for the first test sample."""
    xs = x.detach().cpu().numpy().ravel()
    truth = y_true[0, 0].detach().cpu().numpy()
    pred = y_pred[0, 0].detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    ax.plot(xs, truth, label="Ground truth", linewidth=2)
    ax.plot(xs, pred, label="Prediction", linewidth=2, linestyle="--")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$u_{xx}$")
    ax.set_title(f"{model_name.upper()} — prediction vs ground truth (MSE={mse:.4e})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved plot: {out_path}")


def train_one(model_name: str, epochs: int, n: int, device: torch.device):
    common = dict(
        n_modes=[16],
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
        n_layers=2,
        spatial_dims=1,
        padding=4,
    )
    if model_name == "ffno":
        model = FFNO(**common)
    elif model_name == "r2ffno":
        model = R2FFNO(**common, reduce_k=8)
    elif model_name == "dfno":
        model = DecomposedFNO(
            **{k: v for k, v in common.items() if k != "padding"},
            decomposition_terms=8,
            grid_size=(n,),
        )
    else:
        raise ValueError(model_name)

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"\n=== {model_name.upper()} | grid={n} | device={device} ===")
    model.train()
    for epoch in range(1, epochs + 1):
        u, y = make_batch(16, n, device)
        pred = model(u)
        loss = F.mse_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch == 1 or epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            print(f"epoch {epoch:03d}  mse={loss.item():.4e}")

    model.eval()
    with torch.no_grad():
        u, y = make_batch(32, n, device)
        pred = model(u)
        mse = F.mse_loss(pred, y).item()
    print(f"test mse={mse:.4e}")

    x_coords = torch.linspace(0, 1, n, device=device)
    plot_prediction_vs_truth(
        model_name=model_name,
        x=x_coords,
        y_true=y,
        y_pred=pred,
        mse=mse,
        out_path=OUT_DIR / f"{model_name}_pred_vs_truth.png",
    )
    return mse


def main():
    p = argparse.ArgumentParser(description="1D example: FFNO / R2-FFNO / DFNO")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--n", type=int, default=64, help="1D grid size")
    p.add_argument(
        "--model",
        choices=["ffno", "r2ffno", "dfno", "all"],
        default="all",
    )
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    names = ["ffno", "r2ffno", "dfno"] if args.model == "all" else [args.model]
    for name in names:
        train_one(name, args.epochs, args.n, device)


if __name__ == "__main__":
    main()
