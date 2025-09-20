#!/usr/bin/env python3
"""
StockMixer-ATFNet (Extended)
============================
Adds:
- Optuna hyperparameter search (--optuna_trials > 0)
- Per-date cross-sectional IC (mean/median, distribution)
- Precision@N and Sharpe for simple long-top-N portfolio
- CSV logging of per-date metrics and portfolio returns
- Keeps minimal dual-branch Time+Freq architecture

Quick starts:
  # Smoke test on synthetic:
  python train_stockmixer_atfnet_ext.py --generate_synthetic 1 --epochs 3

  # With Optuna (opt for IC):
  python train_stockmixer_atfnet_ext.py --generate_synthetic 1 --optuna_trials 20 --epochs 5

Data format:
- Columns: date, symbol, <numeric features...>, target_1d
- 'target_1d' is the *future* return aligned to prediction time t.
"""
from __future__ import annotations
import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ----------------------
# Utils
# ----------------------

def spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float('nan')
    return float(stats.spearmanr(x, y, nan_policy="omit").correlation)

def seed_all(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def pick_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ----------------------
# Dataset
# ----------------------

META_COLS = {"date", "symbol"}

class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        lookback: int,
        horizon: int = 1,
    ):
        """
        Build windows over features [t-lookback, ..., t-1] to predict target at time t.
        Returns x, y, date_t, symbol_t for cross-sectional metrics.
        """
        assert horizon >= 1
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.lookback = lookback
        self.horizon = horizon

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values(["symbol", "date"], inplace=True)

        X, y, dates, syms = [], [], [], []
        for sym, g in df.groupby("symbol", sort=False):
            g = g.reset_index(drop=True)
            for t in range(lookback, len(g)):
                x_window = g.loc[t - lookback : t - 1, feature_cols].to_numpy(dtype=np.float32)
                target = g.loc[t, target_col]
                if np.isfinite(x_window).all() and np.isfinite(target):
                    X.append(x_window)
                    y.append(np.float32(target))
                    dates.append(np.asarray(g.loc[t, "date"], dtype="datetime64[ns]"))
                    syms.append(str(sym))
        self.X = np.stack(X, axis=0) if len(X) else np.zeros((0, lookback, len(feature_cols)), dtype=np.float32)
        self.y = np.array(y, dtype=np.float32) if len(y) else np.zeros((0,), dtype=np.float32)
        self.dates = np.array(dates, dtype="datetime64[ns]") if len(dates) else np.array([], dtype="datetime64[ns]")
        self.syms = np.array(syms, dtype=object) if len(syms) else np.array([], dtype=object)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx]              # [T, F]
        x = np.transpose(x, (1, 0))  # [F, T] for Conv1d
        return (
            torch.from_numpy(x),
            torch.tensor(self.y[idx]),
            self.dates[idx],
            self.syms[idx],
        )

# ----------------------
# Model
# ----------------------

class TimeMixer(nn.Module):
    def __init__(self, in_ch: int, hidden: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size=kernel_size, padding=pad, groups=in_ch)
        self.pw1 = nn.Conv1d(in_ch, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv1d(hidden, in_ch, kernel_size=1)
        self.norm = nn.BatchNorm1d(in_ch)

    def forward(self, x):  # [B, C, T]
        residual = x
        x = self.dw(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.norm(x + residual)
        return x

class FreqMixer(nn.Module):
    def __init__(self, in_ch: int, time_len: int, hidden: int):
        super().__init__()
        freq_bins = time_len // 2 + 1
        self.in_dim = in_ch * freq_bins * 2
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

    def forward(self, x):  # [B, C, T]
        Xf = torch.fft.rfft(x, dim=-1)
        z = torch.cat([Xf.real, Xf.imag], dim=1)  # [B, 2C, F]
        z = z.reshape(z.size(0), -1)
        return self.mlp(z)

class ATFNetHead(nn.Module):
    def __init__(self, in_ch: int, time_len: int, hidden: int, head_hidden: int):
        super().__init__()
        self.time_mixer = TimeMixer(in_ch, hidden)
        self.freq_mixer = FreqMixer(in_ch, time_len, hidden)
        self.readout_time = nn.AdaptiveAvgPool1d(1)
        fuse_in = in_ch + hidden
        self.head = nn.Sequential(
            nn.Linear(fuse_in, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x):  # [B, C, T]
        t_branch = self.time_mixer(x)
        t_vec = self.readout_time(t_branch).squeeze(-1)
        f_vec = self.freq_mixer(x)
        fused = torch.cat([t_vec, f_vec], dim=1)
        return self.head(fused).squeeze(-1)

# ----------------------
# Metrics
# ----------------------

def cross_sectional_metrics(preds: np.ndarray, trues: np.ndarray, dates: np.ndarray, syms: np.ndarray,
                            top_n: int = 20) -> Dict[str, float]:
    """Compute per-date IC, Precision@N, and Sharpe for long-top-N equal-weight portfolio."""
    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "symbol": syms,
        "pred": preds,
        "target": trues,
    })
    # group by date
    ics = []
    daily_ret = []   # equal-weight long top-N
    precs = []
    for d, g in df.groupby("date", sort=True):
        if len(g) < 2:
            continue
        ic = spearman_ic(g["pred"].values, g["target"].values)
        ics.append(ic)
        # top-N by pred
        k = min(top_n, len(g))
        top = g.nlargest(k, "pred")
        # Precision@N: fraction of positives among top
        prec = (top["target"].values > 0).mean() if k > 0 else np.nan
        precs.append(prec)
        # Portfolio return: mean of targets in top-N
        daily_ret.append(top["target"].mean() if k > 0 else np.nan)

    ics = np.array([x for x in ics if np.isfinite(x)])
    precs = np.array([x for x in precs if np.isfinite(x)])
    daily_ret = np.array([x for x in daily_ret if np.isfinite(x)])

    mean_ic = float(np.nanmean(ics)) if ics.size else float("nan")
    med_ic = float(np.nanmedian(ics)) if ics.size else float("nan")
    std_ic = float(np.nanstd(ics, ddof=1)) if ics.size > 1 else float("nan")
    mean_prec = float(np.nanmean(precs)) if precs.size else float("nan")

    # Sharpe (daily to annualized with sqrt(252))
    if daily_ret.size > 1:
        mu = float(np.nanmean(daily_ret))
        sig = float(np.nanstd(daily_ret, ddof=1))
        sharpe = float((mu / (sig + 1e-12)) * math.sqrt(252.0))
    else:
        sharpe = float("nan")

    return {
        "mean_ic": mean_ic,
        "median_ic": med_ic,
        "std_ic": std_ic,
        "precision_at_n": mean_prec,
        "sharpe_top_n": sharpe,
        "days": int(len(ics)),
    }

def dump_per_date_csv(path: str, preds: np.ndarray, trues: np.ndarray, dates: np.ndarray, syms: np.ndarray, top_n: int):
    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "symbol": syms,
        "pred": preds,
        "target": trues,
    })
    # rank per date
    df["rank"] = df.groupby("date")["pred"].rank(ascending=False, method="first")
    df["in_topN"] = (df["rank"] <= top_n).astype(int)
    out = path
    df.to_csv(out, index=False)
    return out

# ----------------------
# Training / Eval
# ----------------------

@dataclass
class TrainConfig:
    lookback: int = 16
    batch_size: int = 256
    epochs: int = 5
    lr: float = 1e-3
    hidden: int = 128
    head_hidden: int = 128
    target_col: str = "target_1d"
    num_workers: int = 0
    top_n: int = 20

def train_one_epoch(model, loader, device, optimizer):
    model.train()
    total = 0.0
    n = 0
    for xb, yb, _, _ in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = torch.nn.functional.mse_loss(pred, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)

@torch.no_grad()
def evaluate_collect(model, loader, device):
    model.eval()
    preds, trues, dates, syms = [], [], [], []
    for xb, yb, db, sb in loader:
        xb = xb.to(device)
        p = model(xb).detach().cpu().numpy().ravel()
        preds.append(p)
        trues.append(yb.numpy().ravel())
        dates.append(db.astype("datetime64[ns]"))
        syms.append(np.array(sb, dtype=object))
    if not preds:
        return np.array([]), np.array([]), np.array([]), np.array([])
    return (np.concatenate(preds),
            np.concatenate(trues),
            np.concatenate(dates),
            np.concatenate(syms))

# ----------------------
# Data helpers
# ----------------------

def load_frame(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)

def infer_feature_cols(df: pd.DataFrame, target_col: str) -> List[str]:
    cand = [c for c in df.columns if c not in META_COLS | {target_col}]
    num = [c for c in cand if pd.api.types.is_numeric_dtype(df[c])]
    return num

def chronological_split(df: pd.DataFrame, val_frac: float = 0.2) -> Tuple[pd.DataFrame,pd.DataFrame]:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    dates_sorted = np.sort(df["date"].unique())
    split_idx = int((1.0 - val_frac) * len(dates_sorted))
    split_date = dates_sorted[max(split_idx, 1) - 1]
    train = df[df["date"] <= split_date]
    val = df[df["date"] > split_date]
    return train, val

# ----------------------
# Synthetic data (optional)
# ----------------------

def make_synthetic_csv(path: str, n_symbols=30, n_days=400, n_features=16, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n_days, freq="B")
    rows = []
    for s in range(n_symbols):
        sym = f"S{s:03d}"
        base = rng.normal(0, 0.01, size=n_days)
        signal = np.convolve(base, np.ones(5)/5.0, mode="same")
        feats = rng.normal(0, 1, size=(n_days, n_features))
        for t in range(n_days):
            row = {"date": dates[t], "symbol": sym}
            for f in range(n_features):
                row[f"feat_{f}"] = feats[t, f] + 0.3*signal[t] + 0.05*rng.normal()
            row["target_1d"] = signal[t] + 0.01*rng.normal()
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path

# ----------------------
# Optuna
# ----------------------

def optuna_objective(trial, base_args, train_df, val_df, features, device):
    lookback = trial.suggest_int("lookback", 8, 32, step=4)
    hidden = trial.suggest_int("hidden", 64, 256, step=64)
    head_hidden = trial.suggest_int("head_hidden", 64, 256, step=64)
    lr = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    top_n = trial.suggest_categorical("top_n", [10, 20, 30, 50])

    # Rebuild datasets for the chosen lookback
    train_ds = SlidingWindowDataset(train_df, features, base_args.target_col, lookback)
    val_ds   = SlidingWindowDataset(val_df,   features, base_args.target_col, lookback)

    if len(train_ds) < 10 or len(val_ds) < 10:
        return 0.0

    F = len(features)
    model = ATFNetHead(F, lookback, hidden, head_hidden).to(device)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=base_args.workers, pin_memory=(device.type=="cuda"))
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=base_args.workers, pin_memory=(device.type=="cuda"))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(base_args.epochs):
        _ = train_one_epoch(model, train_dl, device, opt)

    preds, trues, dates, syms = evaluate_collect(model, val_dl, device)
    metrics = cross_sectional_metrics(preds, trues, dates, syms, top_n=top_n)
    # Maximize mean IC
    return metrics.get("mean_ic", 0.0) if math.isfinite(metrics.get("mean_ic", float("nan"))) else 0.0

# ----------------------
# Main
# ----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="", help="CSV/Parquet path")
    parser.add_argument("--target_col", type=str, default="target_1d")
    parser.add_argument("--lookback", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--head_hidden", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--generate_synthetic", type=int, default=0)
    parser.add_argument("--top_n", type=int, default=20, help="Top-N for Prec@N and portfolio")
    parser.add_argument("--optuna_trials", type=int, default=0, help="If >0, run hyperopt to maximize mean IC")
    parser.add_argument("--per_date_csv", type=str, default="/mnt/data/per_date_metrics.csv")
    args = parser.parse_args()

    seed_all(123)
    device = pick_device(args.device)
    print(f"Using device: {device}")

    data_path = args.data
    if args.generate_synthetic or not data_path:
        out = "/mnt/data/synthetic_stockmixer.csv"
        make_synthetic_csv(out)
        data_path = out
        print(f"[synthetic] Wrote toy dataset to {out}")

    df = load_frame(data_path)
    assert {"date","symbol",args.target_col}.issubset(df.columns), "Data must contain date, symbol, and target column"

    features = infer_feature_cols(df, args.target_col)
    print(f"Detected {len(features)} feature columns")

    train_df, val_df = chronological_split(df, args.val_frac)

    # If Optuna requested
    if args.optuna_trials > 0:
        import optuna
        print(f"Starting Optuna: trials={args.optuna_trials}")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: optuna_objective(trial, args, train_df, val_df, features, device),
                       n_trials=args.optuna_trials, show_progress_bar=False)
        print("Best params:", study.best_trial.params)
        # Override args with best
        for k, v in study.best_trial.params.items():
            setattr(args, k, v)

    # Build final datasets/loaders with (possibly tuned) params
    train_ds = SlidingWindowDataset(train_df, features, args.target_col, args.lookback)
    val_ds   = SlidingWindowDataset(val_df,   features, args.target_col, args.lookback)

    print(f"Train windows: {len(train_ds)} | Val windows: {len(val_ds)} | Lookback={args.lookback} | F={len(features)}")

    F = len(features)
    model = ATFNetHead(in_ch=F, time_len=args.lookback, hidden=args.hidden, head_hidden=args.head_hidden).to(device)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.workers, pin_memory=(device.type=="cuda"))
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=(device.type=="cuda"))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_ic = -999.0
    for epoch in range(1, args.epochs+1):
        train_loss = train_one_epoch(model, train_dl, device, opt)
        preds, trues, dates, syms = evaluate_collect(model, val_dl, device)
        metrics = cross_sectional_metrics(preds, trues, dates, syms, top_n=args.top_n)
        val_ic = metrics["mean_ic"]
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.5f} | "
              f"val_IC_mean={val_ic:.4f} | val_IC_med={metrics['median_ic']:.4f} | "
              f"Prec@{args.top_n}={metrics['precision_at_n']:.3f} | Sharpe@{args.top_n}={metrics['sharpe_top_n']:.2f} | days={metrics['days']}")
        best_ic = max(best_ic, val_ic)

    # Dump per-date details for analysis
    out_csv = dump_per_date_csv(args.per_date_csv, preds, trues, dates, syms, top_n=args.top_n)
    print(f"Wrote per-date metrics to {out_csv}")
    print(f"Done. Best validation mean IC: {best_ic:.4f}")

if __name__ == "__main__":
    main()
