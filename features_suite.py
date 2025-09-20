#!/usr/bin/env python3
"""
features_suite.py
=================
Comprehensive, leak-safe feature creation, analysis, and pruning for equity time series.

Inputs
------
- Parquet/CSV with columns: date, symbol, (optional) sector, OHLCV or at least adj_close & volume, plus target_1d.
- `target_1d` is the *future* 1-day return aligned at time t (trainer uses window [t-L..t-1] -> target at t).

Outputs
-------
- A pruned Parquet with meta + selected features + target.
- CSV report of feature quality (per-date cross-sectional metrics).
- JSON summary of pipeline settings and results.
- TXT file listing selected features (one per line).

Usage
-----
python features_suite.py \
  --data /mnt/data/yf_us_2015_present.parquet \
  --out_parquet /mnt/data/yf_us_2015_present_features_pruned.parquet \
  --report_csv /mnt/data/feature_report.csv \
  --summary_json /mnt/data/feature_summary.json \
  --selected_txt /mnt/data/selected_features.txt \
  --xs_mode z --sector_neutral 0 --ic_metric spearman \
  --keep_top_k 128 --ic_min 0.002 --corr_max 0.92 --vif_max 15.0

Notes
-----
- All rolling features are shifted by 1 bar to remain point-in-time safe.
- Cross-sectional transforms (z/rank) operate per date (optionally within sector).
- Correlation pruning is done on *train* split only; same selection is applied to full data (train+val).
- IC metrics are computed per date on the train split and aggregated (mean, ICIR).
"""

from __future__ import annotations
import argparse, json, math, os
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

META_COLS = ["date", "symbol", "sector"]
REQUIRED_COLS_ANY = [["adj_close", "volume"], ["close", "volume"]]  # allow either; prefer adj_close
TARGET_COL_DEFAULT = "target_1d"

# -------------------------
# Utilities
# -------------------------

def _has_cols(df: pd.DataFrame, cols: List[str]) -> bool:
    return all(c in df.columns for c in cols)

def _pick_price_col(df: pd.DataFrame) -> str:
    if "adj_close" in df.columns: return "adj_close"
    if "close" in df.columns: return "close"
    raise ValueError("Need adj_close or close column")

def _spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2: return np.nan
    return float(stats.spearmanr(x, y, nan_policy="omit").correlation)

def _pearson_ic(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2: return np.nan
    return float(stats.pearsonr(x, y)[0])

def _safe_div(a: pd.Series, b: pd.Series, eps: float = 1e-12) -> pd.Series:
    return a / (b.replace(0, np.nan) + eps)

def _group_shifted_roll(g: pd.core.groupby.generic.SeriesGroupBy, win: int, fn: str, minp: int) -> pd.Series:
    # Compute rolling then shift by 1 (past only)
    roll = getattr(g.rolling(win, min_periods=minp), fn)().reset_index(level=0, drop=True)
    return roll.shift(1)

def _winsorize(s: pd.Series, limits: Tuple[float,float]=(0.005,0.995)) -> pd.Series:
    a, b = s.quantile(limits[0]), s.quantile(limits[1])
    return s.clip(lower=a, upper=b)

def _vif(df: pd.DataFrame) -> pd.Series:
    # Simple/fast VIF approximation using R^2 from linear regression of each column on others
    cols = list(df.columns)
    vals = df.values.astype(np.float64)
    out = {}
    for i, c in enumerate(cols):
        X = np.delete(vals, i, axis=1)
        y = vals[:, i]
        # Standardize to stabilize
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
        ys = (y - y.mean()) / (y.std() + 1e-12)
        lr = LinearRegression(fit_intercept=True)
        lr.fit(Xs, ys)
        r2 = max(0.0, min(1.0, lr.score(Xs, ys)))
        out[c] = 1.0 / (1.0 - r2 + 1e-12)
    return pd.Series(out)

# -------------------------
# Feature creation (leak-safe)
# -------------------------

def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    price = _pick_price_col(df)
    g_price = df.groupby("symbol")[price]
    g_vol = df.groupby("symbol")["volume"] if "volume" in df.columns else None

    # 1-step returns
    df["ret_1"] = g_price.pct_change(1)
    df["logret_1"] = np.log(g_price / g_price.shift(1))

    # Rolling vol & momentum on returns
    for win in (5,10,20,60):
        df[f"vol_{win}"] = _group_shifted_roll(df.groupby("symbol")["ret_1"], win, "std", minp=3)
        df[f"mom_sum_{win}"] = _group_shifted_roll(df.groupby("symbol")["ret_1"], win, "sum", minp=3)
        df[f"mom_mean_{win}"] = _group_shifted_roll(df.groupby("symbol")["ret_1"], win, "mean", minp=3)

    # Moving averages and ratios
    for win in (5,10,20,50,100,200):
        ma = _group_shifted_roll(g_price, win, "mean", minp=max(3, win//5))
        df[f"sma_{win}"] = ma
        df[f"prc_div_sma_{win}"] = _safe_div(df[price], ma)

    # RSI (Wilder) - simplified
    win = 14
    delta = g_price.diff()
    up = delta.clip(lower=0.0)
    down = (-delta.clip(upper=0.0))
    roll_up = _group_shifted_roll(up.groupby(df["symbol"]), win, "mean", minp=3)
    roll_down = _group_shifted_roll(down.groupby(df["symbol"]), win, "mean", minp=3)
    rs = roll_up / (roll_down + 1e-12)
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # MACD (12,26) and signal(9) (EMA, shifted)
    span_fast, span_slow, span_sig = 12, 26, 9
    ema_fast = g_price.transform(lambda s: s.ewm(span=span_fast, adjust=False).mean()).shift(1)
    ema_slow = g_price.transform(lambda s: s.ewm(span=span_slow, adjust=False).mean()).shift(1)
    macd = ema_fast - ema_slow
    macd_sig = macd.groupby(df["symbol"]).transform(lambda s: s.ewm(span=span_sig, adjust=False).mean())
    df["macd"] = macd
    df["macd_signal"] = macd_sig

    # Volume-based
    if g_vol is not None:
        for win in (5,20,60):
            df[f"volu_mean_{win}"] = _group_shifted_roll(g_vol, win, "mean", minp=3)
        df["vol_ret"] = g_vol.pct_change(1).shift(1)

    # Missingness flags (can be predictive of illiquidity regimes)
    num_cols = [c for c in df.columns if c not in META_COLS]
    for c in num_cols:
        df[f"isnan_{c}"] = df[c].isna().astype(np.float32)

    return df

def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    # A few cheap interactions
    if "vol_20" in df and "mom_mean_20" in df:
        df["volxmom_20"] = df["vol_20"] * df["mom_mean_20"]
    if "prc_div_sma_20" in df and "rsi_14" in df:
        df["valxmom_20"] = df["prc_div_sma_20"] * (df["rsi_14"] - 50.0)
    return df

# -------------------------
# Cross-sectional transforms (per-date), optional sector neutrality
# -------------------------

def xs_transform(df: pd.DataFrame, cols: List[str], mode: str = "z", by_sector: bool = False) -> pd.DataFrame:
    """
    mode: 'z' -> z-score per date (and sector if by_sector)
          'rank' -> rank-normalize to [0,1] per date (and sector if by_sector)
          'none' -> pass-through
    """
    if mode not in ("z", "rank", "none"):
        raise ValueError("xs_mode must be one of: z, rank, none")
    if mode == "none":
        return df

    def _apply(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        for c in cols:
            s = g[c].astype(float)
            if mode == "z":
                s = (s - s.mean()) / (s.std(ddof=1) + 1e-12)
            else:
                r = s.rank(method="average", na_option="keep")
                s = (r - 1) / (r.count() - 1 + 1e-12)
            g[c] = s
        return g

    if by_sector and "sector" in df.columns:
        return df.groupby(["date", "sector"], group_keys=False).apply(_apply)
    else:
        return df.groupby("date", group_keys=False).apply(_apply)

def sector_neutralize(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Regress out sector dummies per date (OLS) and keep residuals.
    No leakage: uses only contemporaneous (at time t) sector labels and feature values.
    """
    if "sector" not in df.columns:
        return df
    out = df.copy()
    for d, g in df.groupby("date"):
        if g["sector"].nunique() < 2:
            continue
        dummies = pd.get_dummies(g["sector"], drop_first=False)
        X = dummies.values
        for c in cols:
            y = g[c].values.astype(np.float64)
            mask = np.isfinite(y)
            if mask.sum() < 3:
                continue
            lr = LinearRegression(fit_intercept=True)
            lr.fit(X[mask], y[mask])
            y_hat = lr.predict(X[mask])
            resid = y.copy()
            resid[mask] = y[mask] - y_hat
            out.loc[g.index, c] = resid
    return out

# -------------------------
# Pruning
# -------------------------

def prune_zero_variance(df: pd.DataFrame, cols: List[str]) -> List[str]:
    keep = []
    for c in cols:
        s = df[c].astype(float)
        if s.var(skipna=True) > 0.0:
            keep.append(c)
    return keep

def prune_missingness(df: pd.DataFrame, cols: List[str], max_na_frac: float = 0.2) -> List[str]:
    keep = []
    for c in cols:
        na_frac = df[c].isna().mean()
        if na_frac <= max_na_frac:
            keep.append(c)
    return keep

def corr_prune(df_train: pd.DataFrame, cols: List[str], corr_max: float = 0.95) -> List[str]:
    """
    Greedy correlation prune using absolute Pearson corr on *train* split.
    Keeps one representative from highly correlated clusters.
    """
    X = df_train[cols].astype(float).fillna(0.0)
    corr = np.corrcoef(X.values, rowvar=False)
    keep = []
    removed = set()
    col_idx = {i: c for i, c in enumerate(cols)}
    for i in range(len(cols)):
        if i in removed:
            continue
        keep.append(col_idx[i])
        for j in range(i+1, len(cols)):
            if j in removed: continue
            if abs(corr[i, j]) >= corr_max:
                removed.add(j)
    return keep

def vif_prune(df_train: pd.DataFrame, cols: List[str], vif_max: float = 20.0) -> List[str]:
    keep = cols[:]
    changed = True
    while changed and len(keep) > 2:
        v = _vif(df_train[keep].astype(float).fillna(0.0))
        worst = v.idxmax()
        if v[worst] > vif_max:
            keep.remove(worst)
        else:
            changed = False
    return keep

# -------------------------
# Feature quality (train only)
# -------------------------

def per_feature_ic(df_train: pd.DataFrame, cols: List[str], target_col: str, ic_metric: str = "spearman") -> pd.DataFrame:
    # Compute per-date cross-sectional IC for each feature, aggregate mean and ICIR (mean/std)
    if ic_metric not in ("spearman", "pearson"):
        raise ValueError("ic_metric must be 'spearman' or 'pearson'")
    rows = []
    for c in cols:
        daily = []
        for d, g in df_train[["date", c, target_col]].dropna().groupby("date"):
            x = g[c].values
            y = g[target_col].values
            if x.size < 3:
                continue
            if ic_metric == "spearman":
                ic = _spearman_ic(x, y)
            else:
                ic = _pearson_ic(x, y)
            if np.isfinite(ic):
                daily.append(ic)
        if len(daily) == 0:
            mean_ic = np.nan
            icir = np.nan
        else:
            mean_ic = float(np.nanmean(daily))
            std_ic = float(np.nanstd(daily, ddof=1)) if len(daily) > 1 else np.nan
            icir = float(mean_ic / (std_ic + 1e-12)) if np.isfinite(std_ic) else np.nan
        rows.append({"feature": c, "mean_ic": mean_ic, "icir": icir, "n_days": len(daily)})
    rep = pd.DataFrame(rows).sort_values(["mean_ic","icir"], ascending=[False, False])
    return rep

# -------------------------
# Pipeline
# -------------------------

@dataclass
class SuiteConfig:
    xs_mode: str = "z"             # 'z' | 'rank' | 'none'
    by_sector: int = 0             # 1 to apply per-sector transforms
    sector_neutral: int = 0        # 1 to regress out sector effect per date
    max_na_frac: float = 0.2
    corr_max: float = 0.95
    vif_max: float = 20.0
    keep_top_k: int = 256
    ic_min: float = -1.0           # optional floor on mean IC
    ic_metric: str = "spearman"    # 'spearman' | 'pearson'
    val_frac: float = 0.2
    target_col: str = TARGET_COL_DEFAULT

def run_suite(df: pd.DataFrame, cfg: SuiteConfig) -> Dict[str, object]:
    # --- basic checks
    if not any(_has_cols(df, req) for req in REQUIRED_COLS_ANY):
        raise ValueError("Input must include (adj_close, volume) or (close, volume)")
    if cfg.target_col not in df.columns:
        raise ValueError(f"Missing target column '{cfg.target_col}'")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol","date"])

    # --- feature creation
    df = add_base_features(df)
    df = add_interactions(df)

    # collect candidate numeric features (exclude meta + target)
    cand = [c for c in df.columns if c not in META_COLS + [cfg.target_col]]
    # exclude raw OHLCV price columns if present (we already used them)
    drop_raw = ["open","high","low","close","adj_close","volume"]
    cand = [c for c in cand if c not in drop_raw]

    # --- cross-sectional transforms
    df[cand] = df[cand].astype(float)
    df = xs_transform(df, cand, mode=cfg.xs_mode, by_sector=bool(cfg.by_sector))
    if cfg.sector_neutral:
        df = sector_neutralize(df, cand)

    # --- split (chronological)
    dates_sorted = np.sort(df["date"].unique())
    split_idx = int((1.0 - cfg.val_frac) * len(dates_sorted))
    split_idx = max(1, min(split_idx, len(dates_sorted)-1))
    split_date = dates_sorted[split_idx-1]
    df_train = df[df["date"] <= split_date]
    df_val   = df[df["date"] >  split_date]

    # --- pruning steps on TRAIN only
    keep = prune_zero_variance(df_train, cand)
    keep = prune_missingness(df_train, keep, max_na_frac=cfg.max_na_frac)
    # IC-based scoring
    rep = per_feature_ic(df_train, keep, cfg.target_col, ic_metric=cfg.ic_metric)
    # apply IC floor and top-k
    rep = rep[rep["mean_ic"] >= cfg.ic_min].sort_values(["mean_ic","icir"], ascending=[False, False])
    if cfg.keep_top_k > 0 and len(rep) > cfg.keep_top_k:
        rep = rep.head(cfg.keep_top_k)
    keep = rep["feature"].tolist()

    # correlation prune then VIF prune on TRAIN subset restricted to 'keep'
    keep = corr_prune(df_train, keep, corr_max=cfg.corr_max)
    if len(keep) > 2:
        keep = vif_prune(df_train, keep, vif_max=cfg.vif_max)

    # --- assemble outputs
    selected = keep
    feature_report = rep.set_index("feature").loc[selected].reset_index()
    out_cols = META_COLS + selected + [cfg.target_col]
    out_cols = [c for c in out_cols if c in df.columns]
    df_out = df[out_cols].copy()

    stats_summary = {
        "n_rows_total": int(len(df)),
        "n_rows_train": int(len(df_train)),
        "n_rows_val": int(len(df_val)),
        "n_features_raw": int(len(cand)),
        "n_features_after_filters": int(len(rep)),
        "n_features_selected": int(len(selected)),
        "split_date": pd.Timestamp(split_date).isoformat(),
        "cfg": asdict(cfg),
    }

    return {
        "df_out": df_out,
        "report": feature_report,
        "selected": selected,
        "summary": stats_summary,
    }

# -------------------------
# CLI
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Input Parquet/CSV path")
    ap.add_argument("--out_parquet", type=str, required=True, help="Output Parquet with selected features")
    ap.add_argument("--report_csv", type=str, required=True, help="Feature quality report CSV")
    ap.add_argument("--summary_json", type=str, required=True, help="JSON summary path")
    ap.add_argument("--selected_txt", type=str, required=True, help="TXT with selected feature names")
    ap.add_argument("--xs_mode", type=str, default="z", choices=["z","rank","none"], help="Cross-sectional transform per date")
    ap.add_argument("--by_sector", type=int, default=0, help="Apply xs transform within sector buckets (if sector col present)")
    ap.add_argument("--sector_neutral", type=int, default=0, help="Regress out sector dummies per date")
    ap.add_argument("--max_na_frac", type=float, default=0.2)
    ap.add_argument("--corr_max", type=float, default=0.95)
    ap.add_argument("--vif_max", type=float, default=20.0)
    ap.add_argument("--keep_top_k", type=int, default=256)
    ap.add_argument("--ic_min", type=float, default=-1.0)
    ap.add_argument("--ic_metric", type=str, default="spearman", choices=["spearman","pearson"])
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--target_col", type=str, default=TARGET_COL_DEFAULT)
    args = ap.parse_args()

    # Load
    if args.data.endswith(".parquet"):
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data)
    if "date" not in df.columns or "symbol" not in df.columns:
        raise ValueError("Input must include 'date' and 'symbol' columns")
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values(["symbol","date"])

    cfg = SuiteConfig(
        xs_mode=args.xs_mode,
        by_sector=args.by_sector,
        sector_neutral=args.sector_neutral,
        max_na_frac=args.max_na_frac,
        corr_max=args.corr_max,
        vif_max=args.vif_max,
        keep_top_k=args.keep_top_k,
        ic_min=args.ic_min,
        ic_metric=args.ic_metric,
        val_frac=args.val_frac,
        target_col=args.target_col,
    )

    result = run_suite(df, cfg)

    # Save outputs
    out_parquet = args.out_parquet
    report_csv = args.report_csv
    summary_json = args.summary_json
    selected_txt = args.selected_txt

    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    os.makedirs(os.path.dirname(report_csv), exist_ok=True)
    os.makedirs(os.path.dirname(summary_json), exist_ok=True)
    os.makedirs(os.path.dirname(selected_txt), exist_ok=True)

    result["df_out"].to_parquet(out_parquet, index=False)
    result["report"].to_csv(report_csv, index=False)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(result["summary"], f, indent=2)
    with open(selected_txt, "w", encoding="utf-8") as f:
        for c in result["selected"]:
            f.write(c + "\n")

    print(f"[ok] Wrote pruned dataset: {out_parquet}")
    print(f"[ok] Wrote feature report: {report_csv}")
    print(f"[ok] Wrote summary JSON:   {summary_json}")
    print(f"[ok] Wrote selected list:  {selected_txt}")
    print(f"Selected features: {len(result['selected'])}")

if __name__ == "__main__":
    main()
