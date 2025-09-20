# make_yf_parquet.py
# Usage:
#   python make_yf_parquet.py --tickers "AAPL MSFT NVDA AMZN META GOOGL" \
#       --start 2015-01-01 --end 2100-01-01 --out /mnt/data/yf_us_2015_present.parquet
import argparse, sys
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

def build_features(w: pd.DataFrame) -> pd.DataFrame:
    """
    Expects columns: ['date','symbol','open','high','low','close','adj_close','volume']
    Adds simple leak-safe features (only using past info), and target_1d = next-day return at time t.
    """
    w = w.sort_values(['symbol','date']).copy()
    # Basic returns (use adj_close for splits/dividends)
    w['ret_1'] = w.groupby('symbol')['adj_close'].pct_change(1)
    w['logret_1'] = np.log(w.groupby('symbol')['adj_close'].shift(0) / w.groupby('symbol')['adj_close'].shift(1))
    # Rolling stats (min_periods to avoid NaNs early; all shifted by 1 to keep them strictly "past")
    for win in (5, 10, 20):
        grp = w.groupby('symbol')['ret_1']
        w[f'vol_{win}'] = grp.rolling(win, min_periods=3).std().reset_index(level=0, drop=True).shift(1)
        w[f'mom_{win}'] = grp.rolling(win, min_periods=3).sum().reset_index(level=0, drop=True).shift(1)
        w[f'avgvol_{win}'] = w.groupby('symbol')['volume'].rolling(win, min_periods=3).mean().reset_index(level=0, drop=True).shift(1)
    # Price-relative features (shifted)
    w['close_sma_5']  = w.groupby('symbol')['adj_close'].rolling(5, min_periods=3).mean().reset_index(level=0, drop=True).shift(1)
    w['close_sma_20'] = w.groupby('symbol')['adj_close'].rolling(20, min_periods=5).mean().reset_index(level=0, drop=True).shift(1)
    w['close_div_sma20'] = w['adj_close'] / (w['close_sma_20'] + 1e-12)

    # Target: future 1-day return aligned to time t (predicting next day’s return)
    # This matches your trainer: window [t-L..t-1] -> predict target at t (future return over [t..t+1]).
    nxt = w.groupby('symbol')['adj_close'].shift(-1)
    w['target_1d'] = (nxt / w['adj_close']) - 1.0

    # Keep useful cols; drop rows with NaNs in required fields
    feat_cols = [c for c in w.columns if c not in ['open','high','low','close','adj_close','volume']]
    w = w.dropna(subset=['target_1d']).dropna()
    return w

def download_yf(tickers, start, end):
    # yfinance can download all at once; pivot from wide to long
    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,  # we’ll keep both raw close and adj_close
        group_by='ticker',
        threads=True,
        progress=False
    )
    if isinstance(df.columns, pd.MultiIndex):
        # Wide -> long
        out = []
        for tkr in tickers:
            if tkr not in df.columns.levels[0]:
                continue
            sub = df[tkr].reset_index().rename(columns={
                'Date':'date','Open':'open','High':'high','Low':'low','Close':'close','Adj Close':'adj_close','Volume':'volume'
            })
            sub['symbol'] = tkr
            out.append(sub[['date','symbol','open','high','low','close','adj_close','volume']])
        if not out:
            raise RuntimeError("No data downloaded. Check tickers.")
        data = pd.concat(out, ignore_index=True)
    else:
        # Single ticker case
        data = df.reset_index().rename(columns={
            'Date':'date','Open':'open','High':'high','Low':'low','Close':'close','Adj Close':'adj_close','Volume':'volume'
        })
        data['symbol'] = tickers[0]
        data = data[['date','symbol','open','high','low','close','adj_close','volume']]
    data['date'] = pd.to_datetime(data['date']).dt.tz_localize(None)
    data = data.dropna(subset=['adj_close']).sort_values(['symbol','date'])
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tickers', type=str, required=True, help='Space-separated tickers, e.g. "AAPL MSFT NVDA AMZN META GOOGL"')
    ap.add_argument('--start', type=str, default='2015-01-01')
    ap.add_argument('--end', type=str, default='2100-01-01')
    ap.add_argument('--out', type=str, default='/mnt/data/yf_us_2015_present.parquet')
    args = ap.parse_args()

    tickers = args.tickers.strip().replace(',', ' ').split()
    raw = download_yf(tickers, args.start, args.end)
    df = build_features(raw)

    # Save as Parquet (fast + compressed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[ok] Wrote {len(df):,} rows to {out_path}")

if __name__ == '__main__':
    main()
