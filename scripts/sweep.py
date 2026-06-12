"""Multi-asset / multi-timeframe out-of-sample sweep.

Why this matters: a strategy that "works" on one symbol/timeframe is usually
luck (overfitting to one path). A *real* edge shows up consistently across many
independent markets. This script runs the identical, leak-free pipeline across
every (symbol, timeframe) we have cached and prints an honest leaderboard plus
an AGGREGATE verdict.

Usage:
    python -m scripts.sweep
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fath.backtest.engine import CostModel, run_backtest
from fath.backtest.policy import probs_to_position
from fath.data import store
from fath.eval import metrics
from fath.features.build import build_features
from fath.labels.triple_barrier import (
    average_uniqueness_weights,
    triple_barrier_labels,
)
from fath.models.classifier import BarrierClassifier
from fath.models.cv import purge_embargo, walk_forward_splits
from fath.utils.logging import get_logger

log = get_logger("sweep")

# (symbol, timeframe, vol_window, max_horizon)
DEFAULT_GRID = [
    ("BTC/USDT", "1h"), ("ETH/USDT", "1h"), ("SOL/USDT", "1h"),
    ("BNB/USDT", "1h"), ("XRP/USDT", "1h"),
    ("BTC/USDT", "4h"), ("ETH/USDT", "4h"), ("SOL/USDT", "4h"),
    ("BNB/USDT", "4h"), ("XRP/USDT", "4h"),
    ("BTC/USDT", "1d"), ("ETH/USDT", "1d"), ("SOL/USDT", "1d"),
]


def run_one(symbol: str, timeframe: str, source: str = "okx") -> dict | None:
    try:
        ohlcv = store.load(symbol, timeframe, source)
    except FileNotFoundError:
        log.warning("No cache for %s %s; skip", symbol, timeframe)
        return None
    if len(ohlcv) < 1500:
        log.warning("%s %s too short (%d); skip", symbol, timeframe, len(ohlcv))
        return None

    feats = build_features(ohlcv)
    logret = np.log(ohlcv["close"]).diff()
    vol = logret.rolling(24).std(ddof=0)
    labels = triple_barrier_labels(ohlcv["close"], vol, 1.5, 1.5, 24, 0.0)

    common = feats.index.intersection(labels.index)
    X = feats.loc[common]
    y = labels.loc[common, "label"]
    t1 = labels.loc[common, "t1"]
    weights = average_uniqueness_weights(t1, common)
    pos_of = {ts: i for i, ts in enumerate(common)}
    t1_pos = np.array([pos_of.get(t, -1) for t in t1.values])

    n = len(X)
    oos = np.full((n, 3), np.nan)
    accs = []
    for tr, te in walk_forward_splits(n, 5, 0.30, True):
        trc = purge_embargo(tr, te, t1_pos, 0.01, n_samples=n)
        if len(trc) < 200:
            continue
        clf = BarrierClassifier().fit(X.iloc[trc], y.iloc[trc],
                                      sample_weight=weights.iloc[trc].to_numpy())
        oos[te] = clf.predict_proba(X.iloc[te])
        accs.append((clf.predict_signed(X.iloc[te]) == y.iloc[te].to_numpy()).mean())

    valid = ~np.isnan(oos).any(axis=1)
    idxv = X.index[valid]
    pos = probs_to_position(oos[valid], idxv, conf_threshold=0.50, edge_threshold=0.04)
    pos = pos.replace(0.0, np.nan).ffill().fillna(0.0)  # low-turnover hold

    oos_ohlcv = ohlcv.loc[idxv.min():]
    res = run_backtest(oos_ohlcv, pos, CostModel())
    s = metrics.summary(res, oos_ohlcv)
    return {
        "symbol": symbol, "timeframe": timeframe,
        "n": int(n), "oos_acc": float(np.mean(accs)) if accs else None,
        "strat_ret%": round(s["total_return_pct"], 1),
        "bh_ret%": round(s["buyhold_return_pct"], 1),
        "strat_sharpe": round(s["sharpe"], 2),
        "bh_sharpe": round(s["buyhold_sharpe"], 2),
        "maxdd%": round(s["max_drawdown_pct"], 1),
        "fills": s["num_fills"],
        "cost_drag%": round(s["cost_drag_pct"], 1),
        "edge_vs_bh": round(s["sharpe"] - s["buyhold_sharpe"], 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/sweep.json")
    args = ap.parse_args()

    rows = []
    for sym, tf in DEFAULT_GRID:
        log.info("Running %s %s ...", sym, tf)
        r = run_one(sym, tf)
        if r:
            rows.append(r)

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_json(args.out, orient="records", indent=2)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n================ OUT-OF-SAMPLE SWEEP (after costs) ================\n")
    print(df.to_string(index=False))

    # Honest aggregate verdict
    print("\n================ AGGREGATE VERDICT ================")
    mean_acc = df["oos_acc"].mean()
    median_edge = df["edge_vs_bh"].median()
    beat = (df["strat_sharpe"] > df["bh_sharpe"]).mean() * 100
    print(f"Mean OOS accuracy ............ {mean_acc:.4f}  (chance ~0.50 for 2-class)")
    print(f"Median Sharpe edge vs B&H .... {median_edge:+.2f}")
    print(f"% of markets beating B&H ..... {beat:.0f}%")
    if median_edge > 0.2 and beat > 60:
        print("VERDICT: Persistent positive edge across markets. Worth pursuing.")
    elif median_edge > 0:
        print("VERDICT: Weak/inconsistent edge. Needs improvement before any live use.")
    else:
        print("VERDICT: No reliable edge after costs yet. Do NOT trade live.")


if __name__ == "__main__":
    main()
