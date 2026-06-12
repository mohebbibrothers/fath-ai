"""Leveraged futures backtest of the specialist strategy + leverage sweep.

Finds the leverage that maximizes risk-adjusted return WITHOUT blowing up via
liquidation. Runs the proven specialist mean-reversion signals through the
realistic futures engine (liquidation + funding + fees) across the universe.

Run:
    python -m scripts.run_futures --timeframe 1d
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fath.backtest.futures import FuturesCostModel, run_futures_backtest
from fath.backtest import risk as R
from fath.data.universe import DEFAULT_UNIVERSE, load_universe
from fath.eval import metrics
from fath.features import indicators as ta
from fath.models.meta_labeling import MetaModel, bet_size_from_proba
from fath.models.pooled import build_pooled_dataset, time_split
from fath.utils.logging import get_logger

log = get_logger("futures")

NON_FEATURE = {"label", "t1", "weight", "symbol", "ts", "setup", "up", "oos_mp"}


def _signals(timeframe, source, rsi_low, allow_short, rsi_high):
    data = load_universe(DEFAULT_UNIVERSE, timeframe, source)
    sent_map = {}
    try:
        from fath.data.sentiment import fetch_fear_greed, merge_sentiment
        fng = fetch_fear_greed(0)
        sent_map = {s: merge_sentiment(d, fng) for s, d in data.items()}
    except Exception:  # noqa: BLE001
        pass
    pooled = build_pooled_dataset(data, sent_map)
    pooled["setup"] = 0
    pooled.loc[pooled["rsi_14"] < rsi_low, "setup"] = 1
    if allow_short:
        pooled.loc[pooled["rsi_14"] > rsi_high, "setup"] = -1
    pooled["up"] = ((np.sign(pooled["label"]) == np.sign(pooled["setup"]))
                    & (pooled["setup"] != 0)).astype(int)
    fc = [c for c in pooled.columns if c not in NON_FEATURE]
    pooled["oos_mp"] = np.nan
    for tr, te in time_split(pooled, 4, 0.30):
        ctr = tr & (pooled["setup"].to_numpy() != 0)
        cte = te & (pooled["setup"].to_numpy() != 0)
        if ctr.sum() < 150 or cte.sum() < 30:
            continue
        m = MetaModel().fit(pooled.loc[ctr, fc], pooled.loc[ctr, "up"].to_numpy(),
                            sample_weight=pooled.loc[ctr, "weight"].to_numpy())
        pooled.loc[cte, "oos_mp"] = m.predict_bet_proba(pooled.loc[cte, fc])
    return data, pooled


def _hold_n(pos, n=10):
    arr = pos.to_numpy(); out = np.zeros(len(arr)); hold = 0; cur = 0.0
    for i, v in enumerate(arr):
        if not np.isnan(v) and v != 0:
            cur = v; hold = n
        if hold > 0:
            out[i] = cur; hold -= 1
        else:
            cur = 0.0
    return pd.Series(out, index=pos.index)


def build_positions(data, pooled, meta_threshold):
    """Return {symbol: (ohlcv, position_series)} for backtesting."""
    out = {}
    for sym, ohlcv in data.items():
        sub = pooled[(pooled["symbol"] == sym) & pooled["oos_mp"].notna()].copy()
        if len(sub) < 20:
            continue
        idx = pd.to_datetime(sub["ts"], utc=True)
        size = sub["setup"].to_numpy() * bet_size_from_proba(
            sub["oos_mp"].to_numpy(), p_threshold=meta_threshold)
        pos = pd.Series(size, index=idx).reindex(ohlcv.loc[idx.min():].index)
        pos = _hold_n(pos, n=10)
        oos = ohlcv.loc[idx.min():]
        atr = ta.atr(oos["high"], oos["low"], oos["close"], 14)
        pos = R.apply_trailing_stop(pos, oos["close"], atr, k=3.0)
        out[sym] = (oos, pos)
    return out


def run(timeframe="1d", source="okx_swap", rsi_low=30.0, rsi_high=70.0,
        allow_short=False, meta_threshold=0.65,
        leverages=(1, 2, 3, 5, 8, 10), out_dir="artifacts_futures"):
    data, pooled = _signals(timeframe, source, rsi_low, allow_short, rsi_high)
    positions = build_positions(data, pooled, meta_threshold)
    cost = FuturesCostModel(funding_interval_bars=_fund_bars(timeframe))

    # ---- leverage sweep (aggregate across universe) -----------------------
    print(f"\n===== LEVERAGE SWEEP @ {timeframe} (futures, OOS, after costs/funding) =====\n")
    print(f"{'L':>4s} {'mean_ret%':>10s} {'median_ret%':>12s} {'%pos':>6s} "
          f"{'liq_rate%':>10s} {'mean_sharpe':>12s}")
    sweep = []
    for L in leverages:
        rets, sharpes, liqs = [], [], 0
        n_sym = 0
        for sym, (oos, pos) in positions.items():
            res = run_futures_backtest(oos, pos, leverage=float(L), cost=cost)
            r = res.meta["final_equity"] / res.meta["init_equity"] - 1
            rets.append(r * 100)
            sharpes.append(metrics.sharpe(res.returns))
            liqs += res.n_liquidations
            n_sym += 1
        rets = np.array(rets)
        row = {"leverage": L, "mean_ret%": round(rets.mean(), 1),
               "median_ret%": round(float(np.median(rets)), 1),
               "pct_pos": round((rets > 0).mean() * 100, 0),
               "total_liquidations": liqs,
               "mean_sharpe": round(float(np.mean(sharpes)), 2)}
        sweep.append(row)
        print(f"{L:>4d} {rets.mean():>10.1f} {np.median(rets):>12.1f} "
              f"{(rets>0).mean()*100:>6.0f} {liqs:>10d} {np.mean(sharpes):>12.2f}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / f"leverage_sweep_{timeframe}.json").write_text(
        json.dumps(sweep, indent=2))
    # Best leverage = maximize mean return subject to a liquidation-rate cap.
    # Sharpe alone is misleading when returns are negative, and ignoring
    # liquidations rewards reckless leverage. We require liquidations to stay
    # low (the strategy must survive the real path).
    n_sym = len(positions)
    safe = [r for r in sweep if r["total_liquidations"] <= max(1, 0.10 * n_sym)]
    pool = safe if safe else sweep
    best = max(pool, key=lambda r: r["mean_ret%"])
    print(f"\nBest SAFE leverage (liq-capped): {best['leverage']}x  "
          f"mean ret {best['mean_ret%']}%  median {best['median_ret%']}%  "
          f"%pos {best['pct_pos']:.0f}  liquidations {best['total_liquidations']}")
    return sweep, positions, best


def _fund_bars(tf):
    # funding every ~8h; convert to bars
    return {"1h": 8, "4h": 2, "1d": 1}.get(tf, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--source", default="okx_swap")
    ap.add_argument("--allow_short", action="store_true")
    ap.add_argument("--meta_threshold", type=float, default=0.65)
    args = ap.parse_args()
    run(args.timeframe, args.source, allow_short=args.allow_short,
        meta_threshold=args.meta_threshold)


if __name__ == "__main__":
    main()
