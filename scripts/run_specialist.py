"""Specialist mean-reversion strategy across the FULL universe (v0.6).

Only trades high-edge RSI mean-reversion setups, refined by a meta classifier
trained pooled on candidate bars only, validated by global-time walk-forward.
Backtests per symbol with full risk management and reports honestly.

Run:
    python -m scripts.run_specialist --timeframe 1d
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fath.backtest import risk as R
from fath.backtest.engine import CostModel, run_backtest
from fath.data.universe import DEFAULT_UNIVERSE, load_universe
from fath.eval import metrics
from fath.features import indicators as ta
from fath.models.meta_labeling import MetaModel, bet_size_from_proba
from fath.models.pooled import build_pooled_dataset, time_split
from fath.models.specialist import mean_reversion_setups
from fath.utils.logging import get_logger

log = get_logger("specialist")

NON_FEATURE = {"label", "t1", "weight", "symbol", "ts", "oos_sig", "oos_mp",
               "setup", "up"}


def run(timeframe="1d", source="okx", rsi_low=30.0, rsi_high=70.0,
        allow_short=False, meta_threshold=0.65, min_setups=150,
        out_dir="artifacts_specialist"):
    data = load_universe(DEFAULT_UNIVERSE, timeframe, source)
    sent_map = {}
    try:
        from fath.data.sentiment import fetch_fear_greed, merge_sentiment
        fng = fetch_fear_greed(0)
        sent_map = {s: merge_sentiment(d, fng) for s, d in data.items()}
    except Exception as exc:  # noqa: BLE001
        log.warning("Sentiment off (%s)", exc)

    pooled = build_pooled_dataset(data, sent_map)
    pooled["setup"] = mean_reversion_setups(
        pooled.rename_axis(None), rsi_low, rsi_high).values \
        if "rsi_14" in pooled else 0
    # compute setup directly from rsi column (pooled is row-indexed)
    pooled["setup"] = 0
    pooled.loc[pooled["rsi_14"] < rsi_low, "setup"] = 1
    if allow_short:
        pooled.loc[pooled["rsi_14"] > rsi_high, "setup"] = -1

    # "up" target = setup direction was correct (precision target)
    pooled["up"] = ((np.sign(pooled["label"]) == np.sign(pooled["setup"]))
                    & (pooled["setup"] != 0)).astype(int)

    feat_cols = [c for c in pooled.columns if c not in NON_FEATURE]
    pooled["oos_mp"] = np.nan

    accs = []
    for fold, (tr, te) in enumerate(time_split(pooled, 4, 0.30)):
        cand_tr = tr & (pooled["setup"].to_numpy() != 0)
        cand_te = te & (pooled["setup"].to_numpy() != 0)
        if cand_tr.sum() < min_setups or cand_te.sum() < 30:
            continue
        Xtr = pooled.loc[cand_tr, feat_cols]
        ytr = pooled.loc[cand_tr, "up"].to_numpy()
        meta = MetaModel().fit(Xtr, ytr, sample_weight=pooled.loc[cand_tr, "weight"].to_numpy())
        Xte = pooled.loc[cand_te, feat_cols]
        p = meta.predict_bet_proba(Xte)
        pooled.loc[cand_te, "oos_mp"] = p
        pred = (p >= 0.5).astype(int)
        accs.append((pred == pooled.loc[cand_te, "up"].to_numpy()).mean())
        log.info("Fold %d: cand_train=%d cand_test=%d precision_acc=%.4f",
                 fold, cand_tr.sum(), cand_te.sum(), accs[-1])

    # ---- per-symbol backtest ----------------------------------------------
    rows = []
    for sym, ohlcv in data.items():
        sub = pooled[(pooled["symbol"] == sym) & pooled["oos_mp"].notna()].copy()
        if len(sub) < 20:
            continue
        idx = pd.to_datetime(sub["ts"], utc=True)
        direction = sub["setup"].to_numpy()
        size = direction * bet_size_from_proba(sub["oos_mp"].to_numpy(),
                                               p_threshold=meta_threshold)
        pos = pd.Series(size, index=idx)

        first = idx.min()
        oos = ohlcv.loc[first:]
        # hold each mean-reversion trade for a fixed window then exit (the edge
        # is short-horizon); forward-fill for up to `hold` bars
        pos = pos.reindex(oos.index)
        pos = _hold_n(pos, n=10)

        atr = ta.atr(oos["high"], oos["low"], oos["close"], 14)
        pos = R.apply_trailing_stop(pos, oos["close"], atr, k=3.0)
        res = run_backtest(oos, pos, CostModel())
        s = metrics.summary(res, oos)
        rows.append({"symbol": sym, "ret%": round(s["total_return_pct"], 1),
                     "bh%": round(s["buyhold_return_pct"], 1),
                     "sharpe": round(s["sharpe"], 2),
                     "bh_sharpe": round(s["buyhold_sharpe"], 2),
                     "maxDD%": round(s["max_drawdown_pct"], 1),
                     "trades": s["num_fills"]})

    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_json(f"{out_dir}/specialist_{timeframe}.json", orient="records", indent=2)
    pd.set_option("display.width", 200)
    print(f"\n===== SPECIALIST MEAN-REVERSION @ {timeframe} (OOS, after costs) =====\n")
    print(df.to_string(index=False))
    print(f"\nMean setup-precision accuracy . {np.mean(accs):.4f}" if accs else "no folds")
    print(f"Median strategy Sharpe ........ {df['sharpe'].median():.2f}")
    print(f"Median B&H Sharpe ............. {df['bh_sharpe'].median():.2f}")
    print(f"% beating B&H (Sharpe) ....... {(df['sharpe']>df['bh_sharpe']).mean()*100:.0f}%")
    print(f"% positive return ............ {(df['ret%']>0).mean()*100:.0f}%")
    print(f"Mean strategy return ......... {df['ret%'].mean():.1f}%")
    return df


def _hold_n(pos: pd.Series, n: int = 10) -> pd.Series:
    """Enter on a non-zero signal, hold for n bars, then flat (unless re-signal)."""
    arr = pos.to_numpy()
    out = np.zeros(len(arr))
    hold = 0
    cur = 0.0
    for i, v in enumerate(arr):
        if not np.isnan(v) and v != 0:
            cur = v
            hold = n
        if hold > 0:
            out[i] = cur
            hold -= 1
        else:
            cur = 0.0
    return pd.Series(out, index=pos.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--source", default="okx")
    ap.add_argument("--rsi_low", type=float, default=30.0)
    ap.add_argument("--rsi_high", type=float, default=70.0)
    ap.add_argument("--allow_short", action="store_true")
    ap.add_argument("--meta_threshold", type=float, default=0.50)
    args = ap.parse_args()
    run(args.timeframe, args.source, args.rsi_low, args.rsi_high,
        args.allow_short, args.meta_threshold)


if __name__ == "__main__":
    main()
