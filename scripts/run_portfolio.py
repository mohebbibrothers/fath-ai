"""Full-market portfolio engine (v0.9).

Tests the strategy on EVERY chart (symbol × timeframe), ranks them, allocates
more capital to the best performers (leak-free), and reports the combined
portfolio vs an equal-weight benchmark.

This realizes your request: "test on all charts/timeframes, see which earn well,
and weight trades toward those charts."

Run:
    python -m scripts.run_portfolio --leverage 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fath.backtest.futures import FuturesCostModel, run_futures_backtest
from fath.backtest import portfolio as P
from fath.backtest import risk as R
from fath.data.universe import DEFAULT_UNIVERSE, load_universe
from fath.eval import metrics
from fath.features import indicators as ta
from fath.models.meta_labeling import MetaModel, bet_size_from_proba
from fath.models.pooled import build_pooled_dataset, time_split
from fath.utils.logging import get_logger

log = get_logger("portfolio")

NON_FEATURE = {"label", "t1", "weight", "symbol", "ts", "setup", "up", "oos_mp"}
TIMEFRAMES = ["1d", "4h"]


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


def _chart_returns_for_tf(timeframe, source, leverage, meta_threshold,
                          rsi_low=30.0):
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

    cost = FuturesCostModel(funding_interval_bars={"1h": 8, "4h": 2, "1d": 1}[timeframe])
    chart_rets = {}
    chart_stats = {}
    for sym, ohlcv in data.items():
        sub = pooled[(pooled["symbol"] == sym) & pooled["oos_mp"].notna()].copy()
        if len(sub) < 20:
            continue
        idx = pd.to_datetime(sub["ts"], utc=True)
        size = sub["setup"].to_numpy() * bet_size_from_proba(
            sub["oos_mp"].to_numpy(), p_threshold=meta_threshold)
        oos = ohlcv.loc[idx.min():]
        pos = pd.Series(size, index=idx).reindex(oos.index)
        pos = _hold_n(pos, 10)
        atr = ta.atr(oos["high"], oos["low"], oos["close"], 14)
        pos = R.apply_trailing_stop(pos, oos["close"], atr, k=3.0)
        res = run_futures_backtest(oos, pos, leverage=leverage, cost=cost)
        name = f"{sym}@{timeframe}"
        chart_rets[name] = res.returns
        chart_stats[name] = {
            "ret%": round(res.meta["final_equity"] / res.meta["init_equity"] * 100 - 100, 1),
            "sharpe": round(metrics.sharpe(res.returns), 2),
            "maxDD%": round(metrics.max_drawdown(res.equity) * 100, 1),
            "liq": res.n_liquidations,
        }
    return chart_rets, chart_stats


def run(source="okx_swap", leverage=3.0, meta_threshold=0.65,
        out_dir="artifacts_portfolio"):
    all_rets, all_stats = {}, {}
    for tf in TIMEFRAMES:
        r, s = _chart_returns_for_tf(tf, source, leverage, meta_threshold)
        all_rets.update(r); all_stats.update(s)

    # rank charts
    rank = pd.DataFrame(all_stats).T.sort_values("sharpe", ascending=False)
    pd.set_option("display.width", 200, "display.max_rows", 100)
    print(f"\n===== ALL CHARTS RANKED (futures L={leverage}x, OOS, after costs) =====\n")
    print(rank.to_string())

    # leak-free smart allocation
    alloc = P.allocate_and_evaluate(all_rets, learn_frac=0.5, top_k=12, max_weight=0.25)

    print("\n===== SMART CAPITAL ALLOCATION (leak-free: weights from past only) =====")
    if alloc:
        print(f"\nCharts funded: {alloc['n_charts_funded']}")
        print("Top weights:")
        for c, w in sorted(alloc["weights"].items(), key=lambda kv: -kv[1])[:12]:
            print(f"  {c:18s}: {w*100:5.1f}%")
        print("\nEvaluation window (held out) performance:")
        sm, eq = alloc["smart"], alloc["equal_weight"]
        print(f"  {'':14s} {'return%':>10s} {'sharpe':>8s} {'maxDD%':>8s}")
        print(f"  {'Smart-weighted':14s} {sm['total_return_pct']:>10.1f} "
              f"{sm['sharpe']:>8.2f} {sm['max_drawdown_pct']:>8.1f}")
        print(f"  {'Equal-weight':14s} {eq['total_return_pct']:>10.1f} "
              f"{eq['sharpe']:>8.2f} {eq['max_drawdown_pct']:>8.1f}")
        verdict = ("Smart allocation BEATS equal weight" if sm["sharpe"] > eq["sharpe"]
                   else "Smart allocation does NOT beat equal weight (keep equal)")
        print(f"\n  VERDICT: {verdict}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rank.to_json(f"{out_dir}/chart_ranking.json", orient="index", indent=2)
    (Path(out_dir) / "allocation.json").write_text(json.dumps(alloc, indent=2, default=str))
    return rank, alloc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="okx_swap")
    ap.add_argument("--leverage", type=float, default=3.0)
    ap.add_argument("--meta_threshold", type=float, default=0.65)
    args = ap.parse_args()
    run(args.source, args.leverage, args.meta_threshold)


if __name__ == "__main__":
    main()
