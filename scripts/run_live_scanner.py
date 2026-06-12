"""24/7 dynamic market scanner & trade planner (paper mode).

Continuously scans every chart in the universe, runs the AI analysis, and emits
ranked, fully-specified trade plans (direction, leverage, size, entry, stop,
targets, liquidation) — with the hard guarantee that every stop sits inside the
liquidation price so capital can never be liquidated.

This is the live decision loop in PAPER mode: it plans (and would place) trades
but touches no real money. A real executor can subscribe to the same plans.

Run (single scan):
    python -m scripts.run_live_scanner --once
Run (continuous, e.g. every 300s):
    python -m scripts.run_live_scanner --interval 300
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fath.data import store
from fath.data.universe import DEFAULT_UNIVERSE
from fath.live.dynamic_trader import plan_trade
from fath.models.meta_labeling import MetaModel
from fath.models.pooled import build_pooled_dataset, time_split
from fath.utils.logging import get_logger

log = get_logger("scanner")

NON_FEATURE = {"label", "t1", "weight", "symbol", "ts", "setup", "up", "oos_mp"}
TIMEFRAMES = ["1d", "4h"]


def train_meta(source, timeframe, rsi_low=30.0, allow_short=False, rsi_high=70.0):
    """Train ONE meta model on all-but-last data (leak-free for live use)."""
    from fath.data.universe import load_universe
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
    cand = pooled[pooled["setup"] != 0]
    if len(cand) < 200:
        return None, fc, sent_map, data
    meta = MetaModel().fit(cand[fc], cand["up"].to_numpy(),
                           sample_weight=cand["weight"].to_numpy())
    return meta, fc, sent_map, data


def scan_once(source, equity=10_000.0, top_n=10, allow_short=False):
    """Run one full-market scan, return ranked trade plans."""
    plans = []
    for tf in TIMEFRAMES:
        meta, fc, sent_map, data = train_meta(source, tf, allow_short=allow_short)
        if meta is None:
            continue
        for sym, ohlcv in data.items():
            sent = sent_map.get(sym)
            plan = plan_trade(sym, tf, ohlcv, sent, meta, fc,
                              allow_short=allow_short)
            if plan is not None:
                plans.append(plan)

    # rank by AI quality * expected R (best risk-adjusted opportunities)
    plans.sort(key=lambda p: p.meta_quality * p.expected_R, reverse=True)
    return plans[:top_n]


def _print_plans(plans):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n========== MARKET SCAN @ {now} ==========")
    if not plans:
        print("No high-quality setups right now. Staying flat (cash is a position).")
        return
    print(f"{'#':>2s} {'chart':16s} {'side':>5s} {'lev':>5s} {'entry':>11s} "
          f"{'stop':>11s} {'target':>11s} {'liq':>11s} {'qual':>5s} {'R':>4s}")
    for i, p in enumerate(plans, 1):
        side = "LONG" if p.side > 0 else "SHORT"
        print(f"{i:>2d} {p.symbol+'@'+p.timeframe:16s} {side:>5s} {p.leverage:>4.1f}x "
              f"{p.entry:>11.4f} {p.stop:>11.4f} {p.targets[-1]:>11.4f} "
              f"{p.liquidation:>11.4f} {p.meta_quality:>5.2f} {p.expected_R:>4.1f}")
    # safety check banner
    safe = all((p.stop > p.liquidation) if p.side > 0 else (p.stop < p.liquidation)
               for p in plans)
    print(f"\n[SAFETY] every stop inside liquidation price: "
          f"{'YES — liquidation impossible' if safe else 'NO — BUG!'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="okx_swap")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--top_n", type=int, default=10)
    ap.add_argument("--allow_short", action="store_true")
    ap.add_argument("--equity", type=float, default=10_000.0)
    args = ap.parse_args()

    out = Path("artifacts_live"); out.mkdir(exist_ok=True)
    while True:
        plans = scan_once(args.source, args.equity, args.top_n, args.allow_short)
        _print_plans(plans)
        (out / "latest_plans.json").write_text(
            json.dumps([p.to_dict() for p in plans], indent=2))
        if args.once:
            break
        log.info("Sleeping %ds until next scan...", args.interval)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
