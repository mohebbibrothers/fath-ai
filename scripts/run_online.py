"""Demonstrate continuous (incremental) learning + drift monitoring.

Replays full history bar-by-bar with periodic retraining, then reports how the
model's rolling out-of-sample accuracy evolves — i.e. proof that it keeps
learning and adapting, measured honestly.

Run:
    python -m scripts.run_online --symbol BTC/USDT --timeframe 1d
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from fath.data import store
from fath.features.build import build_features
from fath.labels.triple_barrier import average_uniqueness_weights, triple_barrier_labels
from fath.models.online import OnlineLearner
from fath.utils.logging import get_logger

log = get_logger("online")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--source", default="okx")
    ap.add_argument("--retrain_every", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=400)
    args = ap.parse_args()

    ohlcv = store.load(args.symbol, args.timeframe, args.source)
    try:
        from fath.data.sentiment import fetch_fear_greed, merge_sentiment
        sent = merge_sentiment(ohlcv, fetch_fear_greed(0))
    except Exception:  # noqa: BLE001
        sent = None
    feats = build_features(ohlcv, sentiment=sent)
    lr = np.log(ohlcv["close"]).diff()
    vol = lr.rolling(20).std(ddof=0)
    labels = triple_barrier_labels(ohlcv["close"], vol, 1.5, 1.5, 10, 0.0)
    common = feats.index.intersection(labels.index)
    X, y = feats.loc[common], labels.loc[common, "label"]
    w = average_uniqueness_weights(labels.loc[common, "t1"], common)

    learner = OnlineLearner(warmup=args.warmup, retrain_every=args.retrain_every)
    oos, acc_curve = learner.replay(X, y, sample_weight=w)

    print(f"\n=== Continuous learning: {args.symbol} {args.timeframe} ===")
    print(f"Bars replayed ............ {len(X)}")
    print(f"Retrain cadence .......... every {args.retrain_every} bars")
    print(f"Refits performed ......... ~{len(X)//args.retrain_every}")
    if len(acc_curve):
        print(f"Rolling OOS acc (first) .. {acc_curve[0]:.4f}")
        print(f"Rolling OOS acc (last) ... {acc_curve[-1]:.4f}")
        print(f"Rolling OOS acc (mean) ... {acc_curve.mean():.4f}")
    print("Drift status:", learner.drift_status())


if __name__ == "__main__":
    main()
