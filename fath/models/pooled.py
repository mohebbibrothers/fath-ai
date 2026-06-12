"""Pooled cross-asset training.

KEY IDEA (what "train on the whole market" should mean):
Train ONE model on the pooled samples of MANY assets simultaneously. Benefits:
  * ~20x more training data -> less overfitting, better generalization.
  * The model learns market-wide structural patterns (momentum, mean reversion,
    volatility dynamics) that recur across coins, instead of memorizing one
    coin's path.
  * A symbol id / cross-sectional features let it still specialize a bit.

CRITICAL ANTI-LEAKAGE RULE:
With many assets sharing the same calendar, a naive row split leaks the future
(asset A's test date could equal asset B's train date). We therefore split by
GLOBAL TIME: everything strictly before a cutoff date is train; everything at/
after (plus an embargo) is test. This is the only correct way to validate a
pooled cross-sectional model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fath.features.build import build_features
from fath.labels.triple_barrier import average_uniqueness_weights, triple_barrier_labels
from fath.utils.logging import get_logger

log = get_logger(__name__)


def build_pooled_dataset(data: dict, sentiment_map: dict | None = None,
                         tp=1.5, sl=1.5, horizon=10, vol_window=20):
    """Build a stacked dataset from {symbol: ohlcv}.

    Returns a single DataFrame with columns:
        [features..., 'label', 't1', 'weight', 'symbol', 'ts']
    where 'ts' is the bar timestamp (for time-based splitting) and rows from all
    symbols are concatenated.
    """
    frames = []
    for sym, ohlcv in data.items():
        sent = sentiment_map.get(sym) if sentiment_map else None
        feats = build_features(ohlcv, sentiment=sent)
        lr = np.log(ohlcv["close"]).diff()
        vol = lr.rolling(vol_window).std(ddof=0)
        labels = triple_barrier_labels(ohlcv["close"], vol, tp, sl, horizon, 0.0)
        common = feats.index.intersection(labels.index)
        if len(common) < 200:
            continue
        X = feats.loc[common].copy()
        X["label"] = labels.loc[common, "label"].values
        X["t1"] = labels.loc[common, "t1"].values
        w = average_uniqueness_weights(labels.loc[common, "t1"], common)
        X["weight"] = w.reindex(common).fillna(1.0).values
        X["symbol"] = sym
        X["ts"] = common
        frames.append(X)
        log.info("Pooled +%s: %d rows", sym, len(X))

    pooled = pd.concat(frames, ignore_index=True)
    # add a numeric symbol id as a feature (lets the tree specialize)
    pooled["symbol_id"] = pooled["symbol"].astype("category").cat.codes

    # --- cross-sectional features (the key to pooled learning) --------------
    # For each timestamp, rank selected momentum/vol features ACROSS symbols so
    # the model knows where this asset sits relative to the rest of the market
    # (cross-sectional momentum is one of the most robust effects in markets).
    xs_cols = [c for c in ("ret_5", "ret_13", "ret_21", "rvol_21", "mom_z_21",
                            "rsi_14", "adx_14") if c in pooled.columns]
    for col in xs_cols:
        pooled[f"xs_rank_{col}"] = (
            pooled.groupby("ts")[col].rank(pct=True)
        )
    log.info("Pooled dataset: %d rows, %d symbols, +%d cross-sectional feats",
             len(pooled), pooled["symbol"].nunique(), len(xs_cols))
    return pooled


def time_split(pooled: pd.DataFrame, n_splits: int = 4, test_frac: float = 0.25,
               embargo_days: int = 5):
    """Yield (train_mask, test_mask) by GLOBAL time, with an embargo gap.

    Walk-forward over calendar time: each fold's test window is a contiguous
    date range; train is everything before it minus an embargo buffer.
    """
    ts = pd.to_datetime(pooled["ts"], utc=True)
    tmin, tmax = ts.min(), ts.max()
    total = (tmax - tmin)
    test_span = total * test_frac / n_splits
    first_test_start = tmax - n_splits * test_span
    embargo = pd.Timedelta(days=embargo_days)

    for k in range(n_splits):
        test_start = first_test_start + k * test_span
        test_end = test_start + test_span
        test_mask = (ts >= test_start) & (ts < test_end)
        train_mask = ts < (test_start - embargo)
        # also purge train labels whose horizon reaches into the test window
        t1 = pd.to_datetime(pooled["t1"], utc=True)
        train_mask = train_mask & (t1 < test_start)
        yield train_mask.to_numpy(), test_mask.to_numpy()
