"""Advanced pipeline: primary direction model + meta-labeling + bet sizing.

This is the v0.2 research engine. Compared to run_pipeline.py it adds:
  * A META model that decides whether to ACT on each primary signal, trained
    walk-forward with its own purge/embargo (no leakage).
  * Probability-proportional BET SIZING (trade big only when confident).
  * Hold-until-flip position logic to control turnover/cost drag.
  * Optional minimum holding period to further suppress churn.

Run:
    python -m scripts.run_meta_pipeline --symbol BTC/USDT --timeframe 1d
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fath.backtest.engine import CostModel, run_backtest
from fath.data import store
from fath.eval import metrics
from fath.features.build import build_features
from fath.labels.triple_barrier import (
    average_uniqueness_weights,
    triple_barrier_labels,
)
from fath.models.classifier import BarrierClassifier
from fath.models.cv import purge_embargo, walk_forward_splits
from fath.models.meta_labeling import MetaModel, bet_size_from_proba, make_meta_target
from fath.utils.logging import get_logger

log = get_logger("meta")


def build_dataset(symbol, timeframe, source, tp, sl, horizon, vol_window,
                  use_sentiment=True):
    ohlcv = store.load(symbol, timeframe, source)
    sent = None
    if use_sentiment:
        try:
            from fath.data.sentiment import fetch_fear_greed, merge_sentiment
            sent = merge_sentiment(ohlcv, fetch_fear_greed(0))
        except Exception as exc:  # noqa: BLE001
            log.warning("Sentiment unavailable (%s); continuing without", exc)
    feats = build_features(ohlcv, sentiment=sent)
    logret = np.log(ohlcv["close"]).diff()
    vol = logret.rolling(vol_window).std(ddof=0)
    labels = triple_barrier_labels(ohlcv["close"], vol, tp, sl, horizon, 0.0)
    common = feats.index.intersection(labels.index)
    return ohlcv, feats.loc[common], labels.loc[common], common


def run(symbol="BTC/USDT", timeframe="1d", source="okx",
        tp=1.5, sl=1.5, horizon=10, vol_window=20,
        conf_edge=0.03, meta_threshold=0.55, min_hold=2,
        out_dir="artifacts_meta"):
    ohlcv, X, labels, common = build_dataset(
        symbol, timeframe, source, tp, sl, horizon, vol_window)
    y = labels["label"]
    t1 = labels["t1"]
    weights = average_uniqueness_weights(t1, common)
    pos_of = {ts: i for i, ts in enumerate(common)}
    t1_pos = np.array([pos_of.get(t, -1) for t in t1.values])
    n = len(X)

    primary_signal = pd.Series(0, index=common, dtype=float)
    meta_proba = pd.Series(np.nan, index=common, dtype=float)
    accs = []

    for tr, te in walk_forward_splits(n, 5, 0.30, True):
        trc = purge_embargo(tr, te, t1_pos, 0.01, n_samples=n)
        if len(trc) < 250:
            continue

        # --- PRIMARY: direction ------------------------------------------
        prim = BarrierClassifier().fit(
            X.iloc[trc], y.iloc[trc], sample_weight=weights.iloc[trc].to_numpy())
        proba_tr = prim.predict_proba(X.iloc[trc])
        proba_te = prim.predict_proba(X.iloc[te])

        edge_tr = proba_tr[:, 2] - proba_tr[:, 0]
        edge_te = proba_te[:, 2] - proba_te[:, 0]
        sig_tr = np.where(edge_tr >= conf_edge, 1,
                          np.where(edge_tr <= -conf_edge, -1, 0))
        sig_te = np.where(edge_te >= conf_edge, 1,
                          np.where(edge_te <= -conf_edge, -1, 0))

        acc = (prim.predict_signed(X.iloc[te]) == y.iloc[te].to_numpy()).mean()
        accs.append(acc)

        # --- META: should we act? ----------------------------------------
        sig_tr_s = pd.Series(sig_tr, index=X.iloc[trc].index)
        y_tr = y.iloc[trc]
        meta_mask = sig_tr_s != 0
        if meta_mask.sum() < 100:
            # not enough to train meta -> act on all primary signals
            for ix, s in zip(X.iloc[te].index, sig_te):
                primary_signal.loc[ix] = s
                meta_proba.loc[ix] = 1.0
            continue

        meta_y = make_meta_target(sig_tr_s, y_tr)
        # meta features: primary features + the signal direction + |edge|
        Xtr_meta = X.iloc[trc].loc[meta_mask].copy()
        Xtr_meta["_sig"] = sig_tr_s[meta_mask].values
        Xtr_meta["_edge_abs"] = np.abs(edge_tr[meta_mask.values])
        meta = MetaModel().fit(Xtr_meta, meta_y.values)

        te_mask = sig_te != 0
        Xte_meta = X.iloc[te].loc[te_mask].copy()
        if len(Xte_meta):
            Xte_meta["_sig"] = sig_te[te_mask]
            Xte_meta["_edge_abs"] = np.abs(edge_te[te_mask])
            pbet = meta.predict_bet_proba(Xte_meta)
        else:
            pbet = np.array([])

        te_index = X.iloc[te].index
        for ix, s in zip(te_index, sig_te):
            primary_signal.loc[ix] = s
        for ix, pb in zip(Xte_meta.index, pbet):
            meta_proba.loc[ix] = pb

    # --- combine into a sized position -----------------------------------
    valid = meta_proba.notna() & (primary_signal != 0)
    sizes = pd.Series(0.0, index=common)
    pb = meta_proba[valid].to_numpy()
    sz = bet_size_from_proba(pb, p_threshold=meta_threshold, max_size=1.0)
    sizes.loc[valid] = primary_signal[valid].to_numpy() * sz

    # hold-until-flip + minimum holding period to control turnover
    pos = _apply_holding(sizes, min_hold)

    # OOS span (where we actually have predictions)
    pred_idx = sizes[sizes.index.isin(common)].index
    first_pred = primary_signal[primary_signal != 0].index.min()
    oos_ohlcv = ohlcv.loc[first_pred:]
    pos = pos.reindex(oos_ohlcv.index).fillna(0.0)

    res = run_backtest(oos_ohlcv, pos, CostModel())
    s = metrics.summary(res, oos_ohlcv)
    s["mean_oos_accuracy"] = float(np.mean(accs)) if accs else None
    s["avg_position"] = float(pos.abs().mean())
    s["pct_time_in_market"] = float((pos.abs() > 1e-6).mean() * 100)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out = {"symbol": symbol, "timeframe": timeframe, "metrics": s}
    (Path(out_dir) / f"{symbol.replace('/','-')}_{timeframe}.json").write_text(
        json.dumps(out, indent=2, default=str))
    return s


def _apply_holding(sizes: pd.Series, min_hold: int, rebalance_band: float = 0.34) -> pd.Series:
    """Hold position until signal flips sign; enforce a minimum holding period.

    To control turnover we (a) quantize size to discrete steps and (b) only
    rebalance the *size* of an existing position if it moves by more than
    ``rebalance_band`` — tiny size wiggles do not trigger costly fills.
    """
    # quantize to {0, 0.33, 0.66, 1.0} * sign
    q = 0.33
    arr_raw = sizes.to_numpy()
    sign = np.sign(arr_raw)
    mag = np.abs(arr_raw)
    mag = np.round(mag / q) * q
    arr = sign * mag

    out = np.zeros(len(arr))
    held = 0.0
    bars_held = 0
    for i, target in enumerate(arr):
        if held == 0.0:
            if target != 0.0:
                held = target
                bars_held = 0
        else:
            same_dir = (target != 0.0 and np.sign(target) == np.sign(held))
            opp_dir = (target != 0.0 and np.sign(target) != np.sign(held))
            if opp_dir and bars_held >= min_hold:
                held = target
                bars_held = 0
            elif same_dir and abs(target - held) >= rebalance_band:
                held = target  # only rebalance on meaningful size change
            # else: keep holding, no fill
        out[i] = held
        bars_held += 1
    return pd.Series(out, index=sizes.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--source", default="okx")
    ap.add_argument("--tp", type=float, default=1.5)
    ap.add_argument("--sl", type=float, default=1.5)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--vol_window", type=int, default=20)
    ap.add_argument("--conf_edge", type=float, default=0.03)
    ap.add_argument("--meta_threshold", type=float, default=0.55)
    ap.add_argument("--min_hold", type=int, default=2)
    args = ap.parse_args()

    s = run(args.symbol, args.timeframe, args.source, args.tp, args.sl,
            args.horizon, args.vol_window, args.conf_edge,
            args.meta_threshold, args.min_hold)
    print(f"\n=== {args.symbol} {args.timeframe} (meta-labeled, OOS, after costs) ===")
    for k, v in s.items():
        print(f"  {k:24s}: {v}")


if __name__ == "__main__":
    main()
