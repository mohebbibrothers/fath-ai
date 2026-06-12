"""v0.4 full strategy: primary + meta-labeling + advanced risk management.

Combines everything and, crucially, applies the risk layer (trailing stop,
vol targeting, regime gate, drawdown brake) to attack the deep-drawdown problem
identified in the sweep. Reports before/after risk so the effect is visible.

Run:
    python -m scripts.run_full --symbol BTC/USDT --timeframe 1d
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fath.backtest.engine import CostModel, run_backtest
from fath.backtest import risk as R
from fath.data import store
from fath.eval import metrics
from fath.features import indicators as ta
from fath.features.build import build_features
from fath.labels.triple_barrier import average_uniqueness_weights, triple_barrier_labels
from fath.models.classifier import BarrierClassifier
from fath.models.cv import purge_embargo, walk_forward_splits
from fath.models.meta_labeling import MetaModel, bet_size_from_proba, make_meta_target
from fath.utils.logging import get_logger

log = get_logger("full")


def _signals(symbol, timeframe, source, conf_edge, meta_threshold):
    ohlcv = store.load(symbol, timeframe, source)
    try:
        from fath.data.sentiment import fetch_fear_greed, merge_sentiment
        sent = merge_sentiment(ohlcv, fetch_fear_greed(0))
    except Exception:  # noqa: BLE001
        sent = None
    X_all = build_features(ohlcv, sentiment=sent)
    lr = np.log(ohlcv["close"]).diff()
    vol = lr.rolling(20).std(ddof=0)
    labels = triple_barrier_labels(ohlcv["close"], vol, 1.5, 1.5, 10, 0.0)
    common = X_all.index.intersection(labels.index)
    X = X_all.loc[common]
    y = labels.loc[common, "label"]
    t1 = labels.loc[common, "t1"]
    w = average_uniqueness_weights(t1, common)
    pos_of = {ts: i for i, ts in enumerate(common)}
    t1_pos = np.array([pos_of.get(t, -1) for t in t1.values])
    n = len(X)

    sig = pd.Series(0.0, index=common)
    mproba = pd.Series(np.nan, index=common)

    for tr, te in walk_forward_splits(n, 5, 0.30, True):
        trc = purge_embargo(tr, te, t1_pos, 0.01, n_samples=n)
        if len(trc) < 250:
            continue
        prim = BarrierClassifier().fit(X.iloc[trc], y.iloc[trc],
                                       sample_weight=w.iloc[trc].to_numpy())
        ptr = prim.predict_proba(X.iloc[trc]); pte = prim.predict_proba(X.iloc[te])
        etr = ptr[:, 2] - ptr[:, 0]; ete = pte[:, 2] - pte[:, 0]
        str_ = np.where(etr >= conf_edge, 1, np.where(etr <= -conf_edge, -1, 0))
        ste = np.where(ete >= conf_edge, 1, np.where(ete <= -conf_edge, -1, 0))

        str_s = pd.Series(str_, index=X.iloc[trc].index)
        mmask = str_s != 0
        if mmask.sum() >= 100:
            my = make_meta_target(str_s, y.iloc[trc])
            Xtr = X.iloc[trc].loc[mmask].copy()
            Xtr["_sig"] = str_s[mmask].values
            Xtr["_edge_abs"] = np.abs(etr[mmask.values])
            meta = MetaModel().fit(Xtr, my.values)
            temask = ste != 0
            Xte = X.iloc[te].loc[temask].copy()
            if len(Xte):
                Xte["_sig"] = ste[temask]; Xte["_edge_abs"] = np.abs(ete[temask])
                pbet = meta.predict_bet_proba(Xte)
                for ix, pb in zip(Xte.index, pbet):
                    mproba.loc[ix] = pb
            for ix, s in zip(X.iloc[te].index, ste):
                sig.loc[ix] = s
        else:
            for ix, s in zip(X.iloc[te].index, ste):
                sig.loc[ix] = s; mproba.loc[ix] = 1.0

    return ohlcv, X, sig, mproba, common


def _positions(sig, mproba, common, meta_threshold):
    valid = mproba.notna() & (sig != 0)
    sizes = pd.Series(0.0, index=common)
    sz = bet_size_from_proba(mproba[valid].to_numpy(), p_threshold=meta_threshold)
    sizes.loc[valid] = sig[valid].to_numpy() * sz
    # quantize + hold-until-flip (reuse meta-pipeline logic, inline)
    q = 0.33
    s = np.sign(sizes.to_numpy()); m = np.round(np.abs(sizes.to_numpy()) / q) * q
    arr = s * m
    out = np.zeros(len(arr)); held = 0.0
    for i, t in enumerate(arr):
        if held == 0.0:
            held = t if t != 0 else 0.0
        elif t != 0 and np.sign(t) != np.sign(held):
            held = t
        elif t != 0 and np.sign(t) == np.sign(held) and abs(t - held) >= 0.34:
            held = t
        out[i] = held
    return pd.Series(out, index=common)


def run(symbol="BTC/USDT", timeframe="1d", source="okx",
        conf_edge=0.03, meta_threshold=0.55, out_dir="artifacts_full"):
    ohlcv, X, sig, mproba, common = _signals(symbol, timeframe, source,
                                             conf_edge, meta_threshold)
    base_pos = _positions(sig, mproba, common, meta_threshold)

    first = sig[sig != 0].index.min()
    oos = ohlcv.loc[first:]
    base_pos = base_pos.reindex(oos.index).fillna(0.0)

    # ---- risk overlays -----------------------------------------------------
    atr = ta.atr(oos["high"], oos["low"], oos["close"], 14)
    adx = ta.adx(oos["high"], oos["low"], oos["close"], 14)
    rets = oos["close"].pct_change()
    vol_pct = rets.abs().rolling(60).rank(pct=True)

    pos_stop = R.apply_trailing_stop(base_pos, oos["close"], atr, k=3.5)
    gate = R.regime_gate(adx, vol_pct)
    vscale = R.vol_target_scale(rets, target_ann_vol=0.60, lookback=30,
                                bars_per_year=_bpy(timeframe))
    # Quantize the continuous risk multiplier so tiny wiggles don't churn the
    # book (turnover control). Round the combined multiplier to coarse steps.
    mult = (gate * vscale).clip(0.0, 1.0)
    mult = (mult / 0.25).round() * 0.25
    risk_pos_raw = (pos_stop * mult).clip(-1.0, 1.0)
    # only act on meaningful exposure changes (hysteresis)
    risk_pos = _hysteresis(risk_pos_raw, band=0.25)

    cost = CostModel()
    res_base = run_backtest(oos, base_pos, cost)
    res_risk = run_backtest(oos, risk_pos, cost)
    s_base = metrics.summary(res_base, oos)
    s_risk = metrics.summary(res_risk, oos)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rec = {"symbol": symbol, "timeframe": timeframe,
           "base": s_base, "risk_managed": s_risk}
    (Path(out_dir) / f"{symbol.replace('/','-')}_{timeframe}.json").write_text(
        json.dumps(rec, indent=2, default=str))
    return s_base, s_risk


def _hysteresis(pos: pd.Series, band: float = 0.25) -> pd.Series:
    """Hold current exposure unless target moves by >= band (or flips sign)."""
    arr = pos.to_numpy(); out = np.zeros(len(arr)); held = 0.0
    for i, t in enumerate(arr):
        if np.sign(t) != np.sign(held) or abs(t - held) >= band:
            held = t
        out[i] = held
    return pd.Series(out, index=pos.index)


def _bpy(tf: str) -> float:
    return {"1h": 24 * 365.0, "4h": 6 * 365.0, "1d": 365.0}.get(tf, 365.0)


def _fmt(s):
    return (f"ret={s['total_return_pct']:7.1f}%  sharpe={s['sharpe']:5.2f}  "
            f"maxDD={s['max_drawdown_pct']:6.1f}%  fills={s['num_fills']:4d}  "
            f"cost={s['cost_drag_pct']:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--source", default="okx")
    args = ap.parse_args()
    sb, sr = run(args.symbol, args.timeframe, args.source)
    print(f"\n=== {args.symbol} {args.timeframe} (OOS, after costs) ===")
    print(f"  Buy & Hold : ret={sb['buyhold_return_pct']:7.1f}%  sharpe={sb['buyhold_sharpe']:5.2f}  maxDD={sb['buyhold_maxdd_pct']:6.1f}%")
    print(f"  Base       : {_fmt(sb)}")
    print(f"  Risk-mgd   : {_fmt(sr)}")


if __name__ == "__main__":
    main()
