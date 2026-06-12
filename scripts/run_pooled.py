"""Pooled cross-asset pipeline (v0.5).

Trains ONE primary model + ONE meta model on the POOLED data of the entire
universe, validated by global-time walk-forward (no cross-asset leakage), then
backtests the resulting signals per symbol and aggregates honestly.

This is the proper realization of "train on the whole market".

Run:
    python -m scripts.run_pooled --timeframe 1d
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
from fath.models.classifier import BarrierClassifier
from fath.models.meta_labeling import MetaModel, bet_size_from_proba, make_meta_target
from fath.models.pooled import build_pooled_dataset, time_split
from fath.utils.logging import get_logger

log = get_logger("pooled")

NON_FEATURE = {"label", "t1", "weight", "symbol", "ts"}


def _feat_cols(pooled):
    return [c for c in pooled.columns if c not in NON_FEATURE]


def run(timeframe="1d", source="okx", conf_edge=0.03, meta_threshold=0.55,
        use_sentiment=True, out_dir="artifacts_pooled"):
    data = load_universe(DEFAULT_UNIVERSE, timeframe, source)
    if len(data) < 3:
        raise RuntimeError("Need more cached symbols; run fetch_universe first.")

    sent_map = {}
    if use_sentiment:
        try:
            from fath.data.sentiment import fetch_fear_greed, merge_sentiment
            fng = fetch_fear_greed(0)
            for sym, df in data.items():
                sent_map[sym] = merge_sentiment(df, fng)
        except Exception as exc:  # noqa: BLE001
            log.warning("Sentiment off (%s)", exc)

    pooled = build_pooled_dataset(data, sent_map)
    feat_cols = _feat_cols(pooled)
    y = pooled["label"].to_numpy()

    # OOS predictions accumulated across folds
    oos_sig = np.zeros(len(pooled))
    oos_mp = np.full(len(pooled), np.nan)
    accs = []

    for fold, (tr, te) in enumerate(time_split(pooled, n_splits=4, test_frac=0.30)):
        if tr.sum() < 1000 or te.sum() < 100:
            continue
        Xtr, Xte = pooled.loc[tr, feat_cols], pooled.loc[te, feat_cols]
        ytr, yte = y[tr], y[te]
        wtr = pooled.loc[tr, "weight"].to_numpy()

        prim = BarrierClassifier().fit(Xtr, ytr, sample_weight=wtr)
        ptr = prim.predict_proba(Xtr); pte = prim.predict_proba(Xte)
        etr = ptr[:, 2] - ptr[:, 0]; ete = pte[:, 2] - pte[:, 0]
        str_ = np.where(etr >= conf_edge, 1, np.where(etr <= -conf_edge, -1, 0))
        ste = np.where(ete >= conf_edge, 1, np.where(ete <= -conf_edge, -1, 0))
        accs.append((prim.predict_signed(Xte) == yte).mean())

        # meta model on pooled training signals
        mmask = str_ != 0
        if mmask.sum() >= 200:
            sig_s = pd.Series(str_[mmask], index=Xtr.index[mmask])
            y_s = pd.Series(ytr[mmask], index=Xtr.index[mmask])
            my = make_meta_target(sig_s, y_s)
            Xm = Xtr.loc[mmask].copy()
            Xm["_sig"] = str_[mmask]; Xm["_edge_abs"] = np.abs(etr[mmask])
            meta = MetaModel().fit(Xm, my.values)
            temask = ste != 0
            Xtem = Xte.loc[temask].copy()
            if len(Xtem):
                Xtem["_sig"] = ste[temask]; Xtem["_edge_abs"] = np.abs(ete[temask])
                pbet = meta.predict_bet_proba(Xtem)
                oos_mp[np.where(te)[0][temask]] = pbet
        oos_sig[te] = ste
        log.info("Fold %d: train=%d test=%d acc=%.4f", fold, tr.sum(), te.sum(), accs[-1])

    pooled["oos_sig"] = oos_sig
    pooled["oos_mp"] = oos_mp

    # ---- per-symbol backtest of the pooled-model signals -------------------
    rows = []
    eq_curves = {}
    for sym, ohlcv in data.items():
        sub = pooled[pooled["symbol"] == sym].copy()
        sub = sub[sub["oos_mp"].notna() & (sub["oos_sig"] != 0)]
        if len(sub) < 30:
            continue
        idx = pd.to_datetime(sub["ts"], utc=True)
        size = sub["oos_sig"].to_numpy() * bet_size_from_proba(
            sub["oos_mp"].to_numpy(), p_threshold=meta_threshold)
        pos = pd.Series(size, index=idx)
        pos = _quantize_hold(pos)

        first = idx.min()
        oos = ohlcv.loc[first:]
        pos = pos.reindex(oos.index).ffill().fillna(0.0)

        # risk overlay
        atr = ta.atr(oos["high"], oos["low"], oos["close"], 14)
        adx = ta.adx(oos["high"], oos["low"], oos["close"], 14)
        rets = oos["close"].pct_change()
        vp = rets.abs().rolling(60).rank(pct=True)
        pos_s = R.apply_trailing_stop(pos, oos["close"], atr, k=3.5)
        mult = (R.regime_gate(adx, vp) *
                R.vol_target_scale(rets, 0.60, 30, _bpy(timeframe))).clip(0, 1)
        mult = (mult / 0.25).round() * 0.25
        rpos = _hyst((pos_s * mult).clip(-1, 1))

        res = run_backtest(oos, rpos, CostModel())
        s = metrics.summary(res, oos)
        eq_curves[sym] = res.equity
        rows.append({"symbol": sym, "ret%": round(s["total_return_pct"], 1),
                     "bh%": round(s["buyhold_return_pct"], 1),
                     "sharpe": round(s["sharpe"], 2),
                     "bh_sharpe": round(s["buyhold_sharpe"], 2),
                     "maxDD%": round(s["max_drawdown_pct"], 1),
                     "fills": s["num_fills"]})

    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_json(f"{out_dir}/pooled_{timeframe}.json", orient="records", indent=2)

    pd.set_option("display.width", 200)
    print(f"\n===== POOLED CROSS-ASSET MODEL @ {timeframe} (OOS, after costs) =====\n")
    print(df.to_string(index=False))
    print(f"\nMean OOS classification accuracy: {np.mean(accs):.4f}")
    print(f"Median strategy Sharpe ......... {df['sharpe'].median():.2f}")
    print(f"Median B&H Sharpe .............. {df['bh_sharpe'].median():.2f}")
    print(f"% beating B&H (Sharpe) ........ {(df['sharpe']>df['bh_sharpe']).mean()*100:.0f}%")
    print(f"% positive return ............. {(df['ret%']>0).mean()*100:.0f}%")
    print(f"Mean strategy return .......... {df['ret%'].mean():.1f}%")
    return df


def _quantize_hold(sizes, q=0.33):
    s = np.sign(sizes.to_numpy()); m = np.round(np.abs(sizes.to_numpy()) / q) * q
    arr = s * m; out = np.zeros(len(arr)); held = 0.0
    for i, t in enumerate(arr):
        if held == 0.0:
            held = t
        elif t != 0 and np.sign(t) != np.sign(held):
            held = t
        elif t != 0 and np.sign(t) == np.sign(held) and abs(t - held) >= 0.34:
            held = t
        out[i] = held
    return pd.Series(out, index=sizes.index)


def _hyst(pos, band=0.25):
    arr = pos.to_numpy(); out = np.zeros(len(arr)); held = 0.0
    for i, t in enumerate(arr):
        if np.sign(t) != np.sign(held) or abs(t - held) >= band:
            held = t
        out[i] = held
    return pd.Series(out, index=pos.index)


def _bpy(tf):
    return {"1h": 24 * 365.0, "4h": 6 * 365.0, "1d": 365.0}.get(tf, 365.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--source", default="okx")
    ap.add_argument("--conf_edge", type=float, default=0.03)
    ap.add_argument("--meta_threshold", type=float, default=0.55)
    args = ap.parse_args()
    run(args.timeframe, args.source, args.conf_edge, args.meta_threshold)


if __name__ == "__main__":
    main()
