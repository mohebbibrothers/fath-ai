"""End-to-end research pipeline.

  data -> features -> labels -> walk-forward (purged/embargoed) model ->
  out-of-sample probabilities -> policy -> realistic backtest -> honest report

Run:
    python -m scripts.run_pipeline --config config/default.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fath.backtest.engine import CostModel, run_backtest
from fath.backtest.policy import probs_to_position, vol_target_size
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

log = get_logger("pipeline")


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    out_dir = Path(cfg.get("output_dir", "artifacts"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. data ------------------------------------------------------------
    d = cfg["data"]
    try:
        ohlcv = store.load(d["symbol"], d["timeframe"], d["source"])
    except FileNotFoundError:
        ohlcv = store.fetch_and_cache(d["symbol"], d["timeframe"], d["since"], d["source"])
    log.info("Data: %d candles %s..%s", len(ohlcv), ohlcv.index.min(), ohlcv.index.max())

    # --- 2. features --------------------------------------------------------
    feats = build_features(ohlcv)

    # --- 3. labels ----------------------------------------------------------
    lc = cfg["labels"]
    logret = np.log(ohlcv["close"]).diff()
    volatility = logret.rolling(24).std(ddof=0)  # regime-adaptive vol
    labels = triple_barrier_labels(
        ohlcv["close"], volatility,
        tp_mult=lc["tp_mult"], sl_mult=lc["sl_mult"],
        max_horizon=lc["max_horizon"], min_ret=lc["min_ret"],
    )

    # align features & labels
    common = feats.index.intersection(labels.index)
    X = feats.loc[common]
    y = labels.loc[common, "label"]
    t1 = labels.loc[common, "t1"]
    weights = average_uniqueness_weights(t1, common)
    log.info("Aligned dataset: X=%s, label dist=%s", X.shape, y.value_counts().to_dict())

    # integer position of each label's t1 (for purging)
    pos_of = {ts: i for i, ts in enumerate(common)}
    t1_positions = np.array([pos_of.get(t, -1) for t in t1.values])

    # --- 4. walk-forward, purged/embargoed ---------------------------------
    cv = cfg["cv"]
    n = len(X)
    oos_proba = np.full((n, 3), np.nan)
    fold_reports = []

    for fold, (tr, te) in enumerate(
        walk_forward_splits(n, cv["n_splits"], cv["test_frac"], cv["expanding"])
    ):
        tr_clean = purge_embargo(tr, te, t1_positions, cv["embargo_frac"], n_samples=n)
        if len(tr_clean) < 200:
            log.warning("Fold %d skipped: too few clean train samples (%d)", fold, len(tr_clean))
            continue
        clf = BarrierClassifier()
        clf.fit(
            X.iloc[tr_clean], y.iloc[tr_clean],
            sample_weight=weights.iloc[tr_clean].to_numpy(),
        )
        proba = clf.predict_proba(X.iloc[te])
        oos_proba[te] = proba
        acc = (clf.predict_signed(X.iloc[te]) == y.iloc[te].to_numpy()).mean()
        fold_reports.append({"fold": fold, "n_train": int(len(tr_clean)),
                             "n_test": int(len(te)), "oos_accuracy": float(acc)})
        log.info("Fold %d: train=%d test=%d OOS acc=%.3f", fold, len(tr_clean), len(te), acc)

    # --- 5. policy on OOS probabilities ------------------------------------
    valid = ~np.isnan(oos_proba).any(axis=1)
    idx_valid = X.index[valid]
    proba_valid = oos_proba[valid]
    pc = cfg["policy"]
    raw_pos = probs_to_position(
        proba_valid, idx_valid,
        conf_threshold=pc["conf_threshold"], edge_threshold=pc["edge_threshold"],
    )
    # Low-turnover discrete policy: hold the current direction until the signal
    # flips to the opposite side. This is what actually controls cost drag.
    sized_pos = raw_pos.replace(0.0, np.nan).ffill().fillna(0.0)

    # --- 6. realistic backtest on the OOS span -----------------------------
    cc = cfg["costs"]
    cost = CostModel(taker_fee=cc["taker_fee"], slippage_bps=cc["slippage_bps"],
                     funding_per_bar=cc["funding_per_bar"])
    oos_ohlcv = ohlcv.loc[idx_valid.min():]
    result = run_backtest(oos_ohlcv, sized_pos, cost=cost)

    # --- 7. honest report ---------------------------------------------------
    stats = metrics.summary(result, oos_ohlcv)
    stats["mean_oos_accuracy"] = float(np.mean([f["oos_accuracy"] for f in fold_reports])) if fold_reports else None
    report = {
        "config": cfg,
        "data_span": [str(ohlcv.index.min()), str(ohlcv.index.max())],
        "n_features": X.shape[1],
        "folds": fold_reports,
        "oos_metrics": stats,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    _write_markdown(out_dir / "report.md", report)
    result.equity.to_frame().to_parquet(out_dir / "equity.parquet")
    log.info("Report written to %s", out_dir / "report.md")
    print("\n" + (out_dir / "report.md").read_text())


def _write_markdown(path: Path, report: dict):
    m = report["oos_metrics"]
    lines = [
        "# fath-ai — Out-of-Sample Research Report",
        "",
        "> Generated by the purged/embargoed walk-forward pipeline. All metrics",
        "> below are **out-of-sample** and **after costs**. Compare against the",
        "> buy-and-hold benchmark before getting excited.",
        "",
        f"- **Data span:** {report['data_span'][0]} → {report['data_span'][1]}",
        f"- **Features:** {report['n_features']}",
        f"- **Mean OOS classification accuracy:** {m.get('mean_oos_accuracy')}",
        "",
        "## Strategy (after fees + slippage)",
        "",
        "| Metric | Strategy | Buy & Hold |",
        "|---|---|---|",
        f"| Total return % | {m['total_return_pct']:.2f} | {m['buyhold_return_pct']:.2f} |",
        f"| CAGR % | {m['cagr_pct']:.2f} | — |",
        f"| Sharpe | {m['sharpe']:.2f} | {m['buyhold_sharpe']:.2f} |",
        f"| Sortino | {m['sortino']:.2f} | — |",
        f"| Max drawdown % | {m['max_drawdown_pct']:.2f} | {m['buyhold_maxdd_pct']:.2f} |",
        f"| Hit rate % | {m['hit_rate_pct']:.2f} | — |",
        f"| # fills | {m['num_fills']} | — |",
        f"| Cost drag % | {m['cost_drag_pct']:.2f} | — |",
        "",
        "## How to read this honestly",
        "",
        "- If **Strategy Sharpe** is not clearly above **Buy & Hold Sharpe**, the",
        "  model has no demonstrated edge — do **not** trade it.",
        "- A high return with a huge max drawdown is not skill, it's leverage/luck.",
        "- If OOS accuracy is ~33% (3-class chance) the features carry little signal.",
        "- Re-run across symbols/timeframes; an edge that only appears once is noise.",
    ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
