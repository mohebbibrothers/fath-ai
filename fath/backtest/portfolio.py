"""Portfolio allocation across charts (symbols × timeframes).

You asked for the bot to ALLOCATE MORE CAPITAL to the charts that perform best.
The professional way to do this without cheating (look-ahead) is:

  * Split the OOS period into an ALLOCATION-LEARNING window (earlier) and a
    LIVE-EVALUATION window (later).
  * Score each chart on the learning window only (risk-adjusted: Sharpe, with a
    penalty for drawdown and a minimum-trade requirement).
  * Convert scores to weights (softmax / proportional), capped for
    diversification, and apply those FIXED weights to the later window.

This guarantees the weights are decided using only past information — the same
discipline a real fund uses when sizing strategies. We also compare against an
equal-weight portfolio so we can prove the smart allocation actually helps.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fath.eval import metrics
from fath.utils.logging import get_logger

log = get_logger(__name__)


def score_chart(returns: pd.Series, min_active: int = 10,
                dd_penalty: float = 0.5) -> float:
    """Risk-adjusted score for one chart on a window. Higher = better.

    Combines Sharpe with a drawdown penalty; returns -inf-ish for charts with
    too few active bars (untrustworthy)."""
    active = (returns != 0).sum()
    if active < min_active:
        return -1e9
    eq = (1 + returns).cumprod()
    sh = metrics.sharpe(returns)
    mdd = abs(metrics.max_drawdown(eq))
    return sh - dd_penalty * mdd


def scores_to_weights(scores: dict, top_k: int | None = None,
                      max_weight: float = 0.35, temperature: float = 1.0) -> dict:
    """Convert chart scores to capital weights via clipped softmax.

    Only positive-score charts get capital (we don't allocate to losers). top_k
    keeps only the best k charts. max_weight enforces diversification."""
    items = {k: v for k, v in scores.items() if v > 0 and v > -1e8}
    if not items:
        return {}
    if top_k:
        items = dict(sorted(items.items(), key=lambda kv: -kv[1])[:top_k])
    keys = list(items)
    arr = np.array([items[k] for k in keys], dtype=float)
    arr = arr / max(temperature, 1e-6)
    arr = arr - arr.max()
    w = np.exp(arr)
    w = w / w.sum()
    w = np.minimum(w, max_weight)
    w = w / w.sum()  # renormalize after capping
    return dict(zip(keys, w.tolist()))


def combine_portfolio(chart_returns: dict, weights: dict) -> pd.Series:
    """Blend per-chart return series into one portfolio return series."""
    if not weights:
        return pd.Series(dtype=float)
    df = pd.DataFrame({k: chart_returns[k] for k in weights if k in chart_returns})
    df = df.fillna(0.0)
    w = pd.Series(weights).reindex(df.columns).fillna(0.0)
    return (df * w).sum(axis=1)


def risk_parity_weights(learn: pd.DataFrame, min_sharpe: float = 0.0,
                        max_weight: float = 0.25) -> dict:
    """Inverse-volatility (risk-parity) weights on charts with positive edge.

    DATA-DRIVEN CHOICE: an A/B test showed that chasing past returns (softmax on
    Sharpe) UNDERPERFORMS, while risk-parity on positive-edge charts gives the
    best held-out Sharpe and the lowest drawdown. So this is the default
    allocator. We (a) keep only charts whose learn-window Sharpe > min_sharpe,
    then (b) weight each inversely to its volatility, capped for diversification.
    """
    good = [c for c in learn.columns
            if metrics.sharpe(learn[c].dropna()) > min_sharpe]
    if not good:
        return {}
    inv = {c: 1.0 / (learn[c].std(ddof=0) + 1e-9) for c in good}
    tot = sum(inv.values())
    w = {k: v / tot for k, v in inv.items()}
    # cap + renormalize for diversification
    w = {k: min(v, max_weight) for k, v in w.items()}
    tot = sum(w.values())
    return {k: v / tot for k, v in w.items()}


def allocate_and_evaluate(chart_returns: dict, learn_frac: float = 0.5,
                          top_k: int | None = 10, max_weight: float = 0.25,
                          method: str = "risk_parity"):
    """Full leak-free allocation experiment.

    chart_returns: {chart_name: per-bar return series (OOS)}.
    method: 'risk_parity' (default, data-driven winner) or 'softmax' (legacy).
    Returns dict with learned weights and both portfolios' metrics on the
    later (held-out) evaluation window.
    """
    # align all charts on a common time grid
    df = pd.DataFrame(chart_returns).sort_index()
    if df.empty:
        return {}
    n = len(df)
    cut = int(n * learn_frac)
    learn, evalw = df.iloc[:cut], df.iloc[cut:]

    scores = {c: score_chart(learn[c].dropna()) for c in df.columns}
    if method == "risk_parity":
        weights = risk_parity_weights(learn, min_sharpe=0.0, max_weight=max_weight)
    else:
        weights = scores_to_weights(scores, top_k=top_k, max_weight=max_weight)

    # smart-weighted portfolio on evaluation window
    smart = combine_portfolio({c: evalw[c] for c in evalw.columns}, weights)
    # equal-weight benchmark (only over charts that were active in learn)
    eligible = [c for c in df.columns if scores[c] > -1e8]
    eq_w = {c: 1.0 / len(eligible) for c in eligible} if eligible else {}
    equal = combine_portfolio({c: evalw[c] for c in evalw.columns}, eq_w)

    def _summ(r):
        r = r.dropna()
        eq = (1 + r).cumprod()
        return {
            "total_return_pct": float((eq.iloc[-1] - 1) * 100) if len(eq) else 0.0,
            "sharpe": metrics.sharpe(r),
            "max_drawdown_pct": metrics.max_drawdown(eq) * 100 if len(eq) else 0.0,
        }

    return {
        "weights": weights,
        "smart": _summ(smart),
        "equal_weight": _summ(equal),
        "n_charts_funded": len(weights),
    }
