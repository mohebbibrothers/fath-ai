"""Honest performance metrics.

We report the numbers that actually matter and are hard to fake, and we always
compare strategy vs buy-and-hold so a bull-market backtest can't masquerade as
skill.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ann_factor(index: pd.DatetimeIndex) -> float:
    """Annualization factor from median bar spacing."""
    if len(index) < 3:
        return 1.0
    dt = np.median(np.diff(index.view("int64"))) / 1e9  # seconds
    bars_per_year = (365.25 * 24 * 3600) / dt
    return float(bars_per_year)


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    r = returns.dropna()
    if r.std(ddof=0) == 0 or len(r) < 2:
        return 0.0
    af = _ann_factor(r.index)
    return float((r.mean() - rf) / r.std(ddof=0) * np.sqrt(af))


def sortino(returns: pd.Series) -> float:
    r = returns.dropna()
    downside = r[r < 0]
    if downside.std(ddof=0) == 0 or len(r) < 2:
        return 0.0
    af = _ann_factor(r.index)
    return float(r.mean() / downside.std(ddof=0) * np.sqrt(af))


def max_drawdown(equity: pd.Series) -> float:
    eq = equity.dropna()
    if eq.empty:
        return 0.0
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())


def cagr(equity: pd.Series) -> float:
    eq = equity.dropna()
    if len(eq) < 2:
        return 0.0
    years = (eq.index[-1] - eq.index[0]).total_seconds() / (365.25 * 24 * 3600)
    if years <= 0:
        return 0.0
    return float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1)


def hit_rate(returns: pd.Series) -> float:
    r = returns[returns != 0].dropna()
    if r.empty:
        return 0.0
    return float((r > 0).mean())


def summary(result, ohlcv: pd.DataFrame) -> dict:
    """Full metric bundle, including a buy-and-hold benchmark."""
    eq = result.equity
    ret = result.returns
    bh = ohlcv["close"] / ohlcv["close"].iloc[0]
    bh_eq = bh * result.meta.get("init_equity", 1.0)

    return {
        "final_equity": float(eq.iloc[-1]),
        "total_return_pct": float((eq.iloc[-1] / eq.iloc[0] - 1) * 100),
        "cagr_pct": cagr(eq) * 100,
        "sharpe": sharpe(ret),
        "sortino": sortino(ret),
        "max_drawdown_pct": max_drawdown(eq) * 100,
        "hit_rate_pct": hit_rate(ret) * 100,
        "num_fills": int(len(result.trades)),
        "cost_drag_pct": float(result.cost_drag * 100),
        "buyhold_return_pct": float((bh_eq.iloc[-1] / bh_eq.iloc[0] - 1) * 100),
        "buyhold_sharpe": sharpe(ohlcv["close"].pct_change()),
        "buyhold_maxdd_pct": max_drawdown(bh_eq) * 100,
    }
