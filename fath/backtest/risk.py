"""Advanced risk management.

Returns are vanity; risk is sanity. The sweep showed our biggest weakness is
drawdown (40-72%), not lack of edge. This module adds the controls that turn a
high-return-high-pain strategy into something actually tradeable:

  * Volatility targeting at the PORTFOLIO level — scale exposure so the
    realized risk per bar is roughly constant across calm and wild regimes.
  * ATR-based trailing stop — exit a position when price reverses by k*ATR from
    the best level reached, capping single-trade losses.
  * Drawdown brake — automatically de-risk (cut exposure) when equity falls a
    set % from its peak, and re-risk as it recovers. This directly attacks the
    deep-drawdown problem.
  * Regime gate — only allow full exposure in favorable regimes (e.g. trend
    strength via ADX / volatility percentile).

All of these use only past information (no look-ahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def vol_target_scale(returns: pd.Series, target_ann_vol: float = 0.60,
                     lookback: int = 30, bars_per_year: float = 365.0,
                     max_scale: float = 1.5) -> pd.Series:
    """Scale factor so realized vol ~ target. Uses trailing realized vol only."""
    realized = returns.rolling(lookback).std(ddof=0)
    ann = realized * np.sqrt(bars_per_year)
    scale = (target_ann_vol / ann).clip(upper=max_scale)
    return scale.shift(1).fillna(0.0)  # shift: known only from prior bar


def drawdown_brake(equity_proxy: pd.Series, dd_threshold: float = 0.20,
                   cut_to: float = 0.3) -> pd.Series:
    """Reduce exposure to ``cut_to`` while in a drawdown beyond threshold.

    equity_proxy: a cumulative-return proxy of the *strategy* computed causally
    (we feed it the running strategy equity from the backtest loop in practice;
    here we expose a vectorized approximation for research).
    """
    peak = equity_proxy.cummax()
    dd = equity_proxy / peak - 1.0
    brake = pd.Series(1.0, index=equity_proxy.index)
    brake[dd <= -dd_threshold] = cut_to
    return brake.shift(1).fillna(1.0)


def regime_gate(adx: pd.Series, vol_pct: pd.Series,
                adx_min: float = 18.0, vol_pct_max: float = 0.95) -> pd.Series:
    """1.0 when regime is favorable (trending, not in a vol blow-off), else damp.

    - Require some trend strength (ADX above a floor) for trend-following edge.
    - Avoid extreme-volatility blow-off bars (top vol percentile) where slippage
      and gap risk explode.
    """
    gate = pd.Series(1.0, index=adx.index)
    gate[adx < adx_min] = 0.5
    gate[vol_pct > vol_pct_max] = 0.3
    return gate.shift(1).fillna(1.0)


def apply_trailing_stop(positions: pd.Series, close: pd.Series, atr: pd.Series,
                        k: float = 3.0) -> pd.Series:
    """Flatten a position when price retraces k*ATR from the best level.

    Iterates causally: while in a long, track the highest close since entry; if
    close falls below high - k*ATR, exit (set position 0 until the signal flips).
    Symmetric for shorts.
    """
    pos = positions.to_numpy().astype(float)
    c = close.reindex(positions.index).to_numpy()
    a = atr.reindex(positions.index).to_numpy()
    out = pos.copy()

    in_pos = 0.0
    extreme = np.nan  # highest (long) / lowest (short) close since entry
    stopped_dir = 0   # remember we stopped, to stay flat until sign flips

    for i in range(len(pos)):
        target = pos[i]
        # reset stop memory when the underlying signal flips direction
        if np.sign(target) != np.sign(stopped_dir):
            stopped_dir = 0
        if stopped_dir != 0 and np.sign(target) == np.sign(stopped_dir):
            out[i] = 0.0
            in_pos = 0.0
            continue

        if in_pos == 0.0 and target != 0.0:
            in_pos = target
            extreme = c[i]
        elif in_pos != 0.0:
            if np.sign(target) != np.sign(in_pos):  # signal flip -> follow it
                in_pos = target
                extreme = c[i]
            else:
                if in_pos > 0:
                    extreme = max(extreme, c[i])
                    if not np.isnan(a[i]) and c[i] <= extreme - k * a[i]:
                        out[i] = 0.0; in_pos = 0.0; stopped_dir = np.sign(target); continue
                else:
                    extreme = min(extreme, c[i])
                    if not np.isnan(a[i]) and c[i] >= extreme + k * a[i]:
                        out[i] = 0.0; in_pos = 0.0; stopped_dir = np.sign(target); continue
        out[i] = in_pos
    return pd.Series(out, index=positions.index)
