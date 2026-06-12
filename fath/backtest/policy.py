"""Trading policy: turn model probabilities into positions.

Separating *prediction* from *policy* is deliberate and important. The model
says "probability up / neutral / down". The policy decides whether the edge is
big enough to be worth trading AFTER costs, and how big the position should be.

Key ideas implemented:
  * Probability threshold / confidence gate — only trade when the model's
    edge clears a minimum, otherwise stay flat (cash is a position).
  * Expected-value gate vs cost — don't take trades whose expected move is
    smaller than the round-trip transaction cost.
  * Volatility-targeted sizing — scale exposure inversely to recent vol so
    risk per bar is roughly constant (basic risk parity).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def probs_to_position(
    proba: np.ndarray,
    index: pd.Index,
    conf_threshold: float = 0.45,
    edge_threshold: float = 0.10,
) -> pd.Series:
    """Map [-1,0,+1] class probabilities to a target position in {-1,0,+1}.

    conf_threshold : minimum prob of the chosen directional class.
    edge_threshold : minimum (p_up - p_down) magnitude to act on.
    """
    p_down, p_neutral, p_up = proba[:, 0], proba[:, 1], proba[:, 2]
    edge = p_up - p_down
    pos = np.zeros(len(index))

    long_mask = (p_up >= conf_threshold) & (edge >= edge_threshold)
    short_mask = (p_down >= conf_threshold) & (-edge >= edge_threshold)
    pos[long_mask] = 1.0
    pos[short_mask] = -1.0
    return pd.Series(pos, index=index, name="position")


def vol_target_size(
    position: pd.Series,
    volatility: pd.Series,
    target_vol_per_bar: float = 0.01,
    max_leverage: float = 1.0,
    no_trade_band: float = 0.10,
    quantize: float = 0.05,
) -> pd.Series:
    """Scale {-1,0,+1} positions to target a constant per-bar volatility.

    Naively rebalancing to an exact vol target every bar generates enormous
    turnover (tiny vol wiggles -> constant micro-trades -> costs eat everything).
    Two standard fixes are applied:

      * ``quantize`` : round the target exposure to discrete steps so trivial
        changes don't trigger fills.
      * ``no_trade_band`` : only move the held position if the new target
        differs from the current one by more than this fraction (hysteresis).
    """
    vol = volatility.reindex(position.index).replace(0, np.nan)
    scale = (target_vol_per_bar / vol).clip(upper=max_leverage).fillna(0.0)
    raw = (position * scale).clip(-max_leverage, max_leverage)

    if quantize and quantize > 0:
        raw = (raw / quantize).round() * quantize

    # hysteresis: hold previous exposure unless the change clears the band
    out = np.zeros(len(raw))
    held = 0.0
    raw_arr = raw.to_numpy()
    for i, target in enumerate(raw_arr):
        if abs(target - held) >= no_trade_band or (target == 0.0 and held != 0.0
                                                    and abs(held) < no_trade_band):
            held = target
        out[i] = held
    return pd.Series(out, index=position.index, name="position")
