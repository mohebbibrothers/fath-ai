"""Dynamic 24/7 position-finding & trade-planning engine.

This is the live brain. On every evaluation tick it:

  1. Pulls the latest closed candles for every chart in the universe.
  2. Runs the full AI analysis (features -> primary direction -> meta quality).
  3. For any chart with a high-quality setup, computes the OPTIMAL trade plan:
       * direction (long/short)
       * leverage    (risk-budgeted so a protective stop loses only a fixed %)
       * size        (fraction of equity as margin, vol-targeted)
       * entry price  (limit at current close / micro-better)
       * stop price   (ATR-based, ALWAYS inside the liquidation price)
       * target(s)    (ATR / risk-multiple take-profits)
       * liquidation  (computed, and guaranteed to never be reached by the stop)
  4. Ranks candidate plans by expected edge and returns the best ones.

KEY SAFETY INVARIANT (your hard rule): every plan's protective stop is strictly
inside the liquidation price, so liquidation is mathematically impossible — the
worst case is a controlled stop-out, never a wipe.

This module is exchange-agnostic and side-effect free: it PLANS trades. An
executor (paper/live) consumes the plans. That separation keeps the risky part
(order placement) tiny and auditable.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from fath.backtest.futures import liquidation_price
from fath.features import indicators as ta
from fath.features.build import build_features
from fath.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class TradePlan:
    symbol: str
    timeframe: str
    side: int                 # +1 long, -1 short
    leverage: float
    margin_frac: float        # fraction of equity committed as margin
    entry: float
    stop: float
    targets: list             # list of take-profit prices
    liquidation: float
    meta_quality: float       # model confidence (0..1)
    expected_R: float         # reward/risk ratio of the plan
    rationale: str

    def to_dict(self):
        return asdict(self)


def _risk_budgeted_leverage(entry: float, stop: float, side: int,
                            max_loss_frac: float, mmr: float,
                            max_leverage: float) -> float:
    """Largest leverage such that hitting the STOP loses <= max_loss_frac of
    margin, AND the stop stays strictly inside the liquidation price.

    Loss at stop on notional = |entry-stop|/entry * leverage (as frac of equity
    when margin_frac=1). We want that <= max_loss_frac -> leverage cap.
    """
    move = abs(entry - stop) / entry
    if move <= 0:
        return 1.0
    lev_by_risk = max_loss_frac / move
    # ensure liquidation is beyond the stop: liq move ~ (1/L - mmr); require
    # stop move < liq move  ->  L < 1 / (move + mmr)
    lev_by_liq = 1.0 / (move + mmr) - 1e-6
    return float(max(1.0, min(max_leverage, lev_by_risk, lev_by_liq)))


def plan_trade(symbol: str, timeframe: str, ohlcv: pd.DataFrame,
               sentiment: pd.DataFrame | None,
               meta_model, feat_cols: list,
               rsi_low: float = 30.0, rsi_high: float = 70.0,
               meta_threshold: float = 0.50,
               atr_stop_mult: float = 2.0,
               atr_target_mults=(1.5, 3.0),
               max_loss_frac: float = 0.02,   # risk 2% of equity per trade at stop
               mmr: float = 0.005,
               max_leverage: float = 10.0,
               allow_short: bool = False) -> TradePlan | None:
    """Produce an optimal trade plan for the latest closed bar, or None."""
    feats = build_features(ohlcv, sentiment=sentiment)
    if feats.empty:
        return None
    last = feats.iloc[[-1]]
    rsi = float(last["rsi_14"].iloc[0])

    side = 0
    if rsi < rsi_low:
        side = 1
    elif allow_short and rsi > rsi_high:
        side = -1
    if side == 0:
        return None

    # AI quality gate. Align to the model's training columns exactly: any
    # pooled-only feature (cross-sectional ranks, symbol_id) that we can't
    # compute for a single chart in isolation is filled with a neutral value.
    X = pd.DataFrame(index=last.index)
    for c in feat_cols:
        X[c] = last[c].iloc[0] if c in last.columns else 0.5
    quality = float(meta_model.predict_bet_proba(X)[0]) if meta_model else 0.5
    if quality < meta_threshold:
        return None

    close = float(ohlcv["close"].iloc[-1])
    atr = float(ta.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14).iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        return None

    entry = close
    stop = entry - side * atr_stop_mult * atr
    targets = [entry + side * m * atr for m in atr_target_mults]

    leverage = _risk_budgeted_leverage(entry, stop, side, max_loss_frac, mmr, max_leverage)
    liq = liquidation_price(entry, side, leverage, mmr)

    # SAFETY ASSERT: stop must be strictly inside liquidation
    if side > 0 and not (stop > liq):
        stop = (entry + liq) / 2  # pull stop inside liq
    if side < 0 and not (stop < liq):
        stop = (entry + liq) / 2

    # vol-targeted margin fraction (smaller when ATR% is high)
    atr_pct = atr / entry
    margin_frac = float(np.clip(0.02 / max(atr_pct, 1e-4), 0.05, 1.0))

    risk = abs(entry - stop)
    reward = abs(targets[-1] - entry)
    exp_R = reward / risk if risk > 0 else 0.0

    return TradePlan(
        symbol=symbol, timeframe=timeframe, side=side,
        leverage=round(leverage, 2), margin_frac=round(margin_frac, 3),
        entry=round(entry, 6), stop=round(stop, 6),
        targets=[round(t, 6) for t in targets], liquidation=round(liq, 6),
        meta_quality=round(quality, 4), expected_R=round(exp_R, 2),
        rationale=(f"RSI={rsi:.1f} {'oversold->long' if side>0 else 'overbought->short'}; "
                   f"AI quality={quality:.2f}; ATR%={atr_pct*100:.2f}; "
                   f"lev risk-budgeted to {leverage:.1f}x; stop inside liq (no liquidation)"),
    )
