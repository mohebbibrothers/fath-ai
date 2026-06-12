"""Event-driven, cost-aware vectorized backtester.

A backtest that ignores costs is a fairy tale. This engine models:
  * taker fee on entry and exit (configurable; defaults to realistic spot fees)
  * slippage proportional to volatility / a fixed bps assumption
  * one position at a time (flat / long / short), next-bar execution

Execution model — the honest part:
  Signals are generated from information available at the CLOSE of bar t.
  We therefore execute at the OPEN of bar t+1. Using the same bar's close to
  both decide and fill is a classic look-ahead bug; we explicitly avoid it.

Outputs an equity curve and a trade blotter for the metrics module.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fath.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class CostModel:
    taker_fee: float = 0.0010      # 10 bps per side (typical retail spot taker)
    slippage_bps: float = 2.0       # 2 bps assumed slippage per fill
    funding_per_bar: float = 0.0    # set >0 for perp funding on held bars

    def fill_price(self, ref_price: float, side: int) -> float:
        slip = self.slippage_bps / 1e4
        return ref_price * (1 + side * slip)


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    cost_drag: float
    meta: dict = field(default_factory=dict)


def run_backtest(
    ohlcv: pd.DataFrame,
    target_position: pd.Series,
    cost: CostModel | None = None,
    init_equity: float = 10_000.0,
) -> BacktestResult:
    """Backtest a target-position series in {-1,0,+1} (or fractional).

    ``target_position`` is the desired position decided at each bar's close.
    It is executed at the NEXT bar's open. We compute bar-by-bar PnL on the
    held position, subtracting transaction costs whenever the position changes.
    """
    cost = cost or CostModel()
    df = ohlcv.copy()
    pos_target = target_position.reindex(df.index).fillna(0.0)

    # Position actually held during bar t was decided at t-1 (executed at open t)
    held = pos_target.shift(1).fillna(0.0)

    open_ = df["open"].to_numpy()
    close = df["close"].to_numpy()
    held_arr = held.to_numpy()

    n = len(df)
    equity = np.empty(n)
    bar_ret = np.zeros(n)
    eq = init_equity
    total_cost = 0.0
    prev_pos = 0.0
    trades = []

    for t in range(n):
        pos = held_arr[t]

        # transaction cost when the position changes (turnover at open of bar t)
        dpos = pos - prev_pos
        if abs(dpos) > 1e-12:
            side = int(np.sign(dpos))
            fill = cost.fill_price(open_[t], side)
            fee = abs(dpos) * (cost.taker_fee + cost.slippage_bps / 1e4)
            eq *= (1 - fee)
            total_cost += fee
            trades.append(
                {"timestamp": df.index[t], "dpos": dpos, "fill": fill, "fee": fee}
            )

        # market PnL on the held position over bar t (open -> close)
        if abs(pos) > 1e-12:
            mkt = (close[t] / open_[t] - 1.0) * pos
            mkt -= cost.funding_per_bar * abs(pos)  # funding while held
            eq *= (1 + mkt)
            bar_ret[t] = mkt

        equity[t] = eq
        prev_pos = pos

    equity_s = pd.Series(equity, index=df.index, name="equity")
    returns_s = pd.Series(bar_ret, index=df.index, name="returns")
    trades_df = pd.DataFrame(trades)
    log.info("Backtest done: final equity %.2f (%.2f%%), %d fills, cost drag %.4f",
             eq, (eq / init_equity - 1) * 100, len(trades_df), total_cost)
    return BacktestResult(
        equity=equity_s,
        returns=returns_s,
        positions=held,
        trades=trades_df,
        cost_drag=total_cost,
        meta={"init_equity": init_equity, "final_equity": eq},
    )
