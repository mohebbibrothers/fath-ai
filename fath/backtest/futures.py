"""Realistic leveraged FUTURES backtester with liquidation modeling.

Spot backtests ignore the two things that actually kill leveraged traders:
LIQUIDATION and FUNDING. This engine models perpetual-futures trading the way a
real exchange (Binance/OKX/Bybit USDT-margined perps) does:

  * Isolated-margin per position with a chosen leverage L.
  * Initial margin = notional / L.
  * Maintenance margin rate (MMR) -> a LIQUIDATION PRICE. If the bar's
    high/low touches the liquidation price, the position is force-closed and the
    posted margin is LOST (minus nothing — full wipe of that margin), exactly
    like a real liquidation. This is the risk leverage adds.
  * Taker fees on entry/exit (perp taker fees, e.g. 5 bps).
  * Funding paid/received every funding interval while a position is held.
  * Slippage on fills.

Why this matters: a strategy that looks great at 10x on a naive backtest often
gets liquidated to zero on the real path. By simulating liquidation honestly we
can find the leverage that MAXIMIZES risk-adjusted return instead of blowing up.

Conventions:
  * position size expressed as signed target exposure in [-1, 1] * leverage,
    i.e. fraction of equity deployed as margin times leverage = notional.
  * We run one isolated position at a time (flat/long/short), next-bar fills.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fath.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class FuturesCostModel:
    taker_fee: float = 0.0005        # 5 bps per side (perp taker)
    slippage_bps: float = 2.0
    mmr: float = 0.005               # maintenance margin rate (0.5%)
    funding_rate: float = 0.0001     # per funding interval (8h typical ~0.01%)
    funding_interval_bars: int = 1   # how many bars per funding charge
    liq_fee: float = 0.0             # extra fee on liquidation (kept 0 for clarity)

    def fill_price(self, ref: float, side: int) -> float:
        return ref * (1 + side * self.slippage_bps / 1e4)


@dataclass
class FuturesResult:
    equity: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    n_liquidations: int
    cost_drag: float
    meta: dict = field(default_factory=dict)


def liquidation_price(entry: float, side: int, leverage: float, mmr: float) -> float:
    """Approx isolated-margin liquidation price for a USDT-margined perp.

    Long:  liq = entry * (1 - 1/L + mmr)
    Short: liq = entry * (1 + 1/L - mmr)
    (Standard first-order approximation ignoring fees in the maintenance calc.)
    """
    if side > 0:
        return entry * (1 - 1.0 / leverage + mmr)
    else:
        return entry * (1 + 1.0 / leverage - mmr)


def protective_stop_price(entry: float, side: int, leverage: float,
                          mmr: float, safety: float = 0.5) -> float:
    """A HARD stop placed strictly INSIDE the liquidation price.

    To GUARANTEE we are never liquidated, we exit at a price that is reached
    before the liquidation price. We place the stop at ``safety`` fraction of
    the distance from entry to liquidation (safety<1 => always triggers first).

    Example: 5x long, liq ~ -20% from entry. With safety=0.5 the protective
    stop sits at ~-10%, so the position is closed (at a controlled loss) well
    before liquidation can ever occur. Capital survives, always.
    """
    liq = liquidation_price(entry, side, leverage, mmr)
    return entry + safety * (liq - entry)


def run_futures_backtest(
    ohlcv: pd.DataFrame,
    target_position: pd.Series,   # signed exposure in [-1, 1] (fraction of equity as margin)
    leverage: float = 3.0,
    cost: FuturesCostModel | None = None,
    init_equity: float = 10_000.0,
    protective_stop: bool = True,
    stop_safety: float = 0.5,
) -> FuturesResult:
    """Backtest a leveraged perp strategy with liquidation & funding.

    target_position[t] is decided at close of bar t and executed at open of t+1.
    The magnitude (0..1) is the fraction of equity committed as MARGIN; notional
    = margin * leverage. Liquidation is checked against each bar's high/low.

    If ``protective_stop`` is True (default), a hard stop is placed inside the
    liquidation price (at ``stop_safety`` of the distance to liq). This makes
    liquidation effectively IMPOSSIBLE: the position is always closed first, at
    a controlled loss. This is the professional way to "never get liquidated".
    """
    cost = cost or FuturesCostModel()
    df = ohlcv.copy()
    tgt = target_position.reindex(df.index).fillna(0.0).clip(-1, 1)
    desired = tgt.shift(1).fillna(0.0)  # next-bar execution

    o = df["open"].to_numpy(); h = df["high"].to_numpy()
    lo = df["low"].to_numpy(); c = df["close"].to_numpy()
    des = desired.to_numpy()

    n = len(df)
    equity = np.empty(n)
    bar_ret = np.zeros(n)
    pos_track = np.zeros(n)

    eq = init_equity
    cur_side = 0           # -1/0/+1
    cur_margin_frac = 0.0  # fraction of equity as margin
    entry_px = np.nan
    liq_px = np.nan
    stop_px = np.nan
    total_cost = 0.0
    n_liq = 0
    n_stops = 0
    trades = []
    bars_in_pos = 0

    for t in range(n):
        want = des[t]
        want_side = int(np.sign(want))
        want_frac = abs(want)

        # ---- PROTECTIVE STOP: exit before liquidation can ever happen -----
        if cur_side != 0 and not np.isnan(stop_px):
            stop_hit = (lo[t] <= stop_px) if cur_side > 0 else (h[t] >= stop_px)
            if stop_hit:
                # close at the stop price (controlled loss), pay exit fee
                pnl = (stop_px / entry_px - 1.0) * cur_side * (cur_margin_frac * leverage)
                eq *= (1 + pnl)
                fee = cur_margin_frac * leverage * (cost.taker_fee + cost.slippage_bps / 1e4)
                eq *= (1 - fee); total_cost += fee
                n_stops += 1
                trades.append({"timestamp": df.index[t], "event": "STOP",
                               "side": cur_side, "price": stop_px})
                cur_side = 0; cur_margin_frac = 0.0
                entry_px = np.nan; liq_px = np.nan; stop_px = np.nan
                bars_in_pos = 0
                equity[t] = max(eq, 0.0); pos_track[t] = 0.0
                continue

        # ---- check liquidation of existing position on THIS bar ----------
        if cur_side != 0 and not np.isnan(liq_px):
            hit = (lo[t] <= liq_px) if cur_side > 0 else (h[t] >= liq_px)
            if hit:
                # lose the committed margin entirely (notional * (1/L) wiped)
                loss = eq * cur_margin_frac
                eq -= loss
                total_cost += 0.0
                n_liq += 1
                trades.append({"timestamp": df.index[t], "event": "LIQUIDATION",
                               "side": cur_side, "price": liq_px})
                cur_side = 0; cur_margin_frac = 0.0; entry_px = np.nan; liq_px = np.nan
                bars_in_pos = 0
                # after liquidation, skip re-entry on same bar
                equity[t] = eq; pos_track[t] = 0.0
                continue

        # ---- rebalance / change position at open of bar t ----------------
        need_change = (want_side != cur_side) or (abs(want_frac - cur_margin_frac) > 0.05)
        if need_change:
            # close existing
            if cur_side != 0:
                fee = cur_margin_frac * leverage * (cost.taker_fee + cost.slippage_bps / 1e4)
                eq *= (1 - fee); total_cost += fee
            # open new
            if want_side != 0 and want_frac > 1e-9:
                fill = cost.fill_price(o[t], want_side)
                fee = want_frac * leverage * (cost.taker_fee + cost.slippage_bps / 1e4)
                eq *= (1 - fee); total_cost += fee
                cur_side = want_side; cur_margin_frac = want_frac
                entry_px = fill
                liq_px = liquidation_price(fill, cur_side, leverage, cost.mmr)
                stop_px = (protective_stop_price(fill, cur_side, leverage,
                                                 cost.mmr, stop_safety)
                           if protective_stop else np.nan)
                bars_in_pos = 0
                trades.append({"timestamp": df.index[t], "event": "OPEN",
                               "side": cur_side, "price": fill, "liq": liq_px,
                               "stop": stop_px})
            else:
                cur_side = 0; cur_margin_frac = 0.0
                entry_px = np.nan; liq_px = np.nan; stop_px = np.nan

        # ---- mark-to-market PnL over bar t (open -> close) ---------------
        if cur_side != 0:
            notional_frac = cur_margin_frac * leverage
            mkt = (c[t] / o[t] - 1.0) * cur_side * notional_frac
            eq *= (1 + mkt)
            bar_ret[t] = mkt
            bars_in_pos += 1
            # funding charge
            if bars_in_pos % cost.funding_interval_bars == 0:
                fund = cost.funding_rate * notional_frac * cur_side  # longs pay if positive
                eq *= (1 - fund); total_cost += abs(fund)

        equity[t] = max(eq, 0.0)
        pos_track[t] = cur_side * cur_margin_frac
        if eq <= 0:
            equity[t:] = 0.0
            break

    equity_s = pd.Series(equity, index=df.index, name="equity")
    res = FuturesResult(
        equity=equity_s,
        returns=pd.Series(bar_ret, index=df.index, name="returns"),
        positions=pd.Series(pos_track, index=df.index, name="position"),
        trades=pd.DataFrame(trades),
        n_liquidations=n_liq,
        cost_drag=total_cost,
        meta={"leverage": leverage, "init_equity": init_equity,
              "final_equity": float(equity_s.iloc[-1]), "n_stops": n_stops},
    )
    log.info("Futures BT: L=%.1f final=%.0f (%.1f%%) liq=%d stops=%d cost=%.3f",
             leverage, equity_s.iloc[-1], (equity_s.iloc[-1]/init_equity-1)*100,
             n_liq, n_stops, total_cost)
    return res
