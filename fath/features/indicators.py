"""Technical indicators, implemented from scratch (no TA-Lib dependency).

CRITICAL DESIGN RULE — NO LOOK-AHEAD:
Every indicator at row t may use ONLY information from rows <= t. We therefore
never use centered windows, never use .shift(-k), and we are explicit about the
fact that the *current* bar's close is known only at bar close. Features that
will be used to predict bar t+1 are computed on closed bars up to t.

These are deliberately vectorized pandas/numpy implementations so behaviour is
transparent and auditable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = _ema(close, fast) - _ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr


def atr(high, low, close, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def bollinger(close: pd.Series, period: int = 20, n_std: float = 2.0):
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    upper = ma + n_std * sd
    lower = ma - n_std * sd
    width = (upper - lower) / ma
    pctb = (close - lower) / (upper - lower)
    return ma, upper, lower, width, pctb


def stochastic(high, low, close, k: int = 14, d: int = 3):
    lowest = low.rolling(k).min()
    highest = high.rolling(k).max()
    percent_k = 100 * (close - lowest) / (highest - lowest)
    percent_d = percent_k.rolling(d).mean()
    return percent_k, percent_d


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff()).fillna(0.0)
    return (sign * volume).cumsum()


def williams_r(high, low, close, period: int = 14) -> pd.Series:
    """Williams %R: momentum oscillator, -100..0 (oversold near -100)."""
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)


def cci(high, low, close, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    tp = (high + low + close) / 3
    ma = tp.rolling(period).mean()
    md = (tp - ma).abs().rolling(period).mean()
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def mfi(high, low, close, volume, period: int = 14) -> pd.Series:
    """Money Flow Index: volume-weighted RSI (0..100)."""
    tp = (high + low + close) / 3
    rmf = tp * volume
    pos = rmf.where(tp > tp.shift(1), 0.0)
    neg = rmf.where(tp < tp.shift(1), 0.0)
    pos_s = pos.rolling(period).sum()
    neg_s = neg.rolling(period).sum().replace(0, np.nan)
    mr = pos_s / neg_s
    return 100 - 100 / (1 + mr)


def keltner(high, low, close, period: int = 20, mult: float = 2.0):
    """Keltner Channels (EMA +/- mult*ATR). Returns (%position, width)."""
    ma = close.ewm(span=period, adjust=False, min_periods=period).mean()
    rng = atr(high, low, close, period)
    upper = ma + mult * rng
    lower = ma - mult * rng
    pos = (close - lower) / (upper - lower).replace(0, np.nan)
    width = (upper - lower) / ma
    return pos, width


def vwap_dev(high, low, close, volume, period: int = 20) -> pd.Series:
    """Rolling VWAP deviation: how far price is from volume-weighted price."""
    tp = (high + low + close) / 3
    pv = (tp * volume).rolling(period).sum()
    vv = volume.rolling(period).sum().replace(0, np.nan)
    vwap = pv / vv
    return (close - vwap) / vwap


def ichimoku_signals(high, low, close):
    """Ichimoku-derived features: price vs cloud, tenkan/kijun spread.

    Uses only past data (standard backward-looking spans; we do NOT project the
    cloud forward into the future to avoid look-ahead)."""
    conv = (high.rolling(9).max() + low.rolling(9).min()) / 2
    base = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = (conv + base) / 2
    span_b = (high.rolling(52).max() + low.rolling(52).min()) / 2
    tk_spread = (conv - base) / close
    cloud_pos = (close - span_a) / close
    cloud_pos_b = (close - span_b) / close
    return tk_spread, cloud_pos, cloud_pos_b


def adx(high, low, close, period: int = 14) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = ((up > down) & (up > 0)) * up
    minus_dm = ((down > up) & (down > 0)) * down
    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
