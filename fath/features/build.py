"""Feature engineering pipeline.

Produces a leak-free feature matrix from OHLCV. Features are grouped:

  * Returns / momentum   — log returns over multiple horizons
  * Volatility           — ATR, realized vol, Bollinger width
  * Trend                — EMA distances, ADX, MACD
  * Oscillators          — RSI, Stochastic
  * Volume / liquidity   — OBV slope, volume z-score, dollar volume
  * Calendar             — hour-of-day, day-of-week (cyclical encoding)

All features are constructed so that the value at row t depends only on bars
<= t. We then expose a helper to shift the *target* forward, never the features
backward.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fath.features import indicators as ta
from fath.utils.logging import get_logger

log = get_logger(__name__)


def build_features(df: pd.DataFrame, sentiment: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a feature DataFrame aligned to ``df`` index (no look-ahead).

    If ``sentiment`` is provided (output of data.sentiment.merge_sentiment, with
    already-LAGGED columns fng_value/fng_class/fng_change), those columns are
    appended as features. They are pre-shifted upstream so no leakage occurs.
    """
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    f = pd.DataFrame(index=df.index)

    logc = np.log(c)
    logret = logc.diff()

    # --- Returns / momentum -------------------------------------------------
    for k in (1, 2, 3, 5, 8, 13, 21):
        f[f"ret_{k}"] = logc.diff(k)
    f["mom_z_21"] = (logret.rolling(21).mean() / logret.rolling(21).std(ddof=0))

    # --- Volatility ---------------------------------------------------------
    f["atr_14"] = ta.atr(h, l, c, 14)
    f["atr_pct"] = f["atr_14"] / c
    f["rvol_21"] = logret.rolling(21).std(ddof=0)
    f["rvol_63"] = logret.rolling(63).std(ddof=0)
    f["vol_of_vol"] = f["rvol_21"].rolling(21).std(ddof=0)

    # --- Trend --------------------------------------------------------------
    for span in (8, 21, 55):
        ema = c.ewm(span=span, adjust=False, min_periods=span).mean()
        f[f"ema_dist_{span}"] = (c - ema) / c
    macd_line, macd_sig, macd_hist = ta.macd(c)
    f["macd"] = macd_line / c
    f["macd_hist"] = macd_hist / c
    f["adx_14"] = ta.adx(h, l, c, 14)

    # --- Oscillators --------------------------------------------------------
    f["rsi_14"] = ta.rsi(c, 14)
    f["rsi_7"] = ta.rsi(c, 7)
    k_, d_ = ta.stochastic(h, l, c)
    f["stoch_k"] = k_
    f["stoch_d"] = d_
    _, _, _, bb_width, bb_pctb = ta.bollinger(c)
    f["bb_width"] = bb_width
    f["bb_pctb"] = bb_pctb
    f["williams_r"] = ta.williams_r(h, l, c, 14)
    f["cci_20"] = ta.cci(h, l, c, 20)
    f["mfi_14"] = ta.mfi(h, l, c, v, 14)
    kelt_pos, kelt_w = ta.keltner(h, l, c)
    f["kelt_pos"] = kelt_pos
    f["kelt_width"] = kelt_w
    f["vwap_dev"] = ta.vwap_dev(h, l, c, v, 20)
    tk_spread, cloud_a, cloud_b = ta.ichimoku_signals(h, l, c)
    f["ich_tk_spread"] = tk_spread
    f["ich_cloud_a"] = cloud_a
    f["ich_cloud_b"] = cloud_b

    # --- Volume / liquidity -------------------------------------------------
    obv = ta.obv(c, v)
    f["obv_slope_8"] = obv.diff(8) / v.rolling(8).mean().replace(0, np.nan)
    vmean = v.rolling(21).mean()
    vstd = v.rolling(21).std(ddof=0)
    f["vol_z_21"] = (v - vmean) / vstd.replace(0, np.nan)
    f["dollar_vol_z"] = (
        ((c * v) - (c * v).rolling(21).mean()) / (c * v).rolling(21).std(ddof=0)
    )
    # candle shape (intrabar info known at bar close)
    rng = (h - l).replace(0, np.nan)
    f["body_frac"] = (c - o) / rng
    f["upper_wick"] = (h - np.maximum(o, c)) / rng
    f["lower_wick"] = (np.minimum(o, c) - l) / rng

    # --- Calendar (cyclical) ------------------------------------------------
    idx = df.index
    hour = idx.hour + idx.minute / 60.0
    f["hod_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hod_cos"] = np.cos(2 * np.pi * hour / 24)
    dow = idx.dayofweek
    f["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # --- Sentiment / news (already lagged upstream, no look-ahead) ----------
    if sentiment is not None:
        for col in ("fng_value", "fng_class", "fng_change"):
            if col in sentiment.columns:
                f[col] = sentiment[col].reindex(f.index)
        if "fng_value" in f:
            # normalized 0-1 and z-scored over a trailing window
            f["fng_norm"] = f["fng_value"] / 100.0
            f["fng_z_30"] = (
                (f["fng_value"] - f["fng_value"].rolling(30).mean())
                / f["fng_value"].rolling(30).std(ddof=0)
            )

    n_before = len(f)
    f = f.replace([np.inf, -np.inf], np.nan).dropna()
    log.info("Built %d features; %d rows after warmup drop (from %d)",
             f.shape[1], len(f), n_before)
    return f


def feature_columns(f: pd.DataFrame) -> list[str]:
    return list(f.columns)
