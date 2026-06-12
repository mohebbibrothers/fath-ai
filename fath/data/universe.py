"""Trading universe definition & bulk loading.

Instead of training on a single symbol, the professional approach is to POOL
many liquid assets so the model learns patterns that generalize across the
whole market — not curve-fit to one coin's history. This module defines a
liquid universe and provides bulk fetch/load helpers.

Stablecoins and wrapped/pegged assets are excluded (no tradeable trend).
"""
from __future__ import annotations

import pandas as pd

from fath.data import store
from fath.utils.logging import get_logger

log = get_logger(__name__)

# Curated liquid universe (USDT quote). Excludes stables/pegged (USDC, XAUT...).
DEFAULT_UNIVERSE = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT",
    "BNB/USDT", "TRX/USDT", "ADA/USDT", "SUI/USDT", "NEAR/USDT",
    "TON/USDT", "XLM/USDT", "LINK/USDT", "DOT/USDT", "AVAX/USDT",
    "LTC/USDT", "BCH/USDT", "ATOM/USDT", "FIL/USDT", "ETC/USDT",
]

TIMEFRAMES = ["1h", "4h", "1d"]


def fetch_universe(symbols=None, timeframes=None, since="2019-01-01",
                   source="okx_futures") -> dict:
    """Download every (symbol, timeframe) pair. Returns dict of results."""
    symbols = symbols or DEFAULT_UNIVERSE
    timeframes = timeframes or TIMEFRAMES
    out = {}
    for sym in symbols:
        for tf in timeframes:
            try:
                df = store.fetch_and_cache(sym, tf, since, source)
                out[(sym, tf)] = len(df)
                log.info("OK %s %s: %d", sym, tf, len(df))
            except Exception as exc:  # noqa: BLE001
                log.warning("FAILED %s %s: %s", sym, tf, exc)
                out[(sym, tf)] = 0
    return out


def load_universe(symbols=None, timeframe="1d", source="okx_futures",
                  min_rows=400) -> dict:
    """Load all cached symbols for one timeframe. Returns {symbol: ohlcv}."""
    symbols = symbols or DEFAULT_UNIVERSE
    data = {}
    for sym in symbols:
        try:
            df = store.load(sym, timeframe, source)
            if len(df) >= min_rows:
                data[sym] = df
        except FileNotFoundError:
            continue
    log.info("Loaded %d symbols @ %s", len(data), timeframe)
    return data
