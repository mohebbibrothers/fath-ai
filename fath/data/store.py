"""Local parquet cache for OHLCV data.

Keeps a single source of truth on disk so feature building, labeling and
backtesting all read identical, reproducible data. Supports incremental
updates (append only the new tail).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from fath.data.sources import OHLCV_COLS, get_source
from fath.utils.logging import get_logger

log = get_logger(__name__)

DATA_DIR = Path("data")


def _path(symbol: str, timeframe: str, source: str) -> Path:
    safe = symbol.replace("/", "-")
    return DATA_DIR / f"{source}_{safe}_{timeframe}.parquet"


def load(symbol: str, timeframe: str, source: str = "kraken") -> pd.DataFrame:
    """Load cached OHLCV; raises if not present."""
    p = _path(symbol, timeframe, source)
    if not p.exists():
        raise FileNotFoundError(f"No cached data at {p}. Run fetch first.")
    df = pd.read_parquet(p)
    return df[OHLCV_COLS]


def fetch_and_cache(
    symbol: str,
    timeframe: str,
    since: str,
    source: str = "kraken",
    update: bool = True,
) -> pd.DataFrame:
    """Download OHLCV and persist to parquet. Incremental if cache exists."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    p = _path(symbol, timeframe, source)
    src = get_source(source)

    start = pd.Timestamp(since, tz="UTC")
    existing = None
    if update and p.exists():
        existing = pd.read_parquet(p)
        if len(existing):
            start = existing.index.max()  # refetch last bar (may be partial)
            log.info("Incremental update from %s", start)

    fresh = src.fetch_ohlcv(symbol, timeframe, start)
    if existing is not None:
        combined = pd.concat([existing, fresh])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = fresh

    combined.to_parquet(p)
    log.info("Cached %d candles -> %s (%s .. %s)", len(combined), p,
             combined.index.min(), combined.index.max())
    return combined
