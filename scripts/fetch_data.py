"""CLI: download & cache historical OHLCV.

Example:
    python -m scripts.fetch_data --symbol BTC/USD --timeframe 1h --since 2023-01-01
"""
from __future__ import annotations

import argparse

from fath.data import store
from fath.utils.logging import get_logger

log = get_logger("fetch")


def main():
    ap = argparse.ArgumentParser(description="Fetch & cache OHLCV data")
    ap.add_argument("--symbol", default="BTC/USD")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--since", default="2023-01-01")
    ap.add_argument("--source", default="kraken")
    args = ap.parse_args()

    df = store.fetch_and_cache(args.symbol, args.timeframe, args.since, args.source)
    log.info("OK: %d candles, %s .. %s", len(df), df.index.min(), df.index.max())


if __name__ == "__main__":
    main()
