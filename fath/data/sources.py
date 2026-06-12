"""Market data sources.

Why multiple sources: the primary venue (Binance) is geo-restricted from many
locations, and any single API can rate-limit or change. We default to Kraken
(deep history, generous public API) and keep the interface pluggable.

All fetchers return a *clean*, UTC-indexed OHLCV DataFrame with columns:
    ['open', 'high', 'low', 'close', 'volume']
indexed by a tz-aware ``DatetimeIndex`` named 'timestamp', sorted ascending,
de-duplicated. Anything else is a bug.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import pandas as pd
import requests

from fath.utils.logging import get_logger

log = get_logger(__name__)

OHLCV_COLS = ["open", "high", "low", "close", "volume"]

# Map our canonical timeframe strings to seconds.
TIMEFRAME_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


class DataSource(Protocol):
    """Interface every data source must implement."""

    name: str

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, since: pd.Timestamp
    ) -> pd.DataFrame: ...


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce the canonical OHLCV contract."""
    df = df.copy()
    df = df[OHLCV_COLS].astype(float)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    # basic sanity: high >= low, no negatives
    bad = (df["high"] < df["low"]) | (df[OHLCV_COLS] < 0).any(axis=1)
    if bad.any():
        log.warning("Dropping %d malformed candles", int(bad.sum()))
        df = df[~bad]
    return df


@dataclass
class KrakenSource:
    """Kraken public OHLC endpoint.

    Note: Kraken's OHLC endpoint returns at most ~720 candles per request and
    only supports a ``since`` cursor, so we page forward until we reach 'now'.
    Kraken uses its own asset codes (XBT for BTC); we translate common ones.
    """

    name: str = "kraken"
    base: str = "https://api.kraken.com/0/public"
    _alt = {"BTC": "XBT"}
    _interval_min = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}

    def _pair(self, symbol: str) -> str:
        base, quote = symbol.upper().split("/")
        base = self._alt.get(base, base)
        quote = self._alt.get(quote, quote)
        return f"{base}{quote}"

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: pd.Timestamp) -> pd.DataFrame:
        if timeframe not in self._interval_min:
            raise ValueError(f"Kraken: unsupported timeframe {timeframe}")
        pair = self._pair(symbol)
        interval = self._interval_min[timeframe]
        ts = pd.Timestamp(since)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        since_unix = int(ts.timestamp())
        frames: list[pd.DataFrame] = []
        cursor = since_unix
        step_s = TIMEFRAME_SECONDS[timeframe]
        now = int(time.time())

        while cursor < now:
            params = {"pair": pair, "interval": interval, "since": cursor}
            data = self._get(f"{self.base}/OHLC", params)
            result = data.get("result", {})
            # result holds one key (the resolved pair name) plus 'last'
            key = next((k for k in result if k != "last"), None)
            if key is None:
                break
            rows = result[key]
            if not rows:
                break
            df = pd.DataFrame(
                rows, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"]
            )
            df.index = pd.to_datetime(df["time"].astype(int), unit="s", utc=True)
            frames.append(df[OHLCV_COLS])
            last = int(result["last"])
            if last <= cursor:  # no forward progress -> done
                break
            cursor = last
            # Stop if Kraken returned fewer than a full page near 'now'
            if len(rows) < 2 or cursor >= now - step_s:
                break
            time.sleep(1.2)  # be polite to the public API

        if not frames:
            raise RuntimeError(f"Kraken returned no data for {symbol} {timeframe}")
        return _normalize(pd.concat(frames))

    @staticmethod
    def _get(url: str, params: dict, retries: int = 4) -> dict:
        for attempt in range(retries):
            try:
                r = requests.get(url, params=params, timeout=20)
                r.raise_for_status()
                payload = r.json()
                if payload.get("error"):
                    # Kraken rate-limit errors are transient
                    raise RuntimeError(f"Kraken API error: {payload['error']}")
                return payload
            except Exception as exc:  # noqa: BLE001
                wait = 2 ** attempt
                log.warning("Kraken request failed (%s); retry in %ss", exc, wait)
                time.sleep(wait)
        raise RuntimeError("Kraken request failed after retries")


@dataclass
class OKXSource:
    """OKX history-candles endpoint with deep pagination.

    Unlike Kraken's public OHLC (capped at ~720 bars), OKX's
    ``/market/history-candles`` lets us page backwards in time via the
    ``after`` cursor (a millisecond timestamp; returns bars strictly older).
    We page from now back to ``since``, then sort ascending.

    OKX bar codes: 1m,3m,5m,15m,30m,1H,4H,1D (note the uppercase H/D).
    """

    name: str = "okx"
    base: str = "https://www.okx.com/api/v5"
    _bar = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1H", "4h": "4H", "1d": "1D"}

    def _inst(self, symbol: str) -> str:
        base, quote = symbol.upper().split("/")
        if quote == "USD":  # OKX spot is mostly USDT; map for convenience
            quote = "USDT"
        return f"{base}-{quote}"

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: pd.Timestamp) -> pd.DataFrame:
        if timeframe not in self._bar:
            raise ValueError(f"OKX: unsupported timeframe {timeframe}")
        inst = self._inst(symbol)
        bar = self._bar[timeframe]
        ts = pd.Timestamp(since)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        since_ms = int(ts.timestamp() * 1000)

        frames: list[pd.DataFrame] = []
        cursor = None  # 'after' cursor: fetch bars older than this ms
        while True:
            params = {"instId": inst, "bar": bar, "limit": 100}
            if cursor is not None:
                params["after"] = cursor
            data = self._get(f"{self.base}/market/history-candles", params)
            rows = data.get("data", [])
            if not rows:
                break
            df = pd.DataFrame(
                rows,
                columns=["time", "open", "high", "low", "close",
                         "volume", "volCcy", "volCcyQuote", "confirm"],
            )
            df["time"] = df["time"].astype("int64")
            df.index = pd.to_datetime(df["time"], unit="ms", utc=True)
            frames.append(df[OHLCV_COLS])
            oldest = int(df["time"].min())
            if oldest <= since_ms or len(rows) < 100:
                break
            cursor = str(oldest)  # next page: older than current oldest
            time.sleep(0.25)

        if not frames:
            raise RuntimeError(f"OKX returned no data for {symbol} {timeframe}")
        out = _normalize(pd.concat(frames))
        return out[out.index >= ts]

    @staticmethod
    def _get(url: str, params: dict, retries: int = 4) -> dict:
        for attempt in range(retries):
            try:
                r = requests.get(url, params=params, timeout=20)
                r.raise_for_status()
                payload = r.json()
                if payload.get("code") not in ("0", 0):
                    raise RuntimeError(f"OKX API error: {payload.get('msg')}")
                return payload
            except Exception as exc:  # noqa: BLE001
                wait = 2 ** attempt
                log.warning("OKX request failed (%s); retry in %ss", exc, wait)
                time.sleep(wait)
        raise RuntimeError("OKX request failed after retries")


@dataclass
class OKXSwapSource(OKXSource):
    """OKX PERPETUAL FUTURES (USDT-margined swap) data.

    This is the correct data for a futures bot: it trades the perp instrument
    (e.g. BTC-USDT-SWAP), not spot. Perp prices, volume, funding and the
    exchange's maintenance-margin tiers differ from spot, so backtests for a
    leveraged futures strategy MUST be built on this data to be honest.

    Inherits OKX pagination but targets the *-SWAP instrument family and fetches
    perpetual candles. Funding-rate history and position tiers are exposed via
    dedicated helpers (used by the futures engine for accurate liquidation).
    """

    name: str = "okx_swap"

    def _inst(self, symbol: str) -> str:
        base, quote = symbol.upper().split("/")
        if quote == "USD":
            quote = "USDT"
        return f"{base}-{quote}-SWAP"


def fetch_funding_history(symbol: str, since: pd.Timestamp,
                          base_url: str = "https://www.okx.com/api/v5") -> pd.DataFrame:
    """Realized funding-rate history for a USDT perp. UTC-indexed."""
    base, quote = symbol.upper().split("/")
    if quote == "USD":
        quote = "USDT"
    inst = f"{base}-{quote}-SWAP"
    ts = pd.Timestamp(since)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    since_ms = int(ts.timestamp() * 1000)

    rows = []
    cursor = None
    while True:
        params = {"instId": inst, "limit": 100}
        if cursor is not None:
            params["after"] = cursor
        try:
            r = requests.get(f"{base_url}/public/funding-rate-history",
                             params=params, timeout=20)
            r.raise_for_status()
            data = r.json().get("data", [])
        except Exception as exc:  # noqa: BLE001
            log.warning("Funding history fetch failed: %s", exc)
            break
        if not data:
            break
        rows += data
        oldest = int(data[-1]["fundingTime"])
        if oldest <= since_ms or len(data) < 100:
            break
        cursor = str(oldest)
        time.sleep(0.2)

    if not rows:
        return pd.DataFrame(columns=["funding_rate"])
    df = pd.DataFrame(rows)
    df["funding_rate"] = df["realizedRate"].astype(float)
    df.index = pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True)
    df = df[["funding_rate"]].sort_index()
    return df[~df.index.duplicated(keep="last")]


def fetch_position_tiers(symbol: str,
                         base_url: str = "https://www.okx.com/api/v5") -> pd.DataFrame:
    """Maintenance-margin tiers for a perp (max leverage & MMR per notional)."""
    base, quote = symbol.upper().split("/")
    if quote == "USD":
        quote = "USDT"
    family = f"{base}-{quote}"
    try:
        r = requests.get(f"{base_url}/public/position-tiers",
                         params={"instType": "SWAP", "tdMode": "isolated",
                                 "instFamily": family}, timeout=20)
        r.raise_for_status()
        data = r.json().get("data", [])
    except Exception as exc:  # noqa: BLE001
        log.warning("Position tiers fetch failed: %s", exc)
        return pd.DataFrame()
    df = pd.DataFrame(data)
    for col in ("maxLever", "mmr", "maxSz", "minSz"):
        if col in df:
            df[col] = df[col].astype(float)
    return df


def get_source(name: str) -> DataSource:
    sources = {"kraken": KrakenSource(), "okx": OKXSource(), "okx_swap": OKXSwapSource()}
    if name not in sources:
        raise ValueError(f"Unknown data source '{name}'. Available: {list(sources)}")
    return sources[name]
