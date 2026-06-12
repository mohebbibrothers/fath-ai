"""Sentiment & news data sources.

DESIGN NOTE ON LEAKAGE (read this):
News/sentiment is extremely easy to use incorrectly. To predict candle t+1 we
may only use sentiment KNOWN at or before the close of candle t. The
Fear & Greed index is published daily; we therefore SHIFT it by one period
before merging, so each bar sees only the *previous* day's published value.

Two sources:
  * Fear & Greed Index (alternative.me) — free, daily history back to 2018.
    Perfect as a backtestable sentiment feature aligned to past candles.
  * RSS headlines (CoinTelegraph/Decrypt) — for LIVE mode, a lightweight
    headline-count / keyword-impact signal. Historical RSS is not reliably
    available for free, so for backtests we rely on Fear & Greed; for live
    trading we augment with real-time headline flow.

This keeps the backtest honest (no fake historical news) while still wiring in
real-time news for live/paper operation, exactly as requested.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import pandas as pd
import requests

from fath.utils.logging import get_logger

log = get_logger(__name__)


def fetch_fear_greed(limit: int = 0) -> pd.DataFrame:
    """Daily Fear & Greed index. limit=0 -> full history.

    Returns a DataFrame indexed by UTC date with columns:
        fng_value (0-100), fng_class (categorical code).
    """
    url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            data = r.json()["data"]
            break
        except Exception as exc:  # noqa: BLE001
            log.warning("F&G fetch failed (%s); retry", exc)
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError("Fear & Greed fetch failed")

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    df = df.set_index("timestamp").sort_index()
    df["fng_value"] = df["value"].astype(float)
    # ordinal encoding of the class label
    order = {"Extreme Fear": 0, "Fear": 1, "Neutral": 2, "Greed": 3, "Extreme Greed": 4}
    df["fng_class"] = df["value_classification"].map(order).astype(float)
    return df[["fng_value", "fng_class"]]


def merge_sentiment(ohlcv: pd.DataFrame, fng: pd.DataFrame) -> pd.DataFrame:
    """Attach LAGGED sentiment features to OHLCV without look-ahead.

    For each bar we forward-fill the most recent F&G value, then SHIFT by one
    bar so the value used at bar t was published strictly before t.
    """
    s = fng.copy()
    # reindex onto the OHLCV timeline (daily values broadcast to finer bars)
    merged = ohlcv.copy()
    aligned = s.reindex(merged.index.union(s.index)).sort_index().ffill()
    aligned = aligned.reindex(merged.index)
    # critical anti-leak shift
    merged["fng_value"] = aligned["fng_value"].shift(1)
    merged["fng_class"] = aligned["fng_class"].shift(1)
    merged["fng_change"] = merged["fng_value"].diff()
    return merged


# ----------------------------- live RSS news -------------------------------

_BULLISH = re.compile(
    r"\b(surge|rally|soar|bullish|adopt|approval|etf|inflow|record|all[- ]time high|"
    r"partnership|upgrade|halving|institutional)\b", re.I)
_BEARISH = re.compile(
    r"\b(crash|plunge|hack|exploit|ban|lawsuit|sec|bearish|sell[- ]off|liquidat|"
    r"outflow|fraud|collapse|war|sanction|default)\b", re.I)


@dataclass
class NewsFeed:
    """Lightweight real-time headline sentiment for LIVE mode."""

    feeds: list[str] = field(default_factory=lambda: [
        "https://cointelegraph.com/rss",
        "https://decrypt.co/feed",
    ])

    def fetch_headlines(self, max_items: int = 60) -> list[str]:
        heads: list[str] = []
        for url in self.feeds:
            try:
                r = requests.get(url, timeout=15,
                                 headers={"User-Agent": "fath-ai/0.1"})
                r.raise_for_status()
                heads += re.findall(r"<title>(.*?)</title>", r.text, re.S)
            except Exception as exc:  # noqa: BLE001
                log.warning("RSS fetch failed for %s: %s", url, exc)
        # strip CDATA / tags
        clean = [re.sub(r"<.*?>", "", re.sub(r"<!\[CDATA\[|\]\]>", "", h)).strip()
                 for h in heads]
        return [h for h in clean if h][:max_items]

    def headline_sentiment(self) -> dict:
        """Return a simple net sentiment score from current headlines."""
        heads = self.fetch_headlines()
        bull = sum(bool(_BULLISH.search(h)) for h in heads)
        bear = sum(bool(_BEARISH.search(h)) for h in heads)
        n = max(len(heads), 1)
        return {
            "n_headlines": len(heads),
            "bullish": bull,
            "bearish": bear,
            "net_sentiment": (bull - bear) / n,
        }
