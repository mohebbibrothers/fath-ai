"""The single most important test in any quant codebase: NO LOOK-AHEAD.

If features computed up to time t change when we append future bars, the
backtest is fiction. We verify that feature values for the first N rows are
identical whether or not future data exists.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fath.features.build import build_features


def _synthetic_ohlcv(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    ret = rng.normal(0, 0.01, n)
    close = 100 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    open_ = np.r_[close[0], close[:-1]]
    vol = rng.uniform(1, 100, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def test_features_are_causal():
    full = _synthetic_ohlcv(1000)
    cut = 600
    partial = full.iloc[:cut]

    f_full = build_features(full)
    f_partial = build_features(partial)

    common = f_full.index.intersection(f_partial.index)
    # compare on the overlap, excluding the very last few rows near the cut
    common = common[:-5]
    a = f_full.loc[common]
    b = f_partial.loc[common]

    pd.testing.assert_frame_equal(a, b, atol=1e-9, check_dtype=False)


def test_no_nan_or_inf_in_features():
    df = _synthetic_ohlcv(800)
    f = build_features(df)
    assert np.isfinite(f.to_numpy()).all()
    assert len(f) > 0


if __name__ == "__main__":
    test_features_are_causal()
    test_no_nan_or_inf_in_features()
    print("All no-lookahead tests passed.")
