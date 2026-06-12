"""Triple-barrier labeling (López de Prado, *Advances in Financial ML*, ch. 3).

For each event at time t we set three barriers ahead:
  * upper barrier  = entry * (1 + tp_mult * volatility_t)   -> label +1
  * lower barrier  = entry * (1 - sl_mult * volatility_t)   -> label -1
  * vertical barrier = t + max_horizon bars                 -> label  0 (timeout,
                       or sign of the return at timeout if you prefer)

Whichever barrier is touched FIRST decides the label. This is vastly superior
to naive "will price be up in N bars" labels because it respects path and risk
symmetry, and it directly mirrors how a real trade with TP/SL would resolve.

We also return, for each event:
  * t1            : the index timestamp where the label was determined
  * ret           : realized return from entry to barrier touch
  * horizon_bars  : how many bars it took (for sample-uniqueness weighting)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fath.utils.logging import get_logger

log = get_logger(__name__)


def triple_barrier_labels(
    close: pd.Series,
    volatility: pd.Series,
    tp_mult: float = 1.5,
    sl_mult: float = 1.5,
    max_horizon: int = 24,
    min_ret: float = 0.0,
) -> pd.DataFrame:
    """Compute triple-barrier labels.

    Parameters
    ----------
    close : price series (entry at each bar's close).
    volatility : per-bar volatility estimate (e.g. ATR/price or realized vol),
        aligned to ``close``. Barriers scale with this so they adapt to regime.
    tp_mult, sl_mult : barrier widths in units of volatility.
    max_horizon : vertical barrier in bars.
    min_ret : if |timeout return| < min_ret, label stays 0 (neutral).

    Returns a DataFrame indexed like ``close`` (minus the tail that cannot be
    fully evaluated) with columns: label, ret, t1, horizon_bars.
    """
    close = close.astype(float)
    vol = volatility.reindex(close.index).astype(float)
    idx = close.index
    n = len(close)
    c = close.to_numpy()
    varr = vol.to_numpy()

    labels = np.zeros(n, dtype=np.int8)
    rets = np.full(n, np.nan)
    t1_pos = np.full(n, -1, dtype=np.int64)
    horizon = np.zeros(n, dtype=np.int32)

    for i in range(n):
        if not np.isfinite(varr[i]) or varr[i] <= 0:
            t1_pos[i] = -1
            continue
        end = min(i + max_horizon, n - 1)
        if end <= i:
            t1_pos[i] = -1
            continue
        entry = c[i]
        up = entry * (1 + tp_mult * varr[i])
        dn = entry * (1 - sl_mult * varr[i])

        touched = 0
        hit_pos = end
        for j in range(i + 1, end + 1):
            if c[j] >= up:
                touched = 1
                hit_pos = j
                break
            if c[j] <= dn:
                touched = -1
                hit_pos = j
                break

        if touched == 0:  # vertical barrier (timeout)
            r = c[end] / entry - 1.0
            if abs(r) < min_ret:
                touched = 0
            else:
                touched = int(np.sign(r))
            hit_pos = end

        labels[i] = touched
        rets[i] = c[hit_pos] / entry - 1.0
        t1_pos[i] = hit_pos
        horizon[i] = hit_pos - i

    out = pd.DataFrame(
        {
            "label": labels,
            "ret": rets,
            "t1": [idx[p] if p >= 0 else pd.NaT for p in t1_pos],
            "horizon_bars": horizon,
        },
        index=idx,
    )
    # Drop events that could not be evaluated (no valid vol or no room ahead).
    out = out[[p >= 0 for p in t1_pos]]
    dist = out["label"].value_counts().to_dict()
    log.info("Triple-barrier labels: %s (n=%d)", dist, len(out))
    return out


def average_uniqueness_weights(t1: pd.Series, index: pd.Index) -> pd.Series:
    """Sample weights from concurrency of label spans (AFML ch. 4, simplified).

    Overlapping labels share information; weighting by the inverse average
    concurrency prevents the model from over-counting overlapping windows.
    """
    t1 = t1.dropna()
    if t1.empty:
        return pd.Series(dtype=float)
    # concurrency: number of labels whose [t, t1] span covers each bar
    conc = pd.Series(0, index=index, dtype=float)
    pos = index.get_indexer(t1.index)
    end = index.get_indexer(t1.values)
    for s, e in zip(pos, end):
        if s < 0 or e < 0:
            continue
        conc.iloc[s : e + 1] += 1.0
    conc = conc.replace(0, np.nan)

    weights = {}
    for (t0, t1v), s, e in zip(t1.items(), pos, end):
        if s < 0 or e < 0:
            continue
        weights[t0] = (1.0 / conc.iloc[s : e + 1]).mean()
    w = pd.Series(weights, dtype=float)
    return (w / w.mean()).reindex(t1.index).fillna(1.0)
