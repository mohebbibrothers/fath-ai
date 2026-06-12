"""Purged & embargoed walk-forward cross-validation.

Standard k-fold CV is INVALID for financial time series because:
  1. It shuffles, destroying temporal order and leaking the future.
  2. Even contiguous folds leak when labels span multiple bars (a train label
     can overlap a test label) -> "purging" removes overlapping train samples.
  3. Serial correlation near the test boundary leaks -> "embargo" drops a small
     buffer of train samples right after the test set.

This implementation (AFML ch. 7) yields expanding/rolling walk-forward folds:
train always precedes test in time, with purge + embargo applied.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def walk_forward_splits(
    n_samples: int,
    n_splits: int = 5,
    test_frac: float = 0.2,
    expanding: bool = True,
):
    """Yield (train_idx, test_idx) integer arrays in time order.

    The data is divided into ``n_splits`` sequential test blocks at the end of
    the series; each fold trains on everything before its test block.
    """
    test_size = max(1, int(n_samples * test_frac / n_splits))
    first_test = n_samples - n_splits * test_size
    if first_test <= 0:
        raise ValueError("Not enough samples for requested splits.")

    for k in range(n_splits):
        test_start = first_test + k * test_size
        test_end = test_start + test_size
        test_idx = np.arange(test_start, min(test_end, n_samples))
        if expanding:
            train_idx = np.arange(0, test_start)
        else:
            train_idx = np.arange(max(0, test_start - first_test), test_start)
        yield train_idx, test_idx


def purge_embargo(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    t1_positions: np.ndarray,
    embargo_frac: float = 0.01,
    n_samples: int | None = None,
) -> np.ndarray:
    """Remove train samples that overlap the test window, plus an embargo.

    Parameters
    ----------
    t1_positions : for each sample, the integer position where its label ends.
    embargo_frac : embargo size as fraction of total samples.
    """
    if n_samples is None:
        n_samples = int(max(train_idx.max(), test_idx.max())) + 1
    test_start, test_end = test_idx.min(), test_idx.max()
    embargo = int(n_samples * embargo_frac)

    keep = []
    for i in train_idx:
        label_end = t1_positions[i] if t1_positions[i] >= 0 else i
        # purge: train label span must not reach into the test block
        if label_end >= test_start:
            continue
        # embargo: drop train samples just after the test block
        if test_end < i <= test_end + embargo:
            continue
        keep.append(i)
    return np.array(keep, dtype=np.int64)
