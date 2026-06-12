"""Meta-labeling (López de Prado, AFML ch. 3).

Two-stage architecture:
  * PRIMARY model decides DIRECTION (long / short / flat) — high recall.
  * META model decides whether to ACT on the primary signal (bet / no-bet) —
    high precision. It is trained on a binary target: "did the primary signal
    actually make money (after the barrier resolved)?"

Why it works: separating "which way" from "how sure / size it" lets the meta
model suppress the many low-quality signals that cause overtrading. This is the
single most effective, well-documented way to raise precision and slash
turnover — exactly our problem from the sweep.

The meta model's predicted probability also gives us a natural BET SIZE
(Kelly-style / probability-proportional), so we trade big only when the system
is genuinely confident.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb


def make_meta_target(primary_signal: pd.Series, label: pd.Series) -> pd.Series:
    """1 if acting on the primary signal would have been correct, else 0.

    primary_signal in {-1,0,+1}; label in {-1,0,+1} (realized barrier outcome).
    Only rows where primary_signal != 0 are meta-events.
    """
    mask = primary_signal != 0
    correct = (np.sign(primary_signal) == np.sign(label)) & mask
    return correct.astype(int)[mask]


def default_meta_params() -> dict:
    return dict(
        objective="binary",
        learning_rate=0.02,
        num_leaves=31,
        max_depth=4,
        min_child_samples=60,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


class MetaModel:
    """Binary 'should we bet?' model returning P(primary signal is correct)."""

    def __init__(self, params: dict | None = None):
        self.params = params or default_meta_params()
        self.model: lgb.LGBMClassifier | None = None

    def fit(self, X, y, sample_weight=None) -> "MetaModel":
        self.model = lgb.LGBMClassifier(**self.params)
        # guard: if only one class present, fall back to constant predictor
        if len(np.unique(y)) < 2:
            self._const = float(np.mean(y))
            self.model = None
        else:
            self._const = None
            self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_bet_proba(self, X) -> np.ndarray:
        if self.model is None:
            return np.full(len(X), getattr(self, "_const", 0.5))
        return self.model.predict_proba(X)[:, 1]


def bet_size_from_proba(p: np.ndarray, p_threshold: float = 0.55,
                        max_size: float = 1.0) -> np.ndarray:
    """Map meta probability -> position size in [0, max_size].

    Below threshold -> 0 (no bet). Above -> scaled linearly so marginal-edge
    bets are small and high-confidence bets are large. Simple, robust, and
    avoids the instability of raw Kelly on noisy probabilities.
    """
    size = np.clip((p - p_threshold) / (1.0 - p_threshold), 0.0, 1.0)
    return size * max_size
