"""Gradient-boosted classifier wrapper (LightGBM).

We frame the problem as 3-class classification over triple-barrier outcomes:
  -1 (down barrier first), 0 (timeout/neutral), +1 (up barrier first).

Why LightGBM: for tabular, noisy financial features it is consistently among
the strongest, fastest models, handles missing values, and gives feature
importances we can audit. Deep nets rarely beat well-regularized GBMs on this
kind of low-signal tabular data, and they overfit far more easily.

The model outputs calibrated-ish class probabilities; the trading policy
(separate module) decides how to act on them given costs and risk.
"""
from __future__ import annotations

import numpy as np
import lightgbm as lgb

from fath.utils.logging import get_logger

log = get_logger(__name__)

# Map labels {-1,0,1} -> {0,1,2} for LightGBM multiclass.
_TO_DENSE = {-1: 0, 0: 1, 1: 2}
_FROM_DENSE = {0: -1, 1: 0, 2: 1}


def default_params() -> dict:
    """Conservative, regularized defaults to fight overfitting.

    Note: we do NOT hardcode num_class/objective here. LightGBM's sklearn API
    infers the number of classes from the labels actually present, and forcing
    num_class=3 when only 2 classes appear silently misaligns probability
    columns. Let the wrapper handle class alignment explicitly instead.
    """
    return dict(
        learning_rate=0.02,
        num_leaves=31,
        max_depth=5,
        min_child_samples=80,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
        n_estimators=600,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


CANONICAL = (-1, 0, 1)  # column order we always expose to the rest of the system


class BarrierClassifier:
    """3-class barrier classifier with robust class-column alignment.

    The pipeline always receives probabilities in the fixed column order
    [-1, 0, +1], even if some class is absent from a given training fold.
    Missing classes get probability 0. This eliminates the silent
    column-misalignment bug that plagues naive multiclass setups.
    """

    def __init__(self, params: dict | None = None):
        self.params = params or default_params()
        self.model: lgb.LGBMClassifier | None = None
        self.feature_names_: list[str] | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y)
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.model.classes_  # e.g. array([-1, 1]) or [-1,0,1]
        self.feature_names_ = list(getattr(X, "columns", []))
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Return probabilities in canonical column order [-1, 0, +1]."""
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        raw = self.model.predict_proba(X)  # columns follow self.classes_
        out = np.zeros((raw.shape[0], len(CANONICAL)))
        for j, cls in enumerate(self.classes_):
            out[:, CANONICAL.index(int(cls))] = raw[:, j]
        return out

    def predict_signed(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        col = proba.argmax(axis=1)
        return np.array([CANONICAL[c] for c in col])

    def feature_importance(self) -> dict:
        if self.model is None:
            return {}
        imp = self.model.feature_importances_
        names = self.feature_names_ or [f"f{i}" for i in range(len(imp))]
        return dict(sorted(zip(names, imp.tolist()), key=lambda kv: -kv[1]))
