"""Probability calibration.

A classifier that outputs 0.7 should be right ~70% of the time when it says 0.7.
Raw gradient-boosting scores are usually NOT calibrated, which makes any
"confidence gate" meaningless. We use isotonic regression (non-parametric,
monotonic) fit on a held-out slice of the training data — never on test data,
to avoid leakage.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    """Calibrates the P(up) - P(down) "edge" -> P(profitable long)."""

    def __init__(self):
        self.iso_up: IsotonicRegression | None = None

    def fit(self, edge: np.ndarray, y_up: np.ndarray) -> "IsotonicCalibrator":
        """edge in [-1,1]; y_up in {0,1} (1 if the up-barrier was hit)."""
        self.iso_up = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.iso_up.fit(edge, y_up)
        return self

    def transform(self, edge: np.ndarray) -> np.ndarray:
        if self.iso_up is None:
            raise RuntimeError("Calibrator not fitted.")
        return self.iso_up.predict(edge)
