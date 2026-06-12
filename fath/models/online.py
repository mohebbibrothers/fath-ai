"""Continuous / incremental learning engine.

WHAT "NEVER STOP LEARNING" ACTUALLY MEANS (done right):
Retraining literally on every tick is a mistake — single ticks are pure noise,
and refitting that often guarantees overfitting and absurd compute cost. The
professional, statistically sound version of "always improving" is:

  * an EXPANDING-WINDOW walk-forward that periodically (every `retrain_every`
    bars) refits the model on ALL data available up to that point, then
    predicts the next block it has never seen;
  * online tracking of rolling out-of-sample accuracy, so the system *measures*
    whether it is actually getting better over time (and can alert / adapt if
    it degrades — concept drift detection).

This gives you a model that genuinely adapts to new regimes and whose skill is
continuously, honestly monitored — without fooling itself on noise.

The same class powers both research (replay history) and live/paper operation
(append the latest closed bar, retrain on schedule, emit a position).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fath.models.classifier import BarrierClassifier
from fath.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class OnlineLearner:
    """Expanding-window incremental learner with drift monitoring."""

    warmup: int = 500            # min samples before first fit
    retrain_every: int = 50      # refit cadence (bars)
    min_train: int = 300
    drift_window: int = 200      # rolling window for OOS accuracy tracking

    model: BarrierClassifier | None = None
    _last_fit_at: int = -1
    oos_correct: list[int] = field(default_factory=list)
    rolling_acc_history: list[float] = field(default_factory=list)

    def _should_refit(self, i: int) -> bool:
        if self.model is None:
            return i >= self.warmup
        return (i - self._last_fit_at) >= self.retrain_every

    def replay(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None):
        """Replay a full history as if it arrived bar-by-bar.

        Returns OOS probabilities (canonical [-1,0,1]) for every bar after
        warmup, plus the rolling-accuracy curve so you can SEE the model
        improving / adapting over time.
        """
        n = len(X)
        oos = np.full((n, 3), np.nan)
        preds = np.full(n, 0)

        for i in range(n):
            if self.model is not None and i > self.warmup:
                # predict the current (unseen) bar with the last fitted model
                p = self.model.predict_proba(X.iloc[[i]])
                oos[i] = p[0]
                preds[i] = self.model.predict_signed(X.iloc[[i]])[0]
                if y.iloc[i] != 0:
                    self.oos_correct.append(int(preds[i] == y.iloc[i]))
                    if len(self.oos_correct) >= self.drift_window:
                        acc = float(np.mean(self.oos_correct[-self.drift_window:]))
                        self.rolling_acc_history.append(acc)

            if self._should_refit(i) and i >= self.min_train:
                # refit on ALL data strictly before bar i (no leakage)
                sw = sample_weight.iloc[:i].to_numpy() if sample_weight is not None else None
                self.model = BarrierClassifier().fit(X.iloc[:i], y.iloc[:i], sample_weight=sw)
                self._last_fit_at = i
                log.debug("Refit at bar %d on %d samples", i, i)

        return oos, np.array(self.rolling_acc_history)

    # ---- live operation ----------------------------------------------------
    def update_and_predict(self, X_hist: pd.DataFrame, y_hist: pd.Series,
                           x_new: pd.DataFrame) -> dict:
        """Live step: optionally retrain on full history, then predict next bar."""
        i = len(X_hist)
        if self._should_refit(i):
            self.model = BarrierClassifier().fit(X_hist, y_hist)
            self._last_fit_at = i
            log.info("Live retrain on %d samples", i)
        if self.model is None:
            return {"position": 0.0, "proba": None, "note": "warming up"}
        proba = self.model.predict_proba(x_new)[0]
        edge = float(proba[2] - proba[0])
        return {"proba": proba.tolist(), "edge": edge,
                "signal": int(np.sign(edge)) if abs(edge) > 0.03 else 0}

    def drift_status(self) -> dict:
        """Concept-drift readout: is recent skill rising, flat, or falling?"""
        h = self.rolling_acc_history
        if len(h) < 4:
            return {"status": "insufficient_data"}
        recent = np.mean(h[-len(h) // 4:])
        early = np.mean(h[: len(h) // 4])
        trend = recent - early
        return {
            "rolling_acc_latest": round(h[-1], 4),
            "rolling_acc_mean": round(float(np.mean(h)), 4),
            "trend_recent_vs_early": round(float(trend), 4),
            "status": "improving" if trend > 0.01 else
                      "degrading" if trend < -0.01 else "stable",
        }
