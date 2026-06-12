# Honest Findings — fath-ai v0.1

This document records what the framework *actually measured*, with no spin.
Reproducible via `python -m scripts.run_pipeline --config config/default.yaml`.

## Setup
- **Data:** OKX `BTC/USDT`, 1h, 2023-01-01 → 2026-06-12 (~30,200 candles).
- **Features:** 35 leak-free indicators (returns, vol, trend, oscillators,
  volume/liquidity, calendar). No look-ahead — verified by `tests/`.
- **Labels:** triple-barrier (TP/SL = 1.5× vol, 24-bar horizon).
- **Validation:** 5-fold purged + embargoed walk-forward (all metrics OOS).
- **Costs:** 10 bps taker fee/side + 2 bps slippage.

## Results (out-of-sample, after costs)

| Metric | Strategy | Buy & Hold |
|---|---|---|
| Mean OOS classification accuracy | **51.6%** | 50% (chance) |
| Total return % | −98 | −39 |
| Sharpe | ~0.05 | −0.93 |
| Max drawdown % | −98 | −53 |
| # fills | ~3,200 | — |
| **Cost drag %** | **~407** | — |

## What this means (the honest interpretation)

1. **There is a small, real statistical signal.** 51.6% OOS accuracy on a
   ~2-class problem is above chance and consistent across folds. That is *not
   nothing* — but it is *nowhere near* the "99–100%" that's impossible to
   achieve.

2. **The signal does NOT survive trading costs at 1h frequency.** With ~3,200
   fills the strategy paid ~407% in cumulative cost drag. This is the single
   most common way real algo-traders blow up, and the framework surfaces it
   instead of hiding it.

3. **Model "confidence" is not well-calibrated to accuracy.** Filtering for
   high-edge predictions did *not* raise accuracy (it slightly fell). So we
   cannot simply "only take the sure trades" — the sureness is illusory here.

## Honest next steps that could create a *real* edge

These are legitimate research directions — none of them are magic:

- **Lower frequency (4h/1d):** fewer trades → less cost drag. Often the
  difference between a losing and breakeven system.
- **Probability calibration** (isotonic / Platt) + a proper expected-value gate
  vs. cost, so we only trade when `E[move] > round-trip cost`.
- **Better, less-redundant features:** order-flow / microstructure, funding,
  cross-asset (ETH, DXY), on-chain — and aggressive feature selection.
- **Ensemble + regime filters:** only trade in regimes where the edge is
  historically present (e.g. trending vs. chop).
- **Meta-labeling** (López de Prado): a second model that decides whether to
  *act* on the primary signal — proven way to raise precision/lower turnover.

## The bottom line

`fath-ai` is a correct, costs-aware, leakage-free research engine. Right now it
honestly reports that the baseline model has **no tradeable edge after costs**
at 1h. That is the *right* result to get from a v0.1 — it means the measurement
is trustworthy. The work from here is improving the *edge*, with this rigorous
harness making sure we never fool ourselves.
