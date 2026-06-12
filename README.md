# fath-ai

A **realistic, honest, professional** quantitative trading research framework for crypto markets.

> ⚠️ **Read this first — honesty matters more than hype.**
> No model can predict future prices with "99–100% win rate". Markets are
> stochastic and partly driven by information that does not yet exist. Any
> claim of near-certain prediction is almost always **overfitting, data
> leakage, or an unrealistic backtest**. The goal of this project is *not* a
> magic oracle — it is a **rigorously engineered system that measures whether a
> real statistical edge exists**, and trades it with disciplined risk
> management. If the edge is small or absent, the framework tells you the
> truth instead of a pretty number.

---

## What this project actually does

`fath-ai` is built as a clean, modular quant pipeline following industry and
academic best practices (notably Marcos López de Prado, *Advances in Financial
Machine Learning*).

```
data  ─►  features  ─►  labels  ─►  model  ─►  signal  ─►  backtest  ─►  report
 │           │            │          │          │            │
 live      ~50+         triple    LightGBM   position    realistic
 OHLCV   indicators    barrier   (gradient   sizing &   fills: fees,
 (Kraken  + leak-free  labeling   boosting)   risk mgmt  slippage,
  etc.)    transforms                                     funding
```

### Engineering principles (the parts that make it real)

1. **No look-ahead bias.** Every feature at time *t* uses only data available
   at or before *t*. Verified by explicit tests.
2. **Purged & embargoed walk-forward validation.** The only correct way to
   validate a time-series model. Prevents train/test leakage.
3. **Triple-barrier labeling.** Labels are defined by which of {take-profit,
   stop-loss, time-limit} is hit first — the gold standard in financial ML.
4. **Realistic backtester.** Models taker/maker fees, slippage, and funding.
   A backtest that ignores costs is fiction.
5. **Sample weighting by uniqueness** (overlap-aware) so overlapping labels
   don't inflate confidence.
6. **Honest reporting.** Sharpe, Sortino, max drawdown, hit rate, turnover,
   cost drag, and out-of-sample vs in-sample gap.

---

## Quick start

```bash
# 1. install
pip install -r requirements.txt

# 2. download historical data (Kraken public API, free)
python -m scripts.fetch_data --symbol BTC/USD --timeframe 1h --since 2018-01-01

# 3. run the full research pipeline (features → labels → walk-forward → backtest)
python -m scripts.run_pipeline --config config/default.yaml

# 4. read the honest report
cat artifacts/report.md
```

## Project status

This is a **research framework**, in `backtest / paper-trading` mode by design.
Live trading with real money is intentionally **not** enabled until an edge is
demonstrated out-of-sample, repeatedly, after costs.

## License & disclaimer

For research and educational use. **Not financial advice.** Crypto trading
carries substantial risk of total loss. Past backtest performance does not
guarantee future results. You are solely responsible for any use of this code.
