# alpha_autoresearch — Domain Context

## Core Concepts

### Factor (因子)
A cross-sectional ranking signal computed from market data (price, volume, etc.) that predicts future returns. Output is a MultiIndex Series `(datetime, symbol) → value` where higher values indicate stronger buy signals.

### RankIC (Rank Information Coefficient)
The primary predictive power metric. Defined as the daily cross-sectional Spearman correlation between factor values and forward 1-day returns, averaged over the evaluation period. Range [-1, 1], absolute value used for Pareto comparison.

### IC IR (IC Information Ratio)
Stability metric. Defined as `mean(RankIC) / std(RankIC)` over the evaluation period. Higher = more consistent daily predictions. Analogous to a Sharpe ratio for factor signals.

### Turnover Stability
Tradeability metric. Defined as `1 - mean(|rank(factor_t) - rank(factor_{t-1})|)` across all symbols and dates. Range [0, 1]. Higher = less daily ranking change = lower transaction costs.

### Pareto Frontier
The set of non-dominated factors. Factor A dominates Factor B if A ≥ B on ALL three metrics AND A > B on at least one. The frontier is maintained in `pareto_frontier.json` and updated after each experiment.

### Experiment
One cycle of the agent loop: modify `factors.py` (write 1-10 Factor classes) → git commit → `uv run prepare.py` → evaluate metrics → Pareto check → keep or discard → log to `results.tsv`.

### Edit Surface
The single file (`factors.py`) that the agent modifies. Contains Factor subclasses using operators from `ops.*` and columns from the unified dataset.

### Unified Dataset
A fixed panel DataFrame with MultiIndex `(datetime, symbol)` containing all available columns (open, high, low, close, volume, vwap, returns, adv5-adv180) for ~500 A-shares over 2020-2025. Included as `data/panel.parquet` (50-stock sample) or built on demand from klines_daily. Never modified during experiments.

## Operator Glossary (ops.*)

| Operator | Domain meaning |
|----------|---------------|
| `cs_rank` | Cross-sectional percentile rank — how a stock ranks vs peers on a given day |
| `cs_zscore` | Cross-sectional z-score — how many standard deviations from the day's mean |
| `ts_rank` | Time-series percentile rank — how a stock's current value ranks in its own history |
| `rolling_corr` | Rolling Pearson correlation — co-movement between two series |
| `rolling_cov` | Rolling covariance — scaled co-movement (used for factor interaction) |
| `rolling_std` | Rolling standard deviation — volatility over a window |
| `rolling_sum/min/max` | Rolling aggregation over a window |
| `delta` | Difference: current value minus N periods ago (momentum signal) |
| `delay` | Lagged value from N periods ago (reference point) |
| `decay_linear` | Linear decay weighted average — recent values weighted higher |

## Decision Terminology

| Term | Meaning |
|------|---------|
| **KEEP** | Factor is non-dominated (or dominates an existing factor). Advance git branch, update pareto_frontier.json. |
| **DISCARD** | Factor is dominated by ALL frontier factors. Git reset, do not update archive. |
| **CRASH** | Factor computation failed (NaN, exception). Log to results.tsv with zero metrics, skip. |
| **Exploit** | Strategy: modify the best frontier factor (change window, operator, input column) |
| **Explore** | Strategy: invent a completely new factor structure |
| **Combine** | Strategy: merge two frontier factors (weighted average, product, filter) |
