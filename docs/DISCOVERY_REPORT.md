# Alpha Autoresearch — Factor Discovery Report

**Date:** 2026-05-15 22:21 CST
**Experiment:** 10 new factors evaluated
**Result:** 10/10 KEPT, 4 dominated existing frontier factors

---

## Executive Summary

This experiment introduced 10 novel factor designs targeting specific gaps in the Pareto frontier. All 10 factors were successfully evaluated (0 crashes), and 4 factors directly dominated existing frontier entries, expanding the frontier into previously uncovered regions of the IC-IR-Turnover tradeoff space.

**Key Breakthroughs:**
- **cs_zscore_vol** (|IC|=0.054, TO=0.922) — Near hl_range predictive power with 11.7pp better turnover
- **adv_term_struct** (|IC|=0.043, TO=0.941) — Volume term structure as a ranked signal, superior to decay_range
- **hl_spread_chg** (|IC|=0.040, TO=0.797) — Range expansion/contraction as a predictive signal
- **open_vwap_dev** (|IC|=0.020, TO=0.994) — Best turnover among all non-trivial factors

---

## Frontier Evolution

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Frontier size | 15 | 23 | +8 unique |
| Total experiments | 43 | 53 | +10 |
| Best |IC| | 0.0581 (hl_range) | 0.0581 (hl_range) | = |
| Best |IR| | 0.49 (ts_rank_vol) | 0.49 (ts_rank_vol) | = |
| Best Turnover | 0.992 (momentum_5d) | 0.994 (open_vwap_dev) | +0.002 |
| Dominated (new) | 29 | 33 | 4 factors dominated by new entries |

---

## The 10 New Factors

### Top Performers (dominated existing frontier)

#### 1. cs_zscore_vol — Cross-Sectional Z-Score Volume
```
|IC| = 0.0537  |  |IR| = 0.276  |  Turnover = 0.922
```
**Formula:** `ops.cs_zscore(m["volume"])`
**Hypothesis:** cs_zscore produces a different distribution than cs_rank, potentially capturing tail behavior that rank misses. Validated — dominated both decay_range entries.
**Category:** Volume normalization | **Status:** ✅ KEEP (dominates decay_range)

#### 2. adv_term_struct — Volume Term Structure
```
|IC| = 0.0426  |  |IR| = 0.216  |  Turnover = 0.941
```
**Formula:** `ops.cs_rank(m["adv5"] / m["adv60"])`
**Hypothesis:** Short-term vs long-term volume divergence captures institutional attention shifts. Strongly validated — near momentum_vol's IC with vastly better turnover.
**Category:** Volume ratio | **Status:** ✅ KEEP (dominates decay_range)

#### 3. hl_spread_chg — High-Low Spread Change
```
|IC| = 0.0399  |  |IR| = 0.213  |  Turnover = 0.797
```
**Formula:** `ops.delta((m["high"] - m["low"]) / m["close"], 5)`
**Hypothesis:** The rate of range expansion/contraction captures volatility regime changes before they fully manifest in price. Validated.
**Category:** Price range dynamics | **Status:** ✅ KEEP (dominates decay_range)

#### 4. open_vwap_dev — Open-VWAP Deviation
```
|IC| = 0.0201  |  |IR| = 0.092  |  Turnover = 0.994
```
**Formula:** `ops.cs_rank(m["open"] - m["vwap"])`
**Hypothesis:** Morning gap from VWAP captures overnight information that hasn't yet been arbitraged. Modest IC but best-in-class turnover — dominates momentum_5d.
**Category:** VWAP | **Status:** ✅ KEEP (dominates momentum_5d)

---

### Solid Performers (non-dominated, no dominations)

#### 5. range_vol_adj — Volume-Adjusted Range
```
|IC| = 0.0414  |  |IR| = 0.210  |  Turnover = 0.785
```
**Formula:** `(m["high"] - m["low"]) / (m["close"] * ops.rolling_std(m["volume"] / m["adv20"], 20) + 0.001)`
**Hypothesis:** Normalizing range by volume variability reduces noise from illiquid stocks. Strong IC but dominated by hl_range in turnover dimension.
**Category:** Price range / Volume | **Status:** ✅ KEEP

#### 6. ts_rank_returns — TS Rank of Returns
```
|IC| = 0.0226  |  |IR| = 0.116  |  Turnover = 0.675
```
**Formula:** `-ops.ts_rank(m["returns"], 5)`
**Hypothesis:** If TS rank on volume works (IR=0.49), TS rank on returns might be predictive. Modest IC but enters frontier as non-dominated.
**Category:** TS Rank | **Status:** ✅ KEEP

#### 7. close_open_decay — Smoothed Intraday Return
```
|IC| = 0.0233  |  |IR| = 0.145  |  Turnover = 0.662
```
**Formula:** `ops.decay_linear((m["close"] - m["open"]) / m["open"], 10)`
**Hypothesis:** Smoothed gap pattern reduces noise vs raw close-open. Non-dominated entry.
**Category:** Price patterns | **Status:** ✅ KEEP

#### 8. combo_range_vwap — Range × VWAP Combo
```
|IC| = 0.0236  |  |IR| = 0.106  |  Turnover = 0.843
```
**Formula:** `ops.cs_rank((m["high"] - m["low"]) / m["close"]) * ops.cs_rank(m["close"] - m["vwap"])`
**Hypothesis:** Combining range and VWAP signals multiplicatively creates a filtering effect. Non-dominated.
**Category:** Combo | **Status:** ✅ KEEP

#### 9. sharpe_ret — Sharpe-Like Return Ratio
```
|IC| = 0.0198  |  |IR| = 0.124  |  Turnover = 0.666
```
**Formula:** `ops.decay_linear(m["returns"], 10) / (ops.rolling_std(m["returns"], 60) + 0.001)`
**Hypothesis:** Risk-adjusted return captures stocks with positive drift relative to their volatility. Non-dominated.
**Category:** Returns | **Status:** ✅ KEEP

#### 10. decay_returns — Decay-Weighted Returns
```
|IC| = 0.0134  |  |IR| = 0.088  |  Turnover = 0.656
```
**Formula:** `ops.decay_linear(m["returns"], 20)`
**Hypothesis:** Decay-weighted raw returns capture multi-timescale drift. Weakest performer but still non-dominated.
**Category:** Returns | **Status:** ✅ KEEP

---

## Gap Analysis — What Was (and Wasn't) Closed

### Gaps Closed ✓
| Gap | Before | After | Filled By |
|-----|--------|-------|-----------|
| High-IC + High-Turnover (>0.04 IC, >0.90 TO) | *None* | 2 factors | cs_zscore_vol, adv_term_struct |
| Best Turnover (non-trivial) | 0.992 (momentum_5d, IC=0.010) | 0.994 (open_vwap_dev, IC=0.020) | open_vwap_dev |
| Turnover >0.92 with |IC|>0.04 | *None* | 2 factors | cs_zscore_vol, adv_term_struct |

### Gaps Remaining ✗
| Gap | Target | Best Current | Needed |
|-----|--------|-------------|--------|
| |IC|>0.06 (predictive power) | 0.0581 (hl_range) | New range-based formula |
| |IR|>0.50 (consistency) | 0.49 (ts_rank_vol) | More stable volume signals |
| Triple-crown (>0.04 IC, >0.30 IR, >0.90 TO) | All ≥ targets | *None* | Combined multi-signal factor |
| Turnover >0.95 with |IC|>0.04 | *None* | Need low-noise high-signal factors |

---

## Methodology Notes

### Strategy Used: **Explore** (new factor structures)
All 10 factors were completely new designs, not modifications of existing frontier factors.

### Design Heuristics Applied:
1. **Operator diversity:** Used cs_zscore (first time), decay_linear on returns (not just close), delta on derived quantities
2. **Column exploration:** Used returns column extensively, adv60 for term structure, open column for gap signals
3. **Volume normalization:** range_vol_adj tests whether volume-adjustment improves range signal quality
4. **Signal combination:** combo_range_vwap tests multiplicative interaction between proven signals
5. **Risk adjustment:** sharpe_ret introduces return/risk ratio concept to factor space

### Key Insight:
> cs_zscore on volume achieves near-hl_range IC with 11.7pp better turnover. This suggests that volume distribution is more predictive when using z-score normalization (parametric) rather than rank normalization (non-parametric), because z-score captures tail behavior that rank collapses.

---

## Recommendations for Next Iteration

1. **Exploit cs_zscore_vol:** Try cs_zscore on other columns (close, returns, high-low, adv ratios)
2. **Exploit adv_term_struct:** Try different term structure pairs (adv10/adv120, adv20/adv180)
3. **Combine cs_zscore_vol + ts_rank_vol:** Multiplicative combination could push IR even higher
4. **Explore cs_zscore + decay_linear:** Z-score normalization followed by decay-linear smoothing
5. **Attack remaining gap:** Design for triple-crown (simultaneously high IC, IR, turnover)

---

## Visualizations

| Chart | Description |
|-------|-------------|
| ![Pareto Frontier](assets/pareto_frontier.png) | 3-panel Pareto frontier scatter (IC vs TO, IR vs TO, IR vs IC) |
| ![Top Factors](assets/top_factors.png) | Ranked bar charts of all 3 metrics across top 20 factors |
| ![Metric Correlations](assets/metric_correlations.png) | Correlation heatmap between the 3 Pareto metrics |
| ![New Factors](assets/new_factors.png) | Side-by-side comparison of the 10 new factors |
| ![History](assets/experiment_history.png) | Experiment-by-experiment metric evolution |

---

*Generated by alpha_autoresearch — Autonomous Factor Discovery System*
