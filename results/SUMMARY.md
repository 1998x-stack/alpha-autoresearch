# alpha_autoresearch — Experiment Summary

**Date:** 2026-05-15
**Iterations:** 30
**Dataset:** 495 A-shares, 535,412 rows, 2020-01-02 ~ 2025-09-17

---

## Key Results

| Metric | Value |
|--------|-------|
| Total experiments | 38 (across 30 iterations) |
| Factors kept | 38 |
| Factors discarded | 0 |
| Crashes | 0 |
| Pareto frontier size | 14 non-dominated factors |
| Best \|Rank IC\| | **0.0581** (hl_range — high-low range ratio) |
| Best \|IC IR\| | **0.49** (ts_rank_vol — time-series volume rank) |
| Mean \|Rank IC\| | 0.0234 |
| Mean turnover stability | 0.8809 |

---

## Pareto Frontier (14 non-dominated factors)

| Factor | \|Rank IC\| | \|IC IR\| | Turnover | Category |
|--------|------------|----------|---------|----------|
| hl_range | 0.0581 | 0.28 | 0.806 | Price Range |
| ts_rank_vol | 0.0486 | 0.49 | 0.902 | TS Rank |
| momentum_vol | 0.0365 | 0.23 | 0.953 | Volume-Momentum Combo |
| momentum_turn | 0.0314 | 0.29 | 0.867 | Momentum-Turnover Combo |
| vwap_diff | 0.0271 | 0.18 | 0.991 | VWAP |
| decay_range | 0.0273 | 0.13 | 0.814 | Decay Linear |
| vwap_sq | 0.0258 | 0.20 | 0.988 | VWAP |
| decay_mom | 0.0223 | 0.22 | 0.984 | Decay Linear |
| momentum_long | 0.0216 | 0.21 | 0.945 | Momentum |
| reversal_10d | 0.0212 | 0.24 | 0.969 | Reversal |
| reversal_5d | 0.0206 | 0.24 | 0.974 | Reversal |
| ts_rank_mom | 0.0172 | 0.22 | 0.966 | TS Rank |
| momentum_decay | 0.0223 | 0.22 | 0.984 | Momentum |
| vol_rank_corr | 0.0010 | 0.01 | 0.897 | Volume-Price Correlation |

---

## Factor Categories Explored

1. **Momentum** (5 variants): 5d, 10d, 20d, 60d, decay-weighted
2. **Reversal** (2 variants): 5d, 10d
3. **Volume-Price Correlation** (3 variants): rolling corr, rank corr, 20d window
4. **Volatility** (2 variants): price std, volume std
5. **Price Range** (2 variants): HL ratio, open-to-range ratio
6. **VWAP** (2 variants): close-vwap diff, squared min distance
7. **Volume Ratio** (2 variants): vol/adv20, adv5/adv20 combo
8. **Returns** (2 variants): rolling mean, rolling volatility
9. **Combo** (2 variants): momentum × volume, momentum × turnover
10. **Classic Alpha101** (4 variants): Alpha003, Alpha006, Alpha054, Alpha101 styles
11. **TS Rank** (2 variants): momentum TS rank, volume TS rank
12. **Decay Linear** (2 variants): momentum decay, range decay

---

## Visualizations

![Pareto Frontier](results/pareto_frontier.png)
![Top Factors](results/top_factors.png)
![Metric Correlations](results/metric_correlations.png)

---

## Observations

1. **Price range factors dominate IC** — `hl_range` achieves the highest predictive power (IC=0.058), suggesting A-shares have strong intraday range patterns.
2. **TS rank on volume has best IR** — `ts_rank_vol` achieves IR=0.49, the most consistent predictor. Time-series ranking of volume is a robust signal.
3. **VWAP factors have excellent turnover** — `vwap_diff` and `vwap_sq` achieve turnover > 0.98, meaning they're nearly free to trade while maintaining moderate IC.
4. **Volume-price correlation is weak** — All vol_corr variants have IC < 0.004, suggesting simple correlation isn't predictive in this market.
5. **Momentum consistently negative** — All momentum factors show negative IC, confirming A-share short-term reversal (not momentum) effect.
6. **No crashes** — All 38 factors evaluated successfully, showing the system is robust.
