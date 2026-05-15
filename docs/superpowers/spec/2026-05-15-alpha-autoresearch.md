# alpha_autoresearch — System Specification

**Date:** 2026-05-15
**Status:** Draft — pending implementation
**Context:** Combines Karpathy's autoresearch experiment loop with alpha101_factory's factor ecosystem

---

## 1. Overview

### 1.1 Concept

An AI agent autonomously invents, iterates on, and optimizes Alpha101-style quantitative factors. The agent modifies a single edit surface file (`factors.py`), runs a standardized evaluation harness (`prepare.py`), and is guided by 1-3 first-principles metrics. It maintains a Pareto frontier of non-dominated factors, selecting experiments to push the frontier outward — higher predictive power, better stability, lower turnover.

### 1.2 Design Philosophy

- **Single-file edit surface** (Karpathy-style) — agent only touches `factors.py`
- **Immutable evaluation** — `prepare.py` is read-only, containing the fixed dataset, metric computation, and Pareto logic
- **Pareto optimization** — multi-objective frontier, not single-number optimization
- **Factor-count budget** — 10 factors per experiment (not wall-clock, since compute is trivial)
- **Simplicity bias** — prefer 3-line factors over 30-line factors at equal performance

### 1.3 File Structure

```
alpha_autoresearch/
├── prepare.py               # READ-ONLY — evaluation harness
│   ├── Unified dataset loader
│   ├── Factor auto-discovery + evaluation
│   ├── 3-metric computation (RankIC, IC IR, Turnover)
│   ├── Pareto dominance checker + archive manager
│   └── Constants (factor budget, dataset config)
├── factors.py                # AGENT EDITS — the edit surface
│   ├── Factor base class
│   ├── Available operators (ops.*)
│   ├── Available columns (open, high, low, close, volume, vwap, returns)
│   └── Agent writes Factor001..Factor010 classes here
├── program.md                # HUMAN EDITS — agent instructions
│   ├── Setup protocol
│   ├── Experiment loop
│   ├── 6 iteration principles
│   ├── Pareto acceptance criteria
│   └── Output format + results.tsv spec
├── pyproject.toml             # Dependencies (pandas, numpy, pyarrow, loguru)
├── pareto_frontier.json       # Current non-dominated factors (auto-managed)
└── results.tsv                # Experiment log (gitignored)
```

---

## 2. Unified Dataset

### 2.1 Design

A fixed, standardized panel dataset that is identical across all experiments. Changes to the dataset would invalidate cross-experiment comparisons, so it is defined as a constant in `prepare.py`.

### 2.2 Specifications

| Property | Value |
|----------|-------|
| **Universe** | 496 A-shares (SSE + SZSE) |
| **Underlying data** | Daily kline from alpha101_factory parquet cache |
| **Columns** | open, high, low, close, volume, vwap, returns, adv5, adv10, adv20, adv30, adv40, adv60, adv120, adv150, adv180 |
| **Temporal coverage** | 2020-01-02 ~ 2025-09-17 (full available range — no split) |
| **Format** | pandas DataFrame with MultiIndex (datetime, symbol) |
| **Storage** | Pre-computed parquet in `~/.cache/alpha_autoresearch/panel.parquet` |
| **Source** | Links to alpha101_factory `data/` directory — no duplicate storage |

### 2.3 Rationale

- **Fixed split:** Same as alpha101_factory's existing data range. Train/val/test prevents overfitting to a single period.
- **Pre-computed intermediates:** vwap, returns, and advN are pre-computed to avoid repeated calculation across experiments.
- **496 stocks:** Sufficient for statistically meaningful IC/RankIC estimation (cross-sectional N is large enough).
- **No data fetching during experiment:** The evaluation harness reads from cache only — no network calls.

---

## 3. Core Metrics (First-Principles)

### 3.1 Why These Three

From first principles, a factor must:
1. **Predict returns** — otherwise it's useless (captured by IC)
2. **Predict consistently** — unstable predictions are noise (captured by IC IR)
3. **Be tradeable** — high turnover destroys alpha via transaction costs (captured by turnover)

These three form a natural trade-off: you can't maximize all simultaneously. Pushing the Pareto frontier means finding factors that improve one dimension without sacrificing others.

### 3.2 Metric Definitions

#### Metric 1: RankIC (帽IC堆叠)

```
RankIC(t) = SpearmanCorr(factor_values(t), forward_returns(t, horizon=1))
           computed cross-sectionally (within each date)
RankIC = mean(RankIC(t)) over all dates t in evaluation period
```

- **Range:** [-1, 1], higher absolute value = stronger predictive power
- **Sign convention:** The agent can produce + or - signals; the system uses absolute IC for Pareto comparison
- **Computation:** Uses `rankdata` on factor values within each date, then Spearman correlation with forward 1-day returns

#### Metric 2: IC Information Ratio (IC信息比率)

```
IC_IR = mean(RankIC(t)) / std(RankIC(t))
```

- **Range:** [0, ∞), higher = more consistent predictions
- **Interpretation:** Signal-to-noise ratio of the factor's daily predictive power
- **A factor with IC=0.05 but std=0.10 (IR=0.5) is worse than IC=0.04 but std=0.04 (IR=1.0)**

#### Metric 3: Turnover Stability (换手稳定性)

```
Turnover(t) = mean(|rank(factor(t)) - rank(factor(t-1))|) across all symbols
Turnover Stability = 1 - mean(Turnover(t))
```

- **Range:** [0, 1], higher = more stable (lower turnover)
- **Interpretation:** How much the factor's cross-sectional ranking changes day-to-day
- **Turnover of 0 = identical rankings every day (no trading needed). Turnover of 0.5 = half the stocks change rank percentile significantly each day.**

### 3.3 Metric Computation Contract

All three metrics are computed by `prepare.py` on the **full period (2020-2025)**. There is no temporal train/val/test split — factor evaluation needs statistical power (more data = more reliable IC estimates), and overfitting is prevented by the multi-objective Pareto framework and simplicity bias, not by data withholding.

```python
# Output format (printed by prepare.py)
---
factor: momentum_v1
rank_ic:           0.0432
ic_ir:             1.24
turnover_stability: 0.78
pareto_dominates: factor_003, factor_012
pareto_dominated_by: (none — non-dominated)
---
```

---

## 4. Time Budget

### 4.1 Design

Unlike autoresearch where GPU training is the bottleneck (~5 min), factor computation is CPU-bound and fast. The bottleneck is **idea quality**, not compute.

**Budget: 10 factors per experiment.**

| Metric | Value |
|--------|-------|
| Budget unit | Factor count (not wall-clock) |
| Per-experiment limit | 10 factors |
| Wall-clock safety timeout | 60 seconds (kill if exceeded — guards against accidentally expensive factors) |
| Typical compute time | ~10 seconds (1 sec/factor + 2 sec eval) |
| Throughput | ~60 experiments/hour, ~500 overnight |
| Per-session (8 hours) | ~480 experiments, ~4800 factor candidates |

### 4.2 Rationale

- **Wall-clock budget is wrong** for this domain — compute is trivially fast, so a 5-min wall clock budget would waste 4:50 of idle time
- **Factor-count budget** directly constrains the agent's idea output, which IS the bottleneck
- **Room for variation:** 10 factors per experiment lets the agent try "5 variants of the same idea with different rolling windows" — productive local search
- **If the agent writes only 1 factor:** experiment takes ~3 seconds — that's fine, the budget is an upper limit not a requirement

### 4.3 Comparison to autoresearch

| Property | autoresearch (LLM) | alpha_autoresearch |
|----------|-------------------|-------------------|
| Budget type | Wall-clock (300s training) | Factor count (10 factors) |
| Bottleneck | GPU compute | Idea quality |
| Exp/hour | ~12 | ~60 |
| Overnight output | ~100 experiments | ~500 experiments |
| Success criterion | Single number (val_bpb) | Pareto frontier (3 metrics) |

---

## 5. factors.py — Agent Edit Surface

### 5.1 Design

A single Python file that the agent modifies. It imports from `prepare.py` to access the Factor base class and operator library. The agent writes `FactorNNN` classes — each represents one factor candidate. All classes in the file are auto-discovered and evaluated.

### 5.2 Template

```python
# factors.py — Agent edit surface for alpha_autoresearch
# Each Factor* class is auto-discovered and evaluated.
# Available operators: ops.ts_rank, ops.rolling_corr, ops.rolling_cov,
#   ops.cs_rank, ops.delta, ops.delay, ops.decay_linear, ops.rolling_std,
#   ops.rolling_min, ops.rolling_max, ops.rolling_sum, ops.cs_zscore
# Available columns: open, high, low, close, volume, vwap, returns,
#   adv5, adv10, adv20, adv30, adv40, adv60, adv120, adv150, adv180

from prepare import Factor, ops

class Factor001(Factor):
    """[Agent writes a descriptive name here]"""
    name = "momentum_v1"

    def compute(self, df):
        m = df.set_index(["datetime", "symbol"])
        val = ops.cs_rank(m["close"] - ops.delay(m["close"], 5))
        return Factor.as_cs_series(df, val)

# Agent can define Factor002..Factor010 here
```

### 5.3 Constraints

- **Must subclass `Factor`** from prepare.py
- **Must define `name`** (unique string identifier)
- **Must implement `compute(self, df)`** returning a MultiIndex Series
- **No external imports** — only ops.* and standard pandas/numpy
- **No data I/O** — the df argument is the pre-loaded unified panel
- **No package installation** — only what's in pyproject.toml

### 5.4 Available Operators (ops.*)

| Operator | Signature | Description |
|----------|-----------|-------------|
| `ops.cs_rank(s)` | → Series | Cross-sectional percentile rank |
| `ops.cs_zscore(s)` | → Series | Cross-sectional z-score |
| `ops.ts_rank(s, n)` | → Series | Time-series percentile rank (per symbol) |
| `ops.rolling_corr(s1, s2, n)` | → Series | Rolling correlation |
| `ops.rolling_cov(s1, s2, n)` | → Series | Rolling covariance (ddof=0) |
| `ops.rolling_std(s, n)` | → Series | Rolling standard deviation (ddof=0) |
| `ops.rolling_sum(s, n)` | → Series | Rolling sum |
| `ops.rolling_min(s, n)` | → Series | Rolling minimum |
| `ops.rolling_max(s, n)` | → Series | Rolling maximum |
| `ops.delta(s, n=1)` | → Series | Difference: s[t] - s[t-n] |
| `ops.delay(s, n=1)` | → Series | Lag by n periods |
| `ops.decay_linear(s, n)` | → Series | Linear decay weighted average |

---

## 6. prepare.py — Immutable Evaluation Harness

### 6.1 Responsibilities

1. **Load unified dataset** from cache
2. **Auto-discover factors** from `factors.py` (scan for `Factor*` classes)
3. **Evaluate each factor** on the test period — compute all 3 metrics
4. **Pareto dominance check** — compare against `pareto_frontier.json`
5. **Update archive** — if any new factors are non-dominated, add them
6. **Print results** in machine-parseable format for agent to grep

### 6.2 Pareto Dominance Logic

Factor A **dominates** Factor B if A is ≥ B on ALL three metrics AND A > B on at least one.

```
A dominates B ⟺
  A.ic >= B.ic AND A.ir >= B.ir AND A.turnover >= B.turnover
  AND (A.ic > B.ic OR A.ir > B.ir OR A.turnover > B.turnover)
```

**Outcomes:**

| Condition | Action |
|-----------|--------|
| Factor dominates any frontier factor | KEEP — add to frontier, remove dominated ones |
| Factor is dominated by ALL frontier factors | DISCARD |
| Factor is non-dominated but doesn't dominate any existing | KEEP — expand frontier |
| Factor crashes (NaN, exception) | CRASH — log, skip |

### 6.3 Output Format

```
---
factor: momentum_v1
rank_ic:      0.0432
ic_ir:              1.24
turnover_stability: 0.78
dominates:          (none)
status:             keep
---
factor: momentum_v2
rank_ic:      0.0381
ic_ir:              0.92
turnover_stability: 0.83
dominates:          (none)
dominated_by:       momentum_v1
status:             discard
---
```

Agent greps for `^factor:|^rank_ic:|^ic_ir:|^turnover_stability:|^status:` to extract results.

---

## 7. program.md — Agent Instructions

### 7.1 Setup Protocol

```
1. Read pareto_frontier.json — understand current frontier state
2. Read results.tsv — review recent experiment history
3. Read factors.py — understand current active factors
4. Read prepare.py — understand available operators and columns
5. Confirm data exists at ~/.cache/alpha_autoresearch/panel.parquet
6. Initialize results.tsv with header if not exists
7. Create branch alpha_autoresearch/<tag> from master
```

### 7.2 Experiment Loop

```
LOOP FOREVER:
  1. Choose strategy: exploit / explore / combine
  2. Modify factors.py — write 1-10 Factor* classes
  3. git commit with descriptive message
  4. uv run prepare.py
  5. grep results for each factor
  6. For each factor:
     → crash? Log "crash", skip
     → dominates any frontier? KEEP, update frontier
     → dominated by all? DISCARD
     → non-dominated? KEEP (expand frontier)
  7. Log to results.tsv (DO NOT COMMIT)
  8. If kept: advance git branch
  9. If ALL discarded: git reset HEAD~1
  10. If 5 consecutive discards: switch strategy
  11. NEVER ASK PERMISSION TO CONTINUE
```

### 7.3 Six Iteration Principles

**P1 — Attack the weakest metric:** Identify which of the 3 metrics is holding the frontier back. Design factors specifically to improve that dimension while not degrading others.

**P2 — Exploit before exploring:** Before inventing something completely new, try 3-5 local modifications of the best frontier factor. Different rolling windows, different operator combinations, different column inputs.

**P3 — Small mutations win:** A one-parameter change that improves one metric by 5% is better than a complete rewrite. Working structure is valuable — don't throw it away.

**P4 — Combine frontier factors:** Every 10 experiments, try combining two non-dominated factors. Weighted average, product, or use one as a filter for the other. Cross-pollination discovers novel factor structures.

**P5 — Archive awareness:** Read `pareto_frontier.json` before every session. Know exactly what's on the frontier and what's been tried. Don't rediscover known territory. Check `results.tsv` for recent attempts.

**P6 — Simplicity bias:** A 3-line factor with IC=0.05 beats a 30-line factor with IC=0.051. Complexity is a hidden cost — harder to understand, more fragile, more likely to overfit. When equal on metrics, prefer simpler.

### 7.4 Strategy Selection Guide

The agent should choose from three strategies each experiment:

| Strategy | When to use | Action |
|----------|-------------|--------|
| **Exploit** | Default. Always start here. | Take the best frontier factor, modify one aspect (window size, operator, input column) |
| **Explore** | After 3+ consecutive discards, or frontier is stagnant | Invent a completely new factor structure from first principles |
| **Combine** | Every 10th experiment, or when frontier has 5+ factors | Merge two frontier factors into one new factor |

### 7.5 results.tsv Format

Tab-separated (NOT comma-separated):

```
commit	factor_name	rank_ic	ic_ir	turnover	dominates	dominated_by	status	description
a1b2c3d	momentum_v1	0.0432	1.24	0.78	—	—	keep	basic 5d momentum
b2c3d4e	momentum_v2	0.0381	0.92	0.83	—	momentum_v1	discard	longer window, worse IC
c3d4e5f	volume_break	0.0000	0.00	0.00	—	—	crash	division by zero
```

- `results.tsv` is **gitignored** — never commit it
- `pareto_frontier.json` is **committed** — this is the permanent record of progress

---

## 8. Pareto Frontier Archive

### 8.1 Format (pareto_frontier.json)

```json
{
  "metrics": ["rank_ic", "ic_ir", "turnover_stability"],
  "frontier": [
    {
      "name": "momentum_v1",
      "rank_ic": 0.0432,
      "ic_ir": 1.24,
      "turnover_stability": 0.78,
      "description": "5-day price momentum, cs_rank(close - delay(close, 5))",
      "commit": "a1b2c3d",
      "added": "2026-05-15T10:30:00",
      "formula": "ops.cs_rank(close - ops.delay(close, 5))"
    }
  ],
  "dominated_count": 47,
  "total_experiments": 72
}
```

### 8.2 Archive Management

- `prepare.py` auto-updates the archive after each experiment
- Dominated factors are removed when a new factor dominates them
- The archive is a git-tracked file — it's the permanent output of the research
- The agent reads it at the start of each session to understand current state

---

## 9. Implementation Notes

### 9.1 Dependencies (pyproject.toml)

```toml
[project]
name = "alpha_autoresearch"
version = "0.1.0"
requires-python = ">=3.8"
dependencies = [
    "pandas>=1.3.0",
    "numpy>=1.21.0",
    "pyarrow>=10.0.0",
    "loguru>=0.7.0",
]
```

### 9.2 Commands

```bash
cd alpha_autoresearch
uv sync                                    # install deps
uv run prepare.py --build-cache            # one-time: build unified dataset
uv run prepare.py                          # evaluate current factors.py
grep "^factor:\|^status:"                  # extract results
```

### 9.3 Integration with alpha101_factory

- The unified dataset is built FROM alpha101_factory's parquet cache
- `prepare.py` reads from `alpha101_factory/data/klines_daily/` and `alpha101_factory/data/tmp_features/`
- The operator library (`ops.*`) in prepare.py mirrors the one in alpha101_factory but is self-contained (no import dependency)
- This ensures alpha_autoresearch is a standalone project while reusing existing data infrastructure

### 9.4 Security & Constraints

| Constraint | Detail |
|------------|--------|
| Do NOT modify | `prepare.py` |
| Do NOT add packages | Only pandas, numpy, pyarrow, loguru |
| Do NOT commit | `results.tsv` |
| Do NOT change | Metric computation functions (the fixed evaluation) |
| Do NOT stop | Agent runs indefinitely until interrupted |
| Time limit | 10 factors per experiment max |
| VRAM | Not applicable (CPU-only computation) |
| Simplicity | Prefer simpler factors at equal metrics |

---

## 10. Success Criteria

### 10.1 System Working

- [ ] `prepare.py` loads unified dataset without error
- [ ] `prepare.py` auto-discovers Factor* classes from `factors.py`
- [ ] All 3 metrics compute for a valid factor
- [ ] Pareto dominance logic correctly determines keep/discard
- [ ] `pareto_frontier.json` updates correctly after experiments
- [ ] Agent can run the loop: modify → commit → evaluate → log → keep/discard

### 10.2 Research Quality (After 100+ experiments)

- [ ] Pareto frontier has 5+ non-dominated factors
- [ ] At least one factor on the frontier has IC > 0.04 AND IR > 1.0
- [ ] The archive shows progressive frontier expansion (not stagnation)
- [ ] Factor diversity: frontier contains factors from different categories (momentum, mean-reversion, volume-based, etc.)

### 10.3 System Robustness

- [ ] Crash recovery: agent handles factor errors gracefully (no infinite loops)
- [ ] Strategy switching: agent changes approach after consecutive failures
- [ ] Archive integrity: duplicate names rejected, invalid metrics caught
- [ ] Results reproducibility: same factor → same metrics (deterministic evaluation)
