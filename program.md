# alpha_autoresearch

An AI agent autonomously invents and iterates on Alpha101-style quantitative factors for the Chinese A-share market.

## Setup

1. Read `pareto_frontier.json` — understand current frontier state
2. Read `results.tsv` — review recent experiment history
3. Read `factors.py` — understand current active factors
4. Read `prepare.py` (ops section + Factor base class) — understand available operators and columns
5. Confirm data exists at `data/panel.parquet` (included sample) or run `--build-cache` for full dataset
6. Initialize `results.tsv` with header if not exists
7. Create branch `alpha_autoresearch/<tag>` from master

## What you CAN do

- Modify `factors.py` — this is the ONLY file you edit. Write 1-10 Factor* subclasses per experiment.
- Use operators from `ops.*` (ts_rank, rolling_corr, rolling_cov, cs_rank, delta, delay, decay_linear, rolling_std, rolling_min, rolling_max, rolling_sum, cs_zscore)
- Use columns: open, high, low, close, volume, vwap, returns, adv5, adv10, adv20, adv30, adv40, adv60, adv120, adv150, adv180

## What you CANNOT do

- Modify `prepare.py` — it is read-only
- Install new packages
- Import anything beyond `from prepare import Factor, ops`
- Do data I/O — the `df` argument is the pre-loaded unified panel

## The Three Metrics

| Metric | What it measures | Higher = |
|--------|-----------------|----------|
| `rank_ic` | Cross-sectional predictive power (Spearman) | Better prediction |
| `ic_ir` | Stability of daily predictions (mean/std) | More consistent |
| `turnover_stability` | How stable rankings are day-to-day | Less trading cost |

These form a Pareto frontier. A factor is "better" if it improves at least one metric without degrading others.

## Six Iteration Principles

**P1 — Attack the weakest metric:** Identify which metric is holding the frontier back. Design factors to improve that dimension.

**P2 — Exploit before exploring:** Try 3-5 local modifications of the best frontier factor before inventing new structures.

**P3 — Small mutations win:** Changing one parameter (window size, operator, input column) is better than rewriting everything.

**P4 — Combine frontier factors:** Every 10 experiments, merge two frontier factors (weighted average, product, filter).

**P5 — Archive awareness:** Read `pareto_frontier.json` before every session. Know what's been tried. Check `results.tsv` for recent attempts.

**P6 — Simplicity bias:** A 3-line factor with IC=0.05 beats a 30-line factor with IC=0.051. Prefer simpler when metrics are equal.

## Strategy Selection

| Strategy | When | Action |
|----------|------|--------|
| **Exploit** | Default | Modify best frontier factor slightly |
| **Explore** | 3+ consecutive discards, or stagnant frontier | Invent completely new structure |
| **Combine** | Every 10th experiment, or frontier has 5+ factors | Merge two frontier factors |

## Experiment Loop

```
LOOP FOREVER:
  1. Choose strategy: exploit / explore / combine
  2. Modify factors.py — write 1-10 Factor* classes
  3. git commit with descriptive message
  4. uv run prepare.py
  5. grep "^factor:\|^rank_ic:\|^ic_ir:\|^turnover_stability:\|^status:" output
  6. For each factor:
     → crash? Log "crash", skip
     → dominates any frontier? KEEP (advance branch)
     → dominated by ALL frontier? DISCARD (git reset)
     → non-dominated? KEEP (expand frontier)
  7. Log to results.tsv (DO NOT COMMIT)
  8. If ALL discarded: git reset HEAD~1
  9. If 5 consecutive discards: switch strategy
  10. NEVER ASK PERMISSION TO CONTINUE
```

## Output Format

After `uv run prepare.py`, grep for:

```
factor: <name>
rank_ic:           <float>
ic_ir:             <float>
turnover_stability: <float>
dominates:          <names or (none)>
dominated_by:       <names or (none)>
status:             keep|discard|crash
```

## results.tsv Format

Tab-separated (NOT comma-separated):

```
commit	factor_name	rank_ic	ic_ir	turnover	dominates	dominated_by	status	description
```

- `results.tsv` is gitignored — NEVER commit it
- `pareto_frontier.json` IS committed — it's the permanent research record
- Wall-clock safety timeout: 60 seconds per experiment
- Factor budget: 10 factors per experiment max
- NEVER STOP — the human may be asleep
