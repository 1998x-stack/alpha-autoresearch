# Alpha Autoresearch

> Autonomous Alpha101 factor research for Chinese A-shares — AI agent invents, iterates, and optimizes quantitative factors guided by Pareto frontier optimization.

[![Python](https://img.shields.io/badge/python-3.8+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-31%20passed-10B981)]()
[![Factors](https://img.shields.io/badge/factors-38%20evaluated-7C3AED)]()
[![Frontier](https://img.shields.io/badge/frontier-14%20non--dominated-F59E0B)]()

📖 **[中文文档 (Chinese Docs)](docs/README_ZH.md)** | 📊 **[实验报告 (Experiment Report)](docs/REPORT.md)**

---

## Concept

Give an AI agent a factor research setup and let it experiment autonomously. It modifies `factors.py`, runs a standardized evaluation harness, checks 3 first-principles metrics (RankIC, IC IR, Turnover Stability), and maintains a Pareto frontier of non-dominated factors.

**Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch)** — same philosophy, different domain.

```
Agent modifies factors.py → evaluate (3 metrics) → Pareto check → keep/discard → loop
                                                      ↓
                                              pareto_frontier.json
```

---

## Quick Start

```bash
cd alpha_autoresearch
uv sync                          # install deps
uv run python prepare.py         # evaluate factors (sample dataset included)
```

**Requirements:** Python 3.8+, [uv](https://docs.astral.sh/uv/).

> The project includes a sample dataset (50 A-shares, 2020-2025, ~6.7 MB). For the full 495-stock dataset, run `uv run python prepare.py --build-cache` with access to alpha101_factory kline data.

---

## Three Core Metrics (First Principles)

| Metric | Measures | Higher = |
|--------|----------|----------|
| **RankIC** | Cross-sectional predictive power (Spearman) | Stronger signal |
| **IC IR** | Prediction stability (mean/std) | More consistent |
| **Turnover Stability** | Day-over-day ranking stability | Lower trading cost |

These form a Pareto frontier — improve one without degrading others.

---

## Architecture

| File | Role | Modified by |
|------|------|-------------|
| `prepare.py` | Evaluation harness (operators, metrics, Pareto logic) | **Read-only** |
| `factors.py` | Factor definitions (1-10 per experiment) | **AI agent** |
| `program.md` | Agent instructions + iteration principles | **Human** |

---

## Writing a Factor

```python
from prepare import Factor, ops

class Factor001(Factor):
    name = "momentum_5d"

    def compute(self, df):
        m = df.set_index(["datetime", "symbol"])
        val = ops.cs_rank(m["close"] - ops.delay(m["close"], 5))
        return Factor.as_cs_series(df, val)
```

Auto-discovered and evaluated on next `uv run python prepare.py`.

**12 operators available:** `cs_rank`, `cs_zscore`, `ts_rank`, `rolling_corr`, `rolling_cov`, `rolling_std`, `rolling_sum`, `rolling_min`, `rolling_max`, `delta`, `delay`, `decay_linear`

**16 data columns:** `open`, `high`, `low`, `close`, `volume`, `vwap`, `returns`, `adv5`–`adv180`

---

## Experiment Results (30 iterations)

| Metric | Value |
|--------|-------|
| Total experiments | 38 |
| Factors kept | 38 (0 crashes) |
| Pareto frontier | 14 non-dominated factors |
| Best \|RankIC\| | 0.0581 (`hl_range`) |
| Best \|IC IR\| | 0.49 (`ts_rank_vol`) |

See **[REPORT.md](docs/REPORT.md)** for full analysis (Chinese).

---

## Commands

```bash
uv run python prepare.py --build-cache   # build full 495-stock dataset (optional)
uv run python prepare.py                 # evaluate factors.py
uv run pytest tests/ -v                  # run 31 tests
grep "^factor:\|^status:"               # extract results
```

---

## Constraints

| Rule | Detail |
|------|--------|
| Do NOT modify | `prepare.py` |
| Do NOT add packages | Only `pyproject.toml` deps |
| Do NOT commit | `results.tsv` |
| Budget | 10 factors per experiment |
| Timeout | 60s wall-clock safety |
| GPU | Not needed (CPU-only) |

---

## Documentation

| Doc | Language | Content |
|-----|----------|---------|
| [README_ZH.md](docs/README_ZH.md) | 中文 | 完整项目文档 |
| [REPORT.md](docs/REPORT.md) | 中文 | 30 轮实验详细报告 |
| [spec.md](spec.md) | EN | 系统规格 |
| [CONTEXT.md](CONTEXT.md) | EN | 领域术语 |
| [program.md](program.md) | EN | Agent 指令 |

---

## License

MIT
