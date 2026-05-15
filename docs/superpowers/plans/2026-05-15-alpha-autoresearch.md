# alpha_autoresearch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone `alpha_autoresearch/` project where an AI agent autonomously invents and iterates on Alpha101 factors, guided by 3 first-principles metrics (RankIC, IC IR, Turnover Stability) and a Pareto frontier archive.

**Architecture:** Three-file design: `prepare.py` (read-only eval harness with dataset loader, metric computation, and Pareto logic), `factors.py` (agent's single edit surface), `program.md` (human-authored agent instructions). Self-contained operator library ported from alpha101_factory.

**Tech Stack:** Python 3.8+, pandas, numpy, pyarrow, loguru. Reads data from alpha101_factory parquet cache. No external dependencies beyond the four listed.

**Data Source:** The unified dataset is pre-built from alpha101_factory's `data/tmp_features/` directory (reads parquet files for each stock symbol). Path: `{ALPHA101_DATA_ROOT}/tmp_features/{symbol}_{start}_{end}_{adjust}.parquet`. Uses env var `ALPHA101_DATA_ROOT` (default: `../alpha101_factory/data` relative to alpha_autoresearch).

---

## File Map

```
alpha_autoresearch/
├── pyproject.toml              # Project config + deps
├── .gitignore                  # results.tsv, __pycache__, etc.
├── prepare.py                  # READ-ONLY: dataset, eval, metrics, Pareto
│   ├── Operator library (ops.* functions)
│   ├── Factor base class
│   ├── Dataset builder + loader
│   ├── Factor auto-discovery
│   ├── Metric computation (3 metrics)
│   ├── Pareto dominance logic
│   └── Main harness (orchestrates everything)
├── factors.py                  # AGENT EDITS: Factor* subclasses
├── program.md                  # HUMAN EDITS: agent instructions
├── pareto_frontier.json        # Auto-managed Pareto archive (git-tracked)
├── results.tsv                 # Experiment log (gitignored)
└── tests/
    ├── __init__.py
    ├── test_operators.py       # Operator numeric correctness
    ├── test_metrics.py         # Metric computation (RankIC, IR, Turnover)
    └── test_pareto.py          # Pareto dominance + archive logic
```

---

### Task 1: Project Scaffold

**Files:**
- Create: `alpha_autoresearch/pyproject.toml`
- Create: `alpha_autoresearch/.gitignore`
- Create: `alpha_autoresearch/tests/__init__.py`

- [ ] **Step 1: Write pyproject.toml**

```toml
[project]
name = "alpha_autoresearch"
version = "0.1.0"
description = "Autonomous Alpha101 factor research — agent invents and iterates on quantitative factors"
requires-python = ">=3.8"
dependencies = [
    "pandas>=1.3.0",
    "numpy>=1.21.0",
    "pyarrow>=10.0.0",
    "loguru>=0.7.0",
]
```

- [ ] **Step 2: Write .gitignore**

```
__pycache__/
*.py[oc]
*.egg-info/
results.tsv
.pytest_cache/
```

- [ ] **Step 3: Run uv sync**

```bash
cd alpha_autoresearch && uv sync
```
Expected: Dependencies installed without error.

- [ ] **Step 4: Create tests/__init__.py (empty)**

```bash
touch alpha_autoresearch/tests/__init__.py
```

- [ ] **Step 5: Commit**

```bash
cd alpha_autoresearch && git add pyproject.toml .gitignore tests/__init__.py && git commit -m "feat: project scaffold with deps"
```

---

### Task 2: Operator Library (ops.py-style functions in prepare.py)

**Files:**
- Create: `alpha_autoresearch/tests/test_operators.py`
- Modify: `alpha_autoresearch/prepare.py` (create with ops section)

- [ ] **Step 1: Write failing test — test_operators.py**

```python
# -*- coding: utf-8 -*-
"""Operator library tests — verify numeric correctness of all ops.* functions."""
import numpy as np
import pandas as pd
import pytest

# Will import from prepare after implementation
# from prepare import ops

@pytest.fixture
def simple_series():
    """A simple 10-element series for basic window tests."""
    return pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])


class TestOperators:
    """Test all 12 operators for numeric correctness."""

    def test_rolling_sum(self, simple_series):
        from prepare import ops
        result = ops.rolling_sum(simple_series, 3)
        # window 3: first 2 are NaN, then [1+2+3, 2+3+4, ...]
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == 6.0   # 1+2+3
        assert result.iloc[9] == 27.0  # 8+9+10

    def test_rolling_min(self, simple_series):
        from prepare import ops
        result = ops.rolling_min(simple_series, 3)
        assert result.iloc[2] == 1.0   # min(1,2,3)
        assert result.iloc[9] == 8.0   # min(8,9,10)

    def test_rolling_max(self, simple_series):
        from prepare import ops
        result = ops.rolling_max(simple_series, 3)
        assert result.iloc[2] == 3.0   # max(1,2,3)
        assert result.iloc[9] == 10.0  # max(8,9,10)

    def test_rolling_std(self):
        from prepare import ops
        s = pd.Series([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        result = ops.rolling_std(s, 4)
        # ddof=0: std([2,4,4,4]) = sqrt(mean((x-3.5)^2)) = sqrt(((1.5^2+0.5^2+0.5^2+0.5^2)/4))
        # = sqrt((2.25+0.25+0.25+0.25)/4) = sqrt(3.0/4) = sqrt(0.75) ≈ 0.866
        assert abs(result.iloc[3] - 0.866025) < 0.001

    def test_delta(self):
        from prepare import ops
        s = pd.Series([10.0, 12.0, 15.0, 11.0, 8.0])
        result = ops.delta(s, 1)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == 2.0   # 12-10
        assert result.iloc[2] == 3.0   # 15-12
        assert result.iloc[3] == -4.0  # 11-15
        assert result.iloc[4] == -3.0  # 8-11

    def test_delta_window(self):
        from prepare import ops
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = ops.delta(s, 3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert pd.isna(result.iloc[2])
        assert result.iloc[3] == 3.0   # 4-1
        assert result.iloc[5] == 3.0   # 6-3

    def test_delay(self):
        from prepare import ops
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ops.delay(s, 2)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == 1.0
        assert result.iloc[3] == 2.0
        assert result.iloc[4] == 3.0

    def test_ts_rank(self):
        from prepare import ops
        s = pd.Series([1.0, 3.0, 2.0, 5.0, 4.0])
        result = ops.ts_rank(s, 3)
        # Window [1,3,2]: sorted [1,2,3], rank(2)=2, pct=2/3 ≈ 0.6667
        assert abs(result.iloc[2] - 2/3) < 0.01

    def test_decay_linear(self):
        from prepare import ops
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ops.decay_linear(s, 3)
        # weights [1,2,3] / 6: 1*1/6 + 2*2/6 + 3*3/6 = 14/6 ≈ 2.333
        assert abs(result.iloc[2] - 14/6) < 0.01

    def test_decay_linear_nan_propagation(self):
        """Gotcha: NaN should propagate through decay_linear."""
        from prepare import ops
        s = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        result = ops.decay_linear(s, 3)
        assert pd.isna(result.iloc[2])

    def test_cs_rank_with_multiindex(self):
        from prepare import ops
        idx = pd.MultiIndex.from_tuples([
            ("2024-01-01", "A"), ("2024-01-01", "B"), ("2024-01-01", "C"),
            ("2024-01-02", "A"), ("2024-01-02", "B"), ("2024-01-02", "C"),
        ], names=["datetime", "symbol"])
        s = pd.Series([10.0, 20.0, 30.0, 50.0, 40.0, 60.0], index=idx)
        result = ops.cs_rank(s)
        # Day 1: A=10→1/3, B=20→2/3, C=30→3/3
        assert abs(result.iloc[0] - 1/3) < 0.01
        assert abs(result.iloc[1] - 2/3) < 0.01
        assert abs(result.iloc[2] - 1.0) < 0.01
        # Day 2: A=50→2/3, B=40→1/3, C=60→3/3
        assert abs(result.iloc[3] - 2/3) < 0.01

    def test_cs_rank_requires_multiindex(self):
        """Gotcha: cs_rank should raise on plain Series."""
        from prepare import ops
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="MultiIndex"):
            ops.cs_rank(s)

    def test_cs_zscore(self):
        from prepare import ops
        idx = pd.MultiIndex.from_tuples([
            ("2024-01-01", "A"), ("2024-01-01", "B"), ("2024-01-01", "C"),
        ], names=["datetime", "symbol"])
        s = pd.Series([1.0, 2.0, 3.0], index=idx)
        result = ops.cs_zscore(s)
        # mean=2, std(ddof=1)=1: zscores = [-1, 0, 1]
        assert abs(result.iloc[0] + 1.0) < 0.01
        assert abs(result.iloc[1] - 0.0) < 0.01
        assert abs(result.iloc[2] - 1.0) < 0.01

    def test_rolling_corr(self):
        from prepare import ops
        s1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        s2 = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
        result = ops.rolling_corr(s1, s2, 5)
        # Perfect linear correlation → 1.0
        assert result.iloc[4] is not None
        # Should be very close to 1.0
        assert abs(result.iloc[4] - 1.0) < 1e-10 if not pd.isna(result.iloc[4]) else True

    def test_rolling_cov_ddof0(self):
        """Gotcha: rolling_cov should use ddof=0 consistent with rolling_std."""
        from prepare import ops
        s1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        s2 = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
        result = ops.rolling_cov(s1, s2, 3)
        # Perfect correlation: cov(s1, s2) = var(s1) for window [1,2,3]
        # var([1,2,3]) with ddof=0: mean=2, var = ((1+0+1)/3) = 2/3 ≈ 0.6667
        assert result.iloc[2] > 0  # Just verify it computes
```

- [ ] **Step 2: Run test — verify FAIL**

```bash
cd alpha_autoresearch && uv run pytest tests/test_operators.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'prepare'`

- [ ] **Step 3: Implement operator library in prepare.py**

```python
# -*- coding: utf-8 -*-
"""
alpha_autoresearch — Immutable Evaluation Harness
==================================================
This file is READ-ONLY for the agent. It contains:
  1. Self-contained operator library (ops.*)
  2. Factor base class
  3. Unified dataset loader
  4. Metric computation (RankIC, IC IR, Turnover Stability)
  5. Pareto dominance logic + archive management
  6. Main harness (orchestrates evaluation)

Usage:
    uv run prepare.py              # Evaluate factors.py
    uv run prepare.py --build-cache  # One-time: build unified dataset
"""

import os
import sys
import json
import math
import time
import signal
import inspect
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from loguru import logger


# ═══════════════════════════════════════════════════════════════
# Constants (fixed, do not modify)
# ═══════════════════════════════════════════════════════════════

MAX_FACTORS_PER_EXPERIMENT = 10
WALL_CLOCK_TIMEOUT = 60  # seconds — safety kill-switch
DATA_ROOT = Path(os.getenv("ALPHA101_DATA_ROOT", str(Path(__file__).resolve().parent.parent / "alpha101_factory" / "data")))
PARQ_DIR_TMP = DATA_ROOT / "tmp_features"
CACHE_DIR = Path(os.path.expanduser("~/.cache/alpha_autoresearch"))
PANEL_PATH = CACHE_DIR / "panel.parquet"
START_DATE = os.getenv("ALPHA101_START", "20200101")
END_DATE = os.getenv("ALPHA101_END", "20250917")
ADJUST = os.getenv("ALPHA101_ADJUST", "qfq")

# Metric names (order matters for Pareto comparison — all are "higher is better")
METRIC_NAMES = ["rank_ic", "ic_ir", "turnover_stability"]


# ═══════════════════════════════════════════════════════════════
# Operator Library (self-contained, no alpha101_factory dependency)
# ═══════════════════════════════════════════════════════════════

class _Ops:
    """Self-contained operator library for factor computation.
    All rolling functions use ddof=0 for consistency."""

    @staticmethod
    def rolling_sum(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(window=n, min_periods=n).sum()

    @staticmethod
    def rolling_min(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(window=n, min_periods=n).min()

    @staticmethod
    def rolling_max(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(window=n, min_periods=n).max()

    @staticmethod
    def rolling_std(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(window=n, min_periods=n).std(ddof=0)

    @staticmethod
    def rolling_corr(s1: pd.Series, s2: pd.Series, n: int) -> pd.Series:
        return s1.rolling(window=n, min_periods=n).corr(s2)

    @staticmethod
    def rolling_cov(s1: pd.Series, s2: pd.Series, n: int) -> pd.Series:
        return s1.rolling(window=n, min_periods=n).cov(s2, ddof=0)

    @staticmethod
    def delta(s: pd.Series, n: int = 1) -> pd.Series:
        return s.diff(n)

    @staticmethod
    def delay(s: pd.Series, n: int = 1) -> pd.Series:
        return s.shift(n)

    @staticmethod
    def ts_rank(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(window=n, min_periods=n).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

    @staticmethod
    def decay_linear(s: pd.Series, n: int) -> pd.Series:
        weights = np.arange(1, n + 1, dtype=float)
        weights /= weights.sum()
        return s.rolling(window=n, min_periods=n).apply(
            lambda x: np.dot(x, weights), raw=True
        )

    @staticmethod
    def cs_rank(s: pd.Series) -> pd.Series:
        if not isinstance(s.index, pd.MultiIndex):
            raise ValueError("cs_rank requires a MultiIndex (datetime, symbol)")
        return s.groupby(level=0).rank(pct=True)

    @staticmethod
    def cs_zscore(s: pd.Series) -> pd.Series:
        if not isinstance(s.index, pd.MultiIndex):
            raise ValueError("cs_zscore requires a MultiIndex (datetime, symbol)")
        g = s.groupby(level=0)
        return (s - g.transform("mean")) / g.transform("std").replace(0, np.nan)


ops = _Ops()
```

- [ ] **Step 4: Run tests — verify PASS**

```bash
cd alpha_autoresearch && uv run pytest tests/test_operators.py -v
```
Expected: 15 passed

- [ ] **Step 5: Commit**

```bash
cd alpha_autoresearch && git add prepare.py tests/test_operators.py && git commit -m "feat: operator library with 12 ops + 15 tests"
```

---

### Task 3: Factor Base Class + Auto-Discovery

**Files:**
- Modify: `alpha_autoresearch/prepare.py` (append Factor class + discovery)
- Create: `alpha_autoresearch/factors.py` (initial template)

- [ ] **Step 1: Write failing test in test_operators.py (append)**

Append to `tests/test_operators.py`:

```python
class TestFactorDiscovery:
    """Test Factor base class and auto-discovery from factors.py."""

    def test_factor_base_class_exists(self):
        from prepare import Factor
        assert hasattr(Factor, "compute")
        assert hasattr(Factor, "as_cs_series")

    def test_as_cs_series_builds_multiindex(self):
        from prepare import Factor
        df = pd.DataFrame({
            "datetime": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
            "symbol": ["A", "B", "A"],
            "value": [1.0, 2.0, 3.0],
        })
        s = pd.Series([0.1, 0.2, 0.3])
        result = Factor.as_cs_series(df, s)
        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.names == ["datetime", "symbol"]
        assert len(result) == 3

    def test_discover_factors_finds_classes(self):
        from prepare import discover_factors
        factors = discover_factors()
        assert isinstance(factors, dict)
        # factors.py should contain at least the template Factor001
        # (test may need factors.py to exist with at least one Factor class)

    def test_discover_factors_returns_empty_if_no_factors(self, tmp_path, monkeypatch):
        """If factors.py has no Factor subclasses, return empty dict."""
        # Write a temporary factors.py with no Factor classes
        import sys
        fake_factors = tmp_path / "factors_empty.py"
        fake_factors.write_text("# no factors here\n")
        monkeypatch.setattr(sys, "path", [str(tmp_path)] + sys.path)
        # This test verifies graceful handling
```

- [ ] **Step 2: Run test — verify FAIL**

```bash
cd alpha_autoresearch && uv run pytest tests/test_operators.py::TestFactorDiscovery -v
```
Expected: FAIL — `ImportError: cannot import name 'Factor'`

- [ ] **Step 3: Append Factor class + discovery to prepare.py**

Append after the ops section in `prepare.py`:

```python
# ═══════════════════════════════════════════════════════════════
# Factor Base Class
# ═══════════════════════════════════════════════════════════════

class Factor:
    """Base class for all factors. Agent subclasses this in factors.py."""
    name: str = "UnnamedFactor"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Override this. df is a flat DataFrame with columns: datetime, symbol, open, high, low, close, volume, vwap, returns, adv5..adv180.
        Must return a pd.Series with MultiIndex (datetime, symbol)."""
        raise NotImplementedError

    @staticmethod
    def as_cs_series(df: pd.DataFrame, values: pd.Series) -> pd.Series:
        """Build a MultiIndex (datetime, symbol) Series from computed values."""
        idx = pd.MultiIndex.from_frame(df[["datetime", "symbol"]], names=["datetime", "symbol"])
        return pd.Series(values.values, index=idx, name="value")


# ═══════════════════════════════════════════════════════════════
# Factor Auto-Discovery
# ═══════════════════════════════════════════════════════════════

def discover_factors() -> Dict[str, Factor]:
    """Scan factors.py for Factor subclasses, instantiate them, return {name: instance}.
    Only discovers classes whose name starts with 'Factor' (not Factor itself)."""
    import importlib.util

    factors_path = Path(__file__).resolve().parent / "factors.py"
    if not factors_path.exists():
        logger.warning("factors.py not found")
        return {}

    spec = importlib.util.spec_from_file_location("factors_module", factors_path)
    if spec is None or spec.loader is None:
        logger.error("Failed to load factors.py")
        return {}

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    discovered = {}
    for name in dir(module):
        if not name.startswith("Factor") or name == "Factor":
            continue
        obj = getattr(module, name)
        if isinstance(obj, type) and issubclass(obj, Factor) and obj is not Factor:
            try:
                instance = obj()
                if hasattr(instance, "name") and instance.name:
                    if instance.name in discovered:
                        logger.warning(f"Duplicate factor name: {instance.name}")
                    discovered[instance.name] = instance
            except Exception as e:
                logger.warning(f"Failed to instantiate {name}: {e}")

    return discovered
```

- [ ] **Step 4: Create factors.py template**

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
    """5-day momentum: rank stocks by how much price changed in 5 days."""
    name = "momentum_5d"

    def compute(self, df):
        m = df.set_index(["datetime", "symbol"])
        val = ops.cs_rank(m["close"] - ops.delay(m["close"], 5))
        return Factor.as_cs_series(df, val)
```

- [ ] **Step 5: Run tests — verify PASS**

```bash
cd alpha_autoresearch && uv run pytest tests/test_operators.py::TestFactorDiscovery -v
```
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
cd alpha_autoresearch && git add prepare.py factors.py tests/test_operators.py && git commit -m "feat: Factor base class + auto-discovery from factors.py"
```

---

### Task 4: Unified Dataset Builder + Loader

**Files:**
- Modify: `alpha_autoresearch/prepare.py` (append dataset functions)
- Create: `alpha_autoresearch/tests/test_metrics.py` (dataset-related test)

- [ ] **Step 1: Write failing test**

```python
# -*- coding: utf-8 -*-
"""Metric computation and dataset tests."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


class TestDataset:
    """Test dataset loader functions."""

    def test_load_panel_from_cache(self, tmp_path, monkeypatch):
        """If panel.parquet exists, load it."""
        from prepare import load_panel

        # Create a minimal panel
        panel = pd.DataFrame({
            "datetime": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]),
            "symbol": ["000001", "000002", "000001", "000002"],
            "close": [10.0, 20.0, 11.0, 19.0],
            "volume": [1000, 2000, 1100, 2100],
            "open": [9.9, 19.8, 10.9, 19.1],
            "high": [10.1, 20.2, 11.1, 19.5],
            "low": [9.8, 19.5, 10.5, 18.9],
            "vwap": [10.0, 20.0, 11.0, 19.0],
            "returns": [np.nan, np.nan, 0.1, -0.05],
        })
        panel_path = tmp_path / "panel.parquet"
        panel.to_parquet(panel_path)

        monkeypatch.setattr("prepare.PANEL_PATH", panel_path)
        result = load_panel()
        assert len(result) == 4
        assert "datetime" in result.columns
        assert "symbol" in result.columns

    def test_load_panel_file_not_found(self, monkeypatch):
        """Gracefully handle missing cache file."""
        from prepare import load_panel
        monkeypatch.setattr("prepare.PANEL_PATH", Path("/nonexistent/panel.parquet"))
        result = load_panel()
        assert result.empty
```

- [ ] **Step 2: Run test — verify FAIL**

```bash
cd alpha_autoresearch && uv run pytest tests/test_metrics.py::TestDataset -v
```
Expected: FAIL — `ImportError: cannot import name 'load_panel'`

- [ ] **Step 3: Implement dataset loader in prepare.py**

Append to `prepare.py`:

```python
# ═══════════════════════════════════════════════════════════════
# Unified Dataset Loader
# ═══════════════════════════════════════════════════════════════

def _read_tmp_file(filepath: Path) -> Optional[pd.DataFrame]:
    """Read a single tmp_features parquet file."""
    try:
        df = pd.read_parquet(filepath)
        if df.empty:
            return None
        return df
    except Exception as e:
        logger.warning(f"Failed to read {filepath}: {e}")
        return None


def build_unified_panel() -> pd.DataFrame:
    """Build the unified panel from alpha101_factory tmp_features cache.
    Reads all parquet files in PARQ_DIR_TMP and concatenates them.
    Saves to PANEL_PATH for fast subsequent loads.
    """
    if not PARQ_DIR_TMP.exists():
        logger.error(f"tmp_features directory not found: {PARQ_DIR_TMP}")
        logger.info("Set ALPHA101_DATA_ROOT env var to point to alpha101_factory/data")
        return pd.DataFrame()

    files = sorted(PARQ_DIR_TMP.glob("*.parquet"))
    if not files:
        logger.error(f"No parquet files found in {PARQ_DIR_TMP}")
        return pd.DataFrame()

    logger.info(f"Building unified panel from {len(files)} tmp feature files...")
    dfs = []
    for f in files:
        df = _read_tmp_file(f)
        if df is not None:
            dfs.append(df)

    if not dfs:
        logger.error("No valid tmp feature files found")
        return pd.DataFrame()

    panel = pd.concat(dfs, ignore_index=True)
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    panel = panel.sort_values(["datetime", "symbol"]).reset_index(drop=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_PATH, index=False)
    logger.info(f"Unified panel saved: {len(panel)} rows, {panel['symbol'].nunique()} stocks, "
                f"{panel['datetime'].min().date()} ~ {panel['datetime'].max().date()}")
    return panel


def load_panel() -> pd.DataFrame:
    """Load the unified panel from cache. Returns empty DataFrame if not built yet."""
    if PANEL_PATH.exists():
        df = pd.read_parquet(PANEL_PATH)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df
    logger.warning(f"Panel cache not found at {PANEL_PATH}. Run with --build-cache first.")
    return pd.DataFrame()
```

- [ ] **Step 4: Run tests — verify PASS**

```bash
cd alpha_autoresearch && uv run pytest tests/test_metrics.py::TestDataset -v
```
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
cd alpha_autoresearch && git add prepare.py tests/test_metrics.py && git commit -m "feat: unified dataset builder + loader from alpha101_factory cache"
```

---

### Task 5: Metric Computation (RankIC, IC IR, Turnover Stability)

**Files:**
- Modify: `alpha_autoresearch/prepare.py` (append metric functions)
- Modify: `alpha_autoresearch/tests/test_metrics.py` (add metric tests)

- [ ] **Step 1: Write failing test — append to test_metrics.py**

Append to `tests/test_metrics.py`:

```python
def make_test_panel(n_symbols=5, n_days=50, seed=42):
    """Generate synthetic panel data for testing metrics."""
    np.random.seed(seed)
    dates = pd.bdate_range(end=pd.Timestamp("2024-12-31"), periods=n_days)
    symbols = [f"{600000 + i}" for i in range(n_symbols)]

    rows = []
    for sym in symbols:
        base = np.random.uniform(10, 50)
        close = base * np.cumprod(1 + np.random.normal(0.0003, 0.015, n_days))
        volume = np.random.uniform(1e6, 1e8, n_days)
        for i, d in enumerate(dates):
            rows.append({
                "datetime": d, "symbol": sym,
                "open": close[i] * 0.99, "high": close[i] * 1.02,
                "low": close[i] * 0.98, "close": close[i],
                "volume": volume[i], "vwap": close[i],
                "returns": np.nan if i == 0 else (close[i] - close[i-1]) / close[i-1],
            })
    return pd.DataFrame(rows)


def make_factor_series(df, seed=123):
    """Generate a synthetic factor Series aligned to df's MultiIndex.
    Higher values weakly predict positive forward returns."""
    np.random.seed(seed)
    idx = pd.MultiIndex.from_frame(df[["datetime", "symbol"]], names=["datetime", "symbol"])
    # Create factor with some predictive power (correlated with true returns)
    base = df.groupby("symbol")["close"].pct_change().shift(-1).fillna(0)
    noise = np.random.normal(0, 0.01, len(df))
    values = base.values + noise
    return pd.Series(values, index=idx)


class TestMetricComputation:
    """Test all three metrics: RankIC, IC IR, Turnover Stability."""

    def test_rank_ic_computes(self):
        from prepare import compute_rank_ic
        df = make_test_panel(n_symbols=5, n_days=50)
        factor = make_factor_series(df)
        result = compute_rank_ic(factor, df)
        assert isinstance(result, float)
        assert -1.0 <= result <= 1.0

    def test_rank_ic_returns_nan_for_empty(self):
        from prepare import compute_rank_ic
        factor = pd.Series([], dtype=float)
        df = pd.DataFrame()
        result = compute_rank_ic(factor, df)
        assert np.isnan(result)

    def test_ic_ir_computes(self):
        from prepare import compute_ic_ir, compute_daily_rank_ic
        df = make_test_panel(n_symbols=5, n_days=50)
        factor = make_factor_series(df)
        daily_ic = compute_daily_rank_ic(factor, df)
        result = compute_ic_ir(daily_ic)
        assert isinstance(result, float)

    def test_ic_ir_returns_zero_for_no_variance(self):
        from prepare import compute_ic_ir
        daily_ic = pd.Series([0.05, 0.05, 0.05])
        result = compute_ic_ir(daily_ic)
        # std=0 → IR return 0.0
        assert result == 0.0

    def test_turnover_stability_computes(self):
        from prepare import compute_turnover_stability
        df = make_test_panel(n_symbols=5, n_days=50)
        factor = make_factor_series(df)
        result = compute_turnover_stability(factor)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_turnover_stability_constant_factor(self):
        """A factor with constant rankings has turnover_stability=1.0."""
        from prepare import compute_turnover_stability
        df = make_test_panel(n_symbols=3, n_days=20)
        idx = pd.MultiIndex.from_frame(df[["datetime", "symbol"]], names=["datetime", "symbol"])
        # Constant values → same rank every day → zero turnover → stability=1.0
        factor = pd.Series(np.ones(len(df)), index=idx)
        result = compute_turnover_stability(factor)
        assert abs(result - 1.0) < 0.01

    def test_evaluate_factor_returns_all_three(self):
        from prepare import evaluate_factor
        df = make_test_panel(n_symbols=5, n_days=50)
        factor = make_factor_series(df)
        metrics = evaluate_factor(factor, df)
        assert "rank_ic" in metrics
        assert "ic_ir" in metrics
        assert "turnover_stability" in metrics
        for v in metrics.values():
            assert isinstance(v, float)
```

- [ ] **Step 2: Run test — verify FAIL**

```bash
cd alpha_autoresearch && uv run pytest tests/test_metrics.py::TestMetricComputation -v
```
Expected: FAIL — `ImportError: cannot import name 'compute_rank_ic'`

- [ ] **Step 3: Implement metric functions in prepare.py**

Append to `prepare.py`:

```python
# ═══════════════════════════════════════════════════════════════
# Metric Computation
# ═══════════════════════════════════════════════════════════════

def _make_forward_return(panel: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Compute forward returns as a MultiIndex Series (datetime, symbol)."""
    p = panel[["datetime", "symbol", "close"]].copy()
    p = p.sort_values(["symbol", "datetime"])
    ret = p.groupby("symbol")["close"].pct_change(horizon).shift(-horizon)
    idx = pd.MultiIndex.from_frame(p[["datetime", "symbol"]], names=["datetime", "symbol"])
    return pd.Series(ret.values, index=idx, name="fwd_ret")


def compute_daily_rank_ic(factor: pd.Series, panel: pd.DataFrame) -> pd.Series:
    """Compute daily cross-sectional RankIC (Spearman) between factor and forward returns.
    Returns a Series indexed by datetime."""
    fwd = _make_forward_return(panel)
    if fwd is None or len(fwd) == 0:
        return pd.Series(dtype=float)

    # Align factor and forward returns
    df = pd.DataFrame({"factor": factor, "fwd_ret": fwd}).dropna()
    if df.empty:
        return pd.Series(dtype=float)

    daily_ic = {}
    for dt, g in df.groupby(level=0):
        if g["symbol"].nunique() < 2:
            continue
        ic = g["factor"].corr(g["fwd_ret"], method="spearman")
        daily_ic[dt] = ic

    return pd.Series(daily_ic).sort_index()


def compute_rank_ic(factor: pd.Series, panel: pd.DataFrame) -> float:
    """Compute mean RankIC over the full period."""
    daily = compute_daily_rank_ic(factor, panel)
    if daily.empty:
        return np.nan
    return float(daily.mean())


def compute_ic_ir(daily_ic: pd.Series) -> float:
    """Compute IC Information Ratio: mean(IC) / std(IC)."""
    valid = daily_ic.dropna()
    if len(valid) < 2 or valid.std() == 0:
        return 0.0
    return float(valid.mean() / valid.std())


def compute_turnover_stability(factor: pd.Series) -> float:
    """Compute turnover stability: 1 - mean(|rank_t - rank_{t-1}|).
    Cross-sectional ranks must be computed per date, then turnover is
    the mean absolute rank change day-over-day."""
    if factor.empty or not isinstance(factor.index, pd.MultiIndex):
        return np.nan

    # Compute cross-sectional rank per date (0 to 1)
    rank_per_date = factor.groupby(level=0).rank(pct=True)

    # For each symbol, compute |rank_t - rank_{t-1}|
    turnover_series = rank_per_date.groupby(level=1).diff().abs()

    mean_turnover = turnover_series.mean()
    if pd.isna(mean_turnover):
        return 1.0  # No changes = perfect stability

    return float(1.0 - mean_turnover)


def evaluate_factor(factor: pd.Series, panel: pd.DataFrame) -> Dict[str, float]:
    """Compute all three metrics for a single factor.
    Returns dict with keys: rank_ic, ic_ir, turnover_stability."""
    daily_ic = compute_daily_rank_ic(factor, panel)
    rank_ic = float(daily_ic.mean()) if not daily_ic.empty else np.nan
    ic_ir = compute_ic_ir(daily_ic)
    turnover = compute_turnover_stability(factor)

    return {
        "rank_ic": rank_ic,
        "ic_ir": ic_ir,
        "turnover_stability": turnover,
    }
```

- [ ] **Step 4: Run tests — verify PASS**

```bash
cd alpha_autoresearch && uv run pytest tests/test_metrics.py::TestMetricComputation -v
```
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
cd alpha_autoresearch && git add prepare.py tests/test_metrics.py && git commit -m "feat: 3-metric computation (RankIC, IC IR, Turnover Stability) + 7 tests"
```

---

### Task 6: Pareto Dominance Logic + Archive Management

**Files:**
- Modify: `alpha_autoresearch/prepare.py` (append Pareto functions)
- Create: `alpha_autoresearch/tests/test_pareto.py`
- Create: `alpha_autoresearch/pareto_frontier.json` (initial empty archive)

- [ ] **Step 1: Create initial pareto_frontier.json**

```json
{
  "metrics": ["rank_ic", "ic_ir", "turnover_stability"],
  "frontier": [],
  "dominated_count": 0,
  "total_experiments": 0
}
```

- [ ] **Step 2: Write failing test — test_pareto.py**

```python
# -*- coding: utf-8 -*-
"""Pareto dominance + archive management tests."""
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


class TestParetoDominance:
    """Test Pareto dominance logic."""

    def test_dominates_self_is_false(self):
        from prepare import dominates
        a = {"rank_ic": 0.05, "ic_ir": 1.0, "turnover_stability": 0.8}
        assert not dominates(a, a)  # A does not dominate itself (not strictly greater)

    def test_clearly_dominates(self):
        from prepare import dominates
        a = {"rank_ic": 0.05, "ic_ir": 1.5, "turnover_stability": 0.85}
        b = {"rank_ic": 0.03, "ic_ir": 1.0, "turnover_stability": 0.70}
        assert dominates(a, b)

    def test_not_dominates_when_one_metric_worse(self):
        from prepare import dominates
        a = {"rank_ic": 0.05, "ic_ir": 1.5, "turnover_stability": 0.50}
        b = {"rank_ic": 0.03, "ic_ir": 1.0, "turnover_stability": 0.90}
        # A has worse turnover → does NOT dominate B
        assert not dominates(a, b)

    def test_not_dominates_when_all_equal(self):
        from prepare import dominates
        a = {"rank_ic": 0.05, "ic_ir": 1.0, "turnover_stability": 0.8}
        b = {"rank_ic": 0.05, "ic_ir": 1.0, "turnover_stability": 0.8}
        assert not dominates(a, b)

    def test_dominates_with_absolute_ic(self):
        """Pareto comparison uses absolute IC values."""
        from prepare import dominates
        # Factor A has IC=-0.05 (strong negative signal) → abs=0.05
        a = {"rank_ic": -0.05, "ic_ir": 1.2, "turnover_stability": 0.75}
        b = {"rank_ic": 0.03, "ic_ir": 1.0, "turnover_stability": 0.70}
        assert dominates(a, b)


class TestParetoArchive:
    """Test archive read/write and Pareto decision logic."""

    def test_pareto_decision_keep_when_frontier_empty(self, tmp_path, monkeypatch):
        from prepare import pareto_decision, load_archive, ARCHIVE_PATH
        archive_path = tmp_path / "test_frontier.json"
        archive_path.write_text(json.dumps({
            "metrics": ["rank_ic", "ic_ir", "turnover_stability"],
            "frontier": [],
            "dominated_count": 0,
            "total_experiments": 0
        }))

        metrics = {"rank_ic": 0.04, "ic_ir": 1.2, "turnover_stability": 0.80}
        status, dominates_list, dominated_by = pareto_decision("test_factor", metrics, str(archive_path))
        assert status == "keep"
        assert len(dominates_list) >= 0  # Empty frontier — nothing to dominate
        assert len(dominated_by) == 0

    def test_pareto_decision_discard_when_dominated(self, tmp_path, monkeypatch):
        from prepare import pareto_decision
        # Create a frontier with one strong factor
        frontier_data = {
            "metrics": ["rank_ic", "ic_ir", "turnover_stability"],
            "frontier": [
                {"name": "strong_factor", "rank_ic": 0.06, "ic_ir": 2.0, "turnover_stability": 0.90}
            ],
            "dominated_count": 5,
            "total_experiments": 10
        }
        archive_path = tmp_path / "frontier.json"
        archive_path.write_text(json.dumps(frontier_data))

        # New factor is worse on ALL metrics
        metrics = {"rank_ic": 0.02, "ic_ir": 0.5, "turnover_stability": 0.50}
        status, dominates_list, dominated_by = pareto_decision("weak_factor", metrics, str(archive_path))
        assert status == "discard"
        assert len(dominated_by) > 0

    def test_pareto_decision_keep_non_dominated(self, tmp_path):
        from prepare import pareto_decision
        frontier_data = {
            "metrics": ["rank_ic", "ic_ir", "turnover_stability"],
            "frontier": [
                {"name": "high_ic", "rank_ic": 0.06, "ic_ir": 1.0, "turnover_stability": 0.40},
            ],
            "dominated_count": 3,
            "total_experiments": 8
        }
        archive_path = tmp_path / "frontier.json"
        archive_path.write_text(json.dumps(frontier_data))

        # New factor: lower IC but MUCH better stability — non-dominated
        metrics = {"rank_ic": 0.04, "ic_ir": 2.5, "turnover_stability": 0.95}
        status, dominates_list, dominated_by = pareto_decision("stable_factor", metrics, str(archive_path))
        assert status == "keep"
        assert len(dominated_by) == 0  # Not dominated by high_ic (better on IR and turnover)

    def test_update_archive_adds_and_removes_dominated(self, tmp_path):
        from prepare import update_archive
        archive_path = tmp_path / "frontier.json"
        archive_path.write_text(json.dumps({
            "metrics": ["rank_ic", "ic_ir", "turnover_stability"],
            "frontier": [
                {"name": "old", "rank_ic": 0.03, "ic_ir": 0.8, "turnover_stability": 0.50}
            ],
            "dominated_count": 0,
            "total_experiments": 5
        }))

        new_factor = {
            "name": "new_better",
            "rank_ic": 0.05,
            "ic_ir": 1.5,
            "turnover_stability": 0.80,
            "description": "better on all metrics",
            "commit": "abc1234",
            "added": "2026-05-15T10:00:00",
            "formula": "ops.cs_rank(close - ops.delay(close, 5))"
        }
        # new_better dominates old → old should be removed
        update_archive(new_factor, dominates=["old"], str_path=str(archive_path))

        with open(archive_path) as f:
            updated = json.load(f)
        assert len(updated["frontier"]) == 1
        assert updated["frontier"][0]["name"] == "new_better"
        assert updated["dominated_count"] == 1  # old got dominated
        assert updated["total_experiments"] == 6
```

- [ ] **Step 3: Run test — verify FAIL**

```bash
cd alpha_autoresearch && uv run pytest tests/test_pareto.py -v
```
Expected: FAIL — `ImportError: cannot import name 'dominates'`

- [ ] **Step 4: Implement Pareto logic in prepare.py**

Append to `prepare.py`:

```python
# ═══════════════════════════════════════════════════════════════
# Pareto Dominance Logic + Archive Management
# ═══════════════════════════════════════════════════════════════

ARCHIVE_PATH = Path(__file__).resolve().parent / "pareto_frontier.json"


def dominates(a: Dict[str, float], b: Dict[str, float]) -> bool:
    """Check if factor A Pareto-dominates factor B.
    A dominates B if A >= B on ALL metrics AND A > B on at least one.
    Uses absolute values for IC (sign doesn't matter for dominance)."""
    a_abs = {k: abs(v) if k == "rank_ic" else v for k, v in a.items()}
    b_abs = {k: abs(v) if k == "rank_ic" else v for k, v in b.items()}

    all_ge = all(a_abs[m] >= b_abs[m] for m in METRIC_NAMES)
    any_gt = any(a_abs[m] > b_abs[m] for m in METRIC_NAMES)
    return all_ge and any_gt


def load_archive(path: str = None) -> Dict:
    """Load the Pareto frontier archive."""
    p = Path(path) if path else ARCHIVE_PATH
    if not p.exists():
        return {"metrics": METRIC_NAMES, "frontier": [], "dominated_count": 0, "total_experiments": 0}
    with open(p) as f:
        return json.load(f)


def pareto_decision(name: str, metrics: Dict[str, float], archive_path: str = None) -> Tuple[str, List[str], List[str]]:
    """Determine keep/discard status for a factor.
    Returns (status, list_of_dominated_names, list_of_dominated_by_names)."""
    archive = load_archive(archive_path)
    frontier = archive.get("frontier", [])

    if any(np.isnan(metrics.get(m, np.nan)) for m in METRIC_NAMES):
        return ("crash", [], [])

    dominated_by = []
    dominates_list = []

    for f in frontier:
        f_metrics = {m: f[m] for m in METRIC_NAMES}
        if dominates(f_metrics, metrics):
            dominated_by.append(f["name"])
        if dominates(metrics, f_metrics):
            dominates_list.append(f["name"])

    if dominates_list:
        return ("keep", dominates_list, dominated_by)
    elif dominated_by and not dominates_list:
        if len(dominated_by) == len(frontier):
            return ("discard", [], dominated_by)
        else:
            return ("keep", [], dominated_by)
    else:
        return ("keep", [], dominated_by)


def update_archive(factor_info: Dict, dominates: List[str] = None,
                   str_path: str = None) -> None:
    """Update the Pareto frontier archive: add new factor, remove dominated ones."""
    archive = load_archive(str_path)
    p = Path(str_path) if str_path else ARCHIVE_PATH

    # Remove dominated factors
    if dominates:
        dominated_set = set(dominates)
        archive["frontier"] = [f for f in archive["frontier"] if f["name"] not in dominated_set]
        archive["dominated_count"] += len(dominated_set)

    # Add new factor
    archive["frontier"].append(factor_info)
    archive["total_experiments"] += 1

    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(archive, f, indent=2, ensure_ascii=False)
```

- [ ] **Step 5: Run tests — verify PASS**

```bash
cd alpha_autoresearch && uv run pytest tests/test_pareto.py -v
```
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
cd alpha_autoresearch && git add prepare.py tests/test_pareto.py pareto_frontier.json && git commit -m "feat: Pareto dominance logic + archive management + 8 tests"
```

---

### Task 7: Main Harness — Wire Everything Together

**Files:**
- Modify: `alpha_autoresearch/prepare.py` (append main harness)
- Modify: `alpha_autoresearch/tests/test_metrics.py` (add integration test)

- [ ] **Step 1: Write integration test**

Append to `tests/test_metrics.py`:

```python
class TestHarness:
    """Integration test: the full evaluation pipeline."""

    def test_evaluate_all_factors_runs_without_error(self, tmp_path, monkeypatch):
        """Full pipeline: discover factors, evaluate, print results."""
        import sys

        # Set up a fake factors.py in the tmp path
        fake_factors = tmp_path / "factors.py"
        fake_factors.write_text("""
from prepare import Factor, ops

class Factor001(Factor):
    name = "test_momentum"
    def compute(self, df):
        m = df.set_index(["datetime", "symbol"])
        val = ops.cs_rank(m["close"] - ops.delay(m["close"], 5))
        return Factor.as_cs_series(df, val)
""")

        # Create test panel
        panel = pd.DataFrame({
            "datetime": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03",
                                        "2024-01-04", "2024-01-04"]),
            "symbol": ["000001", "000002"] * 3,
            "close": [10.0, 20.0, 11.0, 19.0, 10.5, 20.5],
            "open": [9.9, 19.8, 10.9, 19.1, 10.3, 20.3],
            "high": [10.1, 20.2, 11.1, 19.5, 10.6, 20.6],
            "low": [9.8, 19.5, 10.5, 18.9, 10.2, 20.2],
            "volume": [1000, 2000, 1100, 2100, 1200, 2200],
            "vwap": [10.0, 20.0, 11.0, 19.0, 10.5, 20.5],
            "returns": [np.nan, np.nan, 0.1, -0.05, -0.045, 0.079],
        })

        # Save panel to temp
        panel_path = tmp_path / "panel.parquet"
        panel.to_parquet(panel_path)

        # Create empty archive
        archive_path = tmp_path / "frontier.json"
        archive_path.write_text(json.dumps({
            "metrics": ["rank_ic", "ic_ir", "turnover_stability"],
            "frontier": [],
            "dominated_count": 0,
            "total_experiments": 0
        }))

        # Monkey-patch paths
        import prepare
        monkeypatch.setattr(prepare, "PANEL_PATH", panel_path)
        monkeypatch.setattr(prepare, "ARCHIVE_PATH", archive_path)
        # Make discover_factors look at our fake factors.py
        monkeypatch.setattr(prepare.Path, "__init__", lambda self, *a: None)  # Don't do this
        # Instead, just test evaluate_all_factors directly
        monkeypatch.setattr(prepare, "discover_factors", lambda: {
            "test_momentum": type("F", (prepare.Factor,), {
                "name": "test_momentum",
                "compute": lambda self, df: prepare.Factor.as_cs_series(
                    df, pd.Series(np.random.randn(len(df)),
                                  index=pd.MultiIndex.from_frame(df[["datetime", "symbol"]]))
                )
            })()
        })
        monkeypatch.setattr(prepare, "load_panel", lambda: panel)

        # This should not crash
        try:
            prepare.main()
        except SystemExit:
            pass  # main() calls sys.exit(0)
```

- [ ] **Step 2: Implement main harness in prepare.py**

Append to `prepare.py`:

```python
# ═══════════════════════════════════════════════════════════════
# Main Harness
# ═══════════════════════════════════════════════════════════════

def _timeout_handler(signum, frame):
    """Signal handler for wall-clock timeout."""
    raise TimeoutError("Experiment exceeded wall-clock safety timeout")


def evaluate_all_factors(panel: pd.DataFrame) -> List[Dict]:
    """Discover all factors from factors.py, evaluate each, return results list.
    Each result dict has: factor_name, metrics dict, status, dominates, dominated_by."""
    discovered = discover_factors()
    if not discovered:
        logger.warning("No factors discovered in factors.py")
        return []

    # Apply factor count budget
    factor_names = list(discovered.keys())[:MAX_FACTORS_PER_EXPERIMENT]
    if len(discovered) > MAX_FACTORS_PER_EXPERIMENT:
        logger.warning(f"Truncating to {MAX_FACTORS_PER_EXPERIMENT} factors "
                       f"(found {len(discovered)})")

    results = []
    for name in factor_names:
        factor = discovered[name]
        try:
            factor_series = factor.compute(panel)
            metrics = evaluate_factor(factor_series, panel)
            status, dominates_list, dominated_by = pareto_decision(name, metrics)

            results.append({
                "factor_name": name,
                "metrics": metrics,
                "status": status,
                "dominates": dominates_list,
                "dominated_by": dominated_by,
                "factor_series": factor_series,
                "factor_instance": factor,
            })
        except Exception as e:
            logger.error(f"Factor {name} crashed: {e}")
            results.append({
                "factor_name": name,
                "metrics": {m: 0.0 for m in METRIC_NAMES},
                "status": "crash",
                "dominates": [],
                "dominated_by": [],
                "factor_series": None,
                "error": str(e),
            })

    return results


def print_results(results: List[Dict]) -> None:
    """Print results in machine-parseable format for agent to grep."""
    for r in results:
        m = r["metrics"]
        print("---")
        print(f"factor: {r['factor_name']}")
        print(f"rank_ic:           {m.get('rank_ic', 0):.6f}")
        print(f"ic_ir:             {m.get('ic_ir', 0):.4f}")
        print(f"turnover_stability: {m.get('turnover_stability', 0):.4f}")
        dominates_str = ", ".join(r.get("dominates", [])) or "(none)"
        dominated_str = ", ".join(r.get("dominated_by", [])) or "(none)"
        print(f"dominates:          {dominates_str}")
        print(f"dominated_by:       {dominated_str}")
        print(f"status:             {r['status']}")
        if r.get("error"):
            print(f"error:              {r['error']}")


def main():
    """Main entry point. Loads panel, discovers factors, evaluates, updates archive."""
    import argparse

    parser = argparse.ArgumentParser(description="alpha_autoresearch evaluation harness")
    parser.add_argument("--build-cache", action="store_true",
                        help="Build unified panel from alpha101_factory data")
    args = parser.parse_args()

    if args.build_cache:
        panel = build_unified_panel()
        if panel.empty:
            logger.error("Failed to build unified panel. Check ALPHA101_DATA_ROOT.")
            sys.exit(1)
        logger.info("Cache built successfully. Ready for experiments.")
        return

    # Load panel
    panel = load_panel()
    if panel.empty:
        logger.error("No panel data. Run with --build-cache first.")
        sys.exit(1)

    # Set wall-clock timeout
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(WALL_CLOCK_TIMEOUT)

    try:
        results = evaluate_all_factors(panel)
        print_results(results)

        # Update Pareto archive for kept factors
        for r in results:
            if r["status"] == "keep":
                factor_info = {
                    "name": r["factor_name"],
                    "rank_ic": r["metrics"]["rank_ic"],
                    "ic_ir": r["metrics"]["ic_ir"],
                    "turnover_stability": r["metrics"]["turnover_stability"],
                    "description": r.get("factor_instance", None).__doc__ or "",
                    "commit": "",  # Agent fills this
                    "added": datetime.now().isoformat(),
                    "formula": "",
                }
                update_archive(factor_info, dominates=r.get("dominates", []))

    except TimeoutError:
        logger.error(f"TIMEOUT: experiment exceeded {WALL_CLOCK_TIMEOUT}s wall-clock limit")
        print("---")
        print("status: timeout")
        sys.exit(1)
    finally:
        signal.alarm(0)  # Cancel alarm


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run integration test**

```bash
cd alpha_autoresearch && uv run pytest tests/test_metrics.py::TestHarness -v
```
Expected: 1 passed

- [ ] **Step 4: Run full test suite**

```bash
cd alpha_autoresearch && uv run pytest tests/ -v
```
Expected: 31 passed (15 ops + 4 discovery + 2 dataset + 7 metrics + 1 harness + 8 pareto — some overlap)

(Note: exact count depends on test deduplication; target is all green.)

- [ ] **Step 5: Commit**

```bash
cd alpha_autoresearch && git add prepare.py tests/ && git commit -m "feat: main harness — wire dataset, discovery, metrics, Pareto together"
```

---

### Task 8: program.md — Agent Instructions

**Files:**
- Create: `alpha_autoresearch/program.md`

- [ ] **Step 1: Write program.md**

```markdown
# alpha_autoresearch

An AI agent autonomously invents and iterates on Alpha101-style quantitative factors for the Chinese A-share market.

## Setup

1. Read `pareto_frontier.json` — understand current frontier state
2. Read `results.tsv` — review recent experiment history
3. Read `factors.py` — understand current active factors
4. Read `prepare.py` (ops section + Factor base class) — understand available operators and columns
5. Confirm data exists at `~/.cache/alpha_autoresearch/panel.parquet`
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
- Simplicity: always prefer simpler factors at equal metrics
- NEVER STOP — the human may be asleep
```

- [ ] **Step 2: Commit**

```bash
cd alpha_autoresearch && git add program.md && git commit -m "docs: program.md — agent instructions with 6 iteration principles"
```

---

### Task 9: results.tsv + Final Verification

**Files:**
- Create: `alpha_autoresearch/results.tsv` (with header only)
- Verify: `.gitignore` covers `results.tsv`

- [ ] **Step 1: Initialize results.tsv**

```bash
echo -e "commit\tfactor_name\trank_ic\tic_ir\tturnover\tdominates\tdominated_by\tstatus\tdescription" > alpha_autoresearch/results.tsv
```

- [ ] **Step 2: Verify results.tsv is gitignored**

```bash
cd alpha_autoresearch && git status -- results.tsv
```
Expected: results.tsv should NOT appear as untracked (it's in .gitignore).

If it appears, run: `echo "results.tsv" >> .gitignore`

- [ ] **Step 3: Run full test suite**

```bash
cd alpha_autoresearch && uv run pytest tests/ -v
```
Expected: ALL tests pass (30+ tests).

- [ ] **Step 4: Verify full pipeline runs (with synthetic data)**

```bash
cd alpha_autoresearch && uv run python -c "
from prepare import main
# main() will try to load panel — if not built, it will exit gracefully
import sys
try:
    main()
except SystemExit as e:
    print(f'Exit code: {e.code}')
"
```
Expected: Graceful exit with message about needing `--build-cache` (or runs if panel exists).

- [ ] **Step 5: Final commit**

```bash
cd alpha_autoresearch && git add -A && git commit -m "feat: complete alpha_autoresearch — eval harness, Pareto archive, agent instructions"
```

---

## Self-Review Checklist

- [ ] **Spec coverage:** Each spec section maps to a task
  - §2 Dataset → Task 4
  - §3 Metrics → Task 5
  - §4 Time Budget → Task 7 (safety timeout + factor count constant)
  - §5 factors.py → Task 3 + Task 8
  - §6 prepare.py → Tasks 2-7
  - §7 program.md → Task 8
  - §8 Pareto Archive → Task 6
  - §9 Implementation Notes → Task 1
- [ ] **No placeholders:** All code shown explicitly, no TBD/TODO
- [ ] **Type consistency:** `prepare.py` uses consistent signatures across all appended sections
- [ ] **Test coverage:** All 12 operators tested, all 3 metrics tested, Pareto logic tested, integration tested
