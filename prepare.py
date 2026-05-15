# -*- coding: utf-8 -*-
import os
import sys
import json
import signal
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


# ═══════════════════════════════════════════════════════════════
# Constants (fixed, do not modify)
# ═══════════════════════════════════════════════════════════════

MAX_FACTORS_PER_EXPERIMENT = 10
WALL_CLOCK_TIMEOUT = 60
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.getenv("ALPHA101_DATA_ROOT", str(PROJECT_ROOT.parent / "alpha101_factory" / "data")))
PARQ_DIR_KLINES = DATA_ROOT / "klines_daily"
CACHE_DIR = Path(os.path.expanduser("~/.cache/alpha_autoresearch"))
PANEL_PATH = PROJECT_ROOT / "data" / "panel.parquet"
FULL_PANEL_PATH = CACHE_DIR / "panel.parquet"
START_DATE = os.getenv("ALPHA101_START", "20200101")
END_DATE = os.getenv("ALPHA101_END", "20250917")
ADJUST = os.getenv("ALPHA101_ADJUST", "qfq")

METRIC_NAMES = ["rank_ic", "ic_ir", "turnover_stability"]

ADV_WINDOWS = [5, 10, 20, 30, 40, 60, 120, 150, 180]


# ═══════════════════════════════════════════════════════════════
# Operator Library
# ═══════════════════════════════════════════════════════════════

class _Ops:
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


# ═══════════════════════════════════════════════════════════════
# Factor Base Class
# ═══════════════════════════════════════════════════════════════

class Factor:
    name: str = "UnnamedFactor"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    @staticmethod
    def as_cs_series(df: pd.DataFrame, values: pd.Series) -> pd.Series:
        if len(values) != len(df):
            raise ValueError(
                f"Factor output length ({len(values)}) does not match "
                f"input DataFrame length ({len(df)}). "
                "Ensure compute() returns one value per row."
            )
        idx = pd.MultiIndex.from_frame(df[["datetime", "symbol"]], names=["datetime", "symbol"])
        return pd.Series(values.values, index=idx, name="value")


# ═══════════════════════════════════════════════════════════════
# Factor Auto-Discovery
# ═══════════════════════════════════════════════════════════════

def discover_factors() -> Dict[str, Factor]:
    factors_path = Path(__file__).resolve().parent / "factors.py"
    if not factors_path.exists():
        logger.warning("factors.py not found")
        return {}

    if __name__ == "__main__":
        sys.modules["prepare"] = sys.modules["__main__"]

    try:
        spec = importlib.util.spec_from_file_location("factors_module", factors_path)
        if spec is None or spec.loader is None:
            logger.error("Failed to load factors.py: invalid spec")
            return {}

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        logger.error(f"Failed to load factors.py: {e}")
        return {}

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


# ═══════════════════════════════════════════════════════════════
# Unified Dataset Loader
# ═══════════════════════════════════════════════════════════════

def _build_features_from_kline(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("datetime").reset_index(drop=True)
    df["returns"] = df["close"].pct_change()
    if "amount" in df.columns and not df["amount"].isna().all():
        df["vwap"] = df["amount"] / df["volume"].replace(0, np.nan)
    else:
        df["vwap"] = df["close"]
    for n in ADV_WINDOWS:
        df[f"adv{n}"] = df["volume"].rolling(window=n, min_periods=1).mean()
    return df


def _read_kline_file(filepath: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_parquet(filepath)
        if df.empty:
            return None
        required = ["open", "high", "low", "close", "volume", "datetime", "symbol"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning(f"{filepath.name}: missing columns {missing}, skipping")
            return None
        return df
    except Exception as e:
        logger.warning(f"Failed to read {filepath}: {e}")
        return None


def build_unified_panel() -> pd.DataFrame:
    logger.info("Building unified panel from klines_daily...")

    if not PARQ_DIR_KLINES.exists():
        logger.error(f"klines_daily directory not found: {PARQ_DIR_KLINES}")
        logger.info("Set ALPHA101_DATA_ROOT env var or run alpha101_factory fetch first.")
        return pd.DataFrame()

    kline_files = sorted(PARQ_DIR_KLINES.glob("*.parquet"))
    if not kline_files:
        logger.error(f"No kline files found in {PARQ_DIR_KLINES}")
        return pd.DataFrame()

    logger.info(f"Found {len(kline_files)} kline files. Computing features...")
    dfs = []
    for i, f in enumerate(kline_files):
        if (i + 1) % 100 == 0:
            logger.info(f"  {i + 1}/{len(kline_files)} stocks processed...")
        df = _read_kline_file(f)
        if df is None:
            continue
        df = _build_features_from_kline(df)
        keep_cols = ["symbol", "datetime", "open", "high", "low", "close",
                      "volume", "returns", "vwap"] + [f"adv{n}" for n in ADV_WINDOWS]
        df = df[[c for c in keep_cols if c in df.columns]]
        dfs.append(df)

    if not dfs:
        logger.error("No valid kline files found")
        return pd.DataFrame()

    panel = pd.concat(dfs, ignore_index=True)
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    panel = panel.drop_duplicates(subset=["datetime", "symbol"], keep="first")
    panel = panel.sort_values(["datetime", "symbol"]).reset_index(drop=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(FULL_PANEL_PATH, index=False)
    logger.info(f"Unified panel saved: {len(panel):,} rows, {panel['symbol'].nunique()} stocks, "
                f"{panel['datetime'].min().date()} ~ {panel['datetime'].max().date()}")
    return panel


def load_panel() -> pd.DataFrame:
    for path in [PANEL_PATH, FULL_PANEL_PATH]:
        if path.exists():
            df = pd.read_parquet(path)
            df["datetime"] = pd.to_datetime(df["datetime"])
            return df
    logger.warning(f"No panel data found. Run with --build-cache first.")
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
# Metric Computation
# ═══════════════════════════════════════════════════════════════

def _make_forward_return(panel: pd.DataFrame, horizon: int = 1) -> pd.Series:
    if panel.empty:
        return pd.Series(dtype=float)
    p = panel[["datetime", "symbol", "close"]].copy()
    p = p.sort_values(["symbol", "datetime"])
    ret = p.groupby("symbol")["close"].pct_change(horizon).shift(-horizon)
    idx = pd.MultiIndex.from_frame(p[["datetime", "symbol"]], names=["datetime", "symbol"])
    return pd.Series(ret.values, index=idx, name="fwd_ret")


def compute_daily_rank_ic(factor: pd.Series, panel: pd.DataFrame) -> pd.Series:
    fwd = _make_forward_return(panel)
    if fwd is None or len(fwd) == 0:
        return pd.Series(dtype=float)

    df = pd.DataFrame({"factor": factor, "fwd_ret": fwd}).dropna()
    if df.empty:
        return pd.Series(dtype=float)

    daily_ic = {}
    for dt, g in df.groupby(level=0):
        if g.index.get_level_values(1).nunique() < 2:
            continue
        ic = g["factor"].corr(g["fwd_ret"], method="spearman")
        daily_ic[dt] = ic

    return pd.Series(daily_ic).sort_index()


def compute_rank_ic(factor: pd.Series, panel: pd.DataFrame) -> float:
    daily = compute_daily_rank_ic(factor, panel)
    if daily.empty:
        return np.nan
    return float(daily.mean())


def compute_ic_ir(daily_ic: pd.Series) -> float:
    valid = daily_ic.dropna()
    if len(valid) < 2:
        return 0.0
    mu, sd = valid.mean(), valid.std()
    if pd.isna(sd) or sd < 1e-12:
        return 0.0
    return float(mu / sd)


def compute_turnover_stability(factor: pd.Series) -> float:
    if factor.empty or not isinstance(factor.index, pd.MultiIndex):
        return np.nan

    rank_per_date = factor.groupby(level=0).rank(pct=True)
    turnover_series = rank_per_date.groupby(level=1).diff().abs()
    mean_turnover = turnover_series.mean()

    if pd.isna(mean_turnover):
        return 1.0

    return float(1.0 - mean_turnover)


def evaluate_factor(factor: pd.Series, panel: pd.DataFrame) -> Dict[str, float]:
    daily_ic = compute_daily_rank_ic(factor, panel)
    rank_ic = float(daily_ic.mean()) if not daily_ic.empty else np.nan
    ic_ir = compute_ic_ir(daily_ic)
    turnover = compute_turnover_stability(factor)

    return {
        "rank_ic": rank_ic,
        "ic_ir": ic_ir,
        "turnover_stability": turnover,
    }


# ═══════════════════════════════════════════════════════════════
# Pareto Dominance Logic + Archive Management
# ═══════════════════════════════════════════════════════════════

ARCHIVE_PATH = Path(__file__).resolve().parent / "pareto_frontier.json"


def dominates(a: Dict[str, float], b: Dict[str, float]) -> bool:
    a_abs = {k: abs(v) if k == "rank_ic" else v for k, v in a.items()}
    b_abs = {k: abs(v) if k == "rank_ic" else v for k, v in b.items()}
    all_ge = all(a_abs[m] >= b_abs[m] for m in METRIC_NAMES)
    any_gt = any(a_abs[m] > b_abs[m] for m in METRIC_NAMES)
    return all_ge and any_gt


def load_archive(path: str = None) -> Dict:
    p = Path(path) if path else ARCHIVE_PATH
    if not p.exists():
        return {"metrics": METRIC_NAMES, "frontier": [], "dominated_count": 0, "total_experiments": 0}
    with open(p) as f:
        return json.load(f)


def pareto_decision(name: str, metrics: Dict[str, float], archive_path: str = None) -> Tuple[str, List[str], List[str]]:
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
    archive = load_archive(str_path)
    p = Path(str_path) if str_path else ARCHIVE_PATH

    if dominates:
        dominated_set = set(dominates)
        archive["frontier"] = [f for f in archive["frontier"] if f["name"] not in dominated_set]
        archive["dominated_count"] += len(dominated_set)

    archive["frontier"].append(factor_info)
    archive["total_experiments"] += 1

    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(archive, f, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════
# Main Harness
# ═══════════════════════════════════════════════════════════════

def _timeout_handler(signum, frame):
    raise TimeoutError("Experiment exceeded wall-clock safety timeout")


def evaluate_all_factors(panel: pd.DataFrame) -> List[Dict]:
    discovered = discover_factors()
    if not discovered:
        logger.warning("No factors discovered in factors.py")
        return []

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
            })
        except Exception as e:
            logger.error(f"Factor {name} crashed: {e}")
            results.append({
                "factor_name": name,
                "metrics": {m: 0.0 for m in METRIC_NAMES},
                "status": "crash",
                "dominates": [],
                "dominated_by": [],
                "error": str(e),
            })

    return results


def print_results(results: List[Dict]) -> None:
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

    panel = load_panel()
    if panel.empty:
        logger.error("No panel data. Run with --build-cache first.")
        sys.exit(1)

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(WALL_CLOCK_TIMEOUT)

    try:
        results = evaluate_all_factors(panel)
        print_results(results)

        for r in results:
            if r["status"] == "keep":
                factor_info = {
                    "name": r["factor_name"],
                    "rank_ic": r["metrics"]["rank_ic"],
                    "ic_ir": r["metrics"]["ic_ir"],
                    "turnover_stability": r["metrics"]["turnover_stability"],
                    "description": "",
                    "commit": "",
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
        signal.alarm(0)


if __name__ == "__main__":
    main()
