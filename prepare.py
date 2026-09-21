# -*- coding: utf-8 -*-
"""Deterministic evaluation harness for cross-sectional equity factors.

Version 2 writes a separate archive: historical v1 metrics used a misaligned
forward-return label and must not be compared with corrected measurements.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

MAX_FACTORS_PER_EXPERIMENT = 10
WALL_CLOCK_TIMEOUT = 60
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("ALPHA101_DATA_ROOT", ""))
PARQ_DIR_KLINES = DATA_ROOT / "klines_daily"
CACHE_DIR = Path(os.path.expanduser("~/.cache/alpha_autoresearch"))
PANEL_PATH = PROJECT_ROOT / "data" / "panel.parquet"
FULL_PANEL_PATH = CACHE_DIR / "panel.parquet"
ARCHIVE_PATH = PROJECT_ROOT / "pareto_frontier.v2.json"
METRIC_NAMES = ["rank_ic", "ic_ir", "turnover_stability"]
ADV_WINDOWS = [5, 10, 20, 30, 40, 60, 120, 150, 180]
EVALUATION_VERSION = 2


def _validate_index(s: pd.Series) -> None:
    if not isinstance(s.index, pd.MultiIndex) or s.index.nlevels != 2 or list(s.index.names) != ["datetime", "symbol"]:
        raise ValueError("Factor series requires a MultiIndex (datetime, symbol)")
    if not s.index.is_unique:
        raise ValueError("Factor series has duplicate (datetime, symbol) keys")


def _unary_ts(s: pd.Series, operation) -> pd.Series:
    """Apply a time-series operation independently to each chronological symbol."""
    if not isinstance(s.index, pd.MultiIndex):
        return operation(s)
    _validate_index(s)
    ordered = s.sort_index(level=["symbol", "datetime"], sort_remaining=False)
    result = ordered.groupby(level="symbol", sort=False).transform(operation)
    return result.reindex(s.index)


def _pair_ts(s1: pd.Series, s2: pd.Series, operation) -> pd.Series:
    if not isinstance(s1.index, pd.MultiIndex) and not isinstance(s2.index, pd.MultiIndex):
        return operation(s1, s2)
    _validate_index(s1)
    _validate_index(s2)
    if not s1.index.equals(s2.index):
        raise ValueError("Paired time-series inputs must have identical indices and row order")
    ordered = pd.DataFrame({"a": s1, "b": s2}).sort_index(level=["symbol", "datetime"], sort_remaining=False)
    result = ordered.groupby(level="symbol", sort=False, group_keys=False).apply(
        lambda part: operation(part["a"], part["b"])
    )
    return result.reindex(s1.index)


class _Ops:
    @staticmethod
    def rolling_sum(s: pd.Series, n: int) -> pd.Series:
        return _unary_ts(s, lambda x: x.rolling(n, min_periods=n).sum())

    @staticmethod
    def rolling_min(s: pd.Series, n: int) -> pd.Series:
        return _unary_ts(s, lambda x: x.rolling(n, min_periods=n).min())

    @staticmethod
    def rolling_max(s: pd.Series, n: int) -> pd.Series:
        return _unary_ts(s, lambda x: x.rolling(n, min_periods=n).max())

    @staticmethod
    def rolling_std(s: pd.Series, n: int) -> pd.Series:
        return _unary_ts(s, lambda x: x.rolling(n, min_periods=n).std(ddof=0))

    @staticmethod
    def rolling_corr(s1: pd.Series, s2: pd.Series, n: int) -> pd.Series:
        return _pair_ts(s1, s2, lambda x, y: x.rolling(n, min_periods=n).corr(y))

    @staticmethod
    def rolling_cov(s1: pd.Series, s2: pd.Series, n: int) -> pd.Series:
        return _pair_ts(s1, s2, lambda x, y: x.rolling(n, min_periods=n).cov(y, ddof=0))

    @staticmethod
    def delta(s: pd.Series, n: int = 1) -> pd.Series:
        return _unary_ts(s, lambda x: x.diff(n))

    @staticmethod
    def delay(s: pd.Series, n: int = 1) -> pd.Series:
        return _unary_ts(s, lambda x: x.shift(n))

    @staticmethod
    def ts_rank(s: pd.Series, n: int) -> pd.Series:
        return _unary_ts(s, lambda x: x.rolling(n, min_periods=n).apply(
            lambda values: pd.Series(values).rank(pct=True).iloc[-1], raw=True
        ))

    @staticmethod
    def decay_linear(s: pd.Series, n: int) -> pd.Series:
        if n < 1:
            raise ValueError("n must be positive")
        weights = np.arange(1, n + 1, dtype=float)
        weights /= weights.sum()
        return _unary_ts(s, lambda x: x.rolling(n, min_periods=n).apply(
            lambda values: float(np.dot(values, weights)), raw=True
        ))

    @staticmethod
    def cs_rank(s: pd.Series) -> pd.Series:
        _validate_index(s)
        return s.groupby(level="datetime").rank(pct=True)

    @staticmethod
    def cs_zscore(s: pd.Series) -> pd.Series:
        _validate_index(s)
        grouped = s.groupby(level="datetime")
        return (s - grouped.transform("mean")) / grouped.transform(
            lambda values: values.std(ddof=0)
        ).replace(0, np.nan)


ops = _Ops()


class Factor:
    name: str = "UnnamedFactor"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    @staticmethod
    def as_cs_series(df: pd.DataFrame, values: pd.Series) -> pd.Series:
        if not isinstance(values, pd.Series):
            raise TypeError("Factor output must be a pandas Series")
        if len(values) != len(df):
            raise ValueError(f"Factor output length ({len(values)}) does not match input ({len(df)})")
        index = pd.MultiIndex.from_frame(df[["datetime", "symbol"]], names=["datetime", "symbol"])
        if not index.is_unique:
            raise ValueError("Panel has duplicate (datetime, symbol) keys")
        if isinstance(values.index, pd.MultiIndex):
            _validate_index(values)
            if not values.index.equals(index):
                if not values.index.difference(index).empty or not index.difference(values.index).empty:
                    raise ValueError("Factor output keys do not match panel keys")
                values = values.reindex(index)
        elif not values.index.equals(df.index):
            raise ValueError("Unindexed factor output must preserve the panel row index")
        return pd.Series(pd.to_numeric(values, errors="raise").to_numpy(), index=index, name="value")


def discover_factors() -> Dict[str, Factor]:
    factors_path = PROJECT_ROOT / "factors.py"
    if not factors_path.exists():
        logger.warning("factors.py not found")
        return {}
    if __name__ == "__main__":
        sys.modules["prepare"] = sys.modules["__main__"]
    try:
        spec = importlib.util.spec_from_file_location("factors_module", factors_path)
        if spec is None or spec.loader is None:
            raise ImportError("Invalid factor module specification")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:
        logger.error(f"Failed to load factors.py: {exc}")
        return {}
    discovered: Dict[str, Factor] = {}
    for class_name in sorted(dir(module)):
        if not class_name.startswith("Factor") or class_name == "Factor":
            continue
        obj = getattr(module, class_name)
        if isinstance(obj, type) and issubclass(obj, Factor) and obj is not Factor and obj.__module__ == module.__name__:
            try:
                instance = obj()
                if not isinstance(instance.name, str) or not instance.name.strip():
                    raise ValueError("Factor name must be a non-empty string")
                if instance.name in discovered:
                    raise ValueError(f"Duplicate factor name: {instance.name}")
                discovered[instance.name] = instance
            except Exception as exc:
                logger.warning(f"Skipping {class_name}: {exc}")
    return discovered


def _build_features_from_kline(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("datetime").reset_index(drop=True).copy()
    df["returns"] = df["close"].pct_change(fill_method=None)
    if "amount" in df.columns and not df["amount"].isna().all():
        df["vwap"] = df["amount"] / df["volume"].replace(0, np.nan)
    else:
        df["vwap"] = df["close"]
    for n in ADV_WINDOWS:
        df[f"adv{n}"] = df["volume"].rolling(n, min_periods=1).mean()
    return df


def _read_kline_file(filepath: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_parquet(filepath)
        if df.empty:
            return None
        required = ["open", "high", "low", "close", "volume", "datetime", "symbol"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            logger.warning(f"{filepath.name}: missing columns {missing}, skipping")
            return None
        return df
    except Exception as exc:
        logger.warning(f"Failed to read {filepath}: {exc}")
        return None


def build_unified_panel() -> pd.DataFrame:
    if not PARQ_DIR_KLINES.exists():
        logger.error(f"klines_daily directory not found: {PARQ_DIR_KLINES}")
        return pd.DataFrame()
    frames = []
    for filepath in sorted(PARQ_DIR_KLINES.glob("*.parquet")):
        raw = _read_kline_file(filepath)
        if raw is None:
            continue
        raw["datetime"] = pd.to_datetime(raw["datetime"])
        raw = raw.drop_duplicates(["datetime", "symbol"], keep="first")
        for _, stock in raw.groupby("symbol", sort=False):
            features = _build_features_from_kline(stock)
            columns = ["symbol", "datetime", "open", "high", "low", "close", "volume", "returns", "vwap"] + [
                f"adv{n}" for n in ADV_WINDOWS
            ]
            frames.append(features[columns])
    if not frames:
        logger.error("No valid kline files found")
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["datetime", "symbol"]).reset_index(drop=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(FULL_PANEL_PATH, index=False)
    logger.info(f"Saved {len(panel):,} rows across {panel['symbol'].nunique()} symbols")
    return panel


def load_panel() -> pd.DataFrame:
    for path in (PANEL_PATH, FULL_PANEL_PATH):
        if path.exists():
            panel = pd.read_parquet(path)
            panel["datetime"] = pd.to_datetime(panel["datetime"])
            if panel.duplicated(["datetime", "symbol"]).any():
                raise ValueError(f"Panel {path} has duplicate (datetime, symbol) keys")
            return panel.sort_values(["datetime", "symbol"]).reset_index(drop=True)
    logger.warning("No panel data found. Supply data/panel.parquet or run --build-cache")
    return pd.DataFrame()


def _make_forward_return(panel: pd.DataFrame, horizon: int = 1) -> pd.Series:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if panel.empty:
        return pd.Series(dtype=float)
    if panel.duplicated(["datetime", "symbol"]).any():
        raise ValueError("Panel has duplicate (datetime, symbol) keys")
    sorted_panel = panel[["datetime", "symbol", "close"]].sort_values(["symbol", "datetime"])
    close = pd.to_numeric(sorted_panel["close"], errors="raise")
    future_close = close.groupby(sorted_panel["symbol"], sort=False).shift(-horizon)
    forward = (future_close / close - 1).replace([np.inf, -np.inf], np.nan)
    index = pd.MultiIndex.from_frame(sorted_panel[["datetime", "symbol"]], names=["datetime", "symbol"])
    return pd.Series(forward.to_numpy(), index=index, name="fwd_ret")


def compute_daily_rank_ic(factor: pd.Series, panel: pd.DataFrame) -> pd.Series:
    if panel.empty:
        return pd.Series(dtype=float)
    _validate_index(factor)
    forward = _make_forward_return(panel)
    if not factor.index.equals(forward.index):
        if not factor.index.difference(forward.index).empty or not forward.index.difference(factor.index).empty:
            raise ValueError("Factor and return labels have different (datetime, symbol) keys")
    values = pd.concat([factor.rename("factor"), forward], axis=1)
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    daily = {}
    for date, subset in values.groupby(level="datetime"):
        if len(subset) < 2 or subset["factor"].nunique() < 2 or subset["fwd_ret"].nunique() < 2:
            continue
        correlation = subset["factor"].corr(subset["fwd_ret"], method="spearman")
        if np.isfinite(correlation):
            daily[date] = float(correlation)
    return pd.Series(daily, dtype=float).sort_index()


def compute_rank_ic(factor: pd.Series, panel: pd.DataFrame) -> float:
    daily = compute_daily_rank_ic(factor, panel)
    return float(daily.mean()) if not daily.empty else float("nan")


def compute_ic_ir(daily_ic: pd.Series) -> float:
    valid = daily_ic.replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 2:
        return 0.0
    standard_deviation = valid.std()
    if pd.isna(standard_deviation) or standard_deviation < 1e-12:
        return 0.0
    return float(valid.mean() / standard_deviation)


def compute_turnover_stability(factor: pd.Series) -> float:
    if factor.empty:
        return float("nan")
    _validate_index(factor)
    clean = factor.replace([np.inf, -np.inf], np.nan)
    ordered = clean.sort_index(level=["symbol", "datetime"], sort_remaining=False)
    ranks = ordered.groupby(level="datetime").rank(pct=True)
    changes = ranks.groupby(level="symbol", sort=False).diff().abs()
    if changes.notna().sum() == 0:
        return float("nan")
    return float(1.0 - changes.mean())


def evaluate_factor(factor: pd.Series, panel: pd.DataFrame) -> Dict[str, float]:
    _validate_index(factor)
    daily = compute_daily_rank_ic(factor, panel)
    return {"rank_ic": float(daily.mean()) if not daily.empty else float("nan"),
            "ic_ir": compute_ic_ir(daily),
            "turnover_stability": compute_turnover_stability(factor)}


def _valid_metrics(metrics: Dict[str, float]) -> bool:
    try:
        return all(np.isfinite(float(metrics[key])) for key in METRIC_NAMES)
    except (TypeError, ValueError, KeyError):
        return False


def dominates(a: Dict[str, float], b: Dict[str, float]) -> bool:
    if not _valid_metrics(a) or not _valid_metrics(b):
        return False
    a_values = [abs(a[m]) if m in ("rank_ic", "ic_ir") else a[m] for m in METRIC_NAMES]
    b_values = [abs(b[m]) if m in ("rank_ic", "ic_ir") else b[m] for m in METRIC_NAMES]
    return all(left >= right for left, right in zip(a_values, b_values)) and any(
        left > right for left, right in zip(a_values, b_values)
    )


def _empty_archive() -> Dict:
    return {"evaluation_version": EVALUATION_VERSION, "metrics": METRIC_NAMES,
            "frontier": [], "dominated_count": 0, "total_experiments": 0}


def load_archive(path: Optional[str] = None) -> Dict:
    archive_path = Path(path) if path else ARCHIVE_PATH
    if not archive_path.exists():
        return _empty_archive()
    with archive_path.open(encoding="utf-8") as file:
        archive = json.load(file)
    if archive_path == ARCHIVE_PATH and archive.get("evaluation_version") != EVALUATION_VERSION:
        raise ValueError("Incompatible archive evaluation version; restore a v2 archive or use a new path")
    return archive


def pareto_decision(name: str, metrics: Dict[str, float], archive_path: Optional[str] = None) -> Tuple[str, List[str], List[str]]:
    if not _valid_metrics(metrics):
        return "crash", [], []
    frontier = load_archive(archive_path).get("frontier", [])
    dominated_by = []
    dominates_list = []
    for item in frontier:
        if item["name"] == name:
            return "discard", [], [name]
        current = {m: item[m] for m in METRIC_NAMES}
        if dominates(current, metrics):
            dominated_by.append(item["name"])
        if dominates(metrics, current):
            dominates_list.append(item["name"])
    if dominated_by:
        return "discard", [], dominated_by
    return "keep", dominates_list, []


def update_archive(factor_info: Dict, dominates: Optional[List[str]] = None,
                   str_path: Optional[str] = None) -> None:
    archive = load_archive(str_path)
    path = Path(str_path) if str_path else ARCHIVE_PATH
    name = factor_info["name"]
    if not _valid_metrics(factor_info):
        raise ValueError("Cannot archive non-finite factor metrics")
    if any(item["name"] == name for item in archive["frontier"]):
        raise ValueError(f"Factor {name!r} already exists in the archive")
    status, removed_names, _ = pareto_decision(name, factor_info, str(path))
    if status != "keep":
        raise ValueError(f"Factor {name!r} is not Pareto-optimal: {status}")
    removed = set(removed_names)
    archive["frontier"] = [item for item in archive["frontier"] if item["name"] not in removed]
    archive["frontier"].append(factor_info)
    archive["dominated_count"] = archive.get("dominated_count", 0) + len(removed)
    archive["total_experiments"] = archive.get("total_experiments", 0) + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(archive, file, indent=2, ensure_ascii=False, allow_nan=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def evaluate_all_factors(panel: pd.DataFrame) -> List[Dict]:
    discovered = discover_factors()
    if len(discovered) > MAX_FACTORS_PER_EXPERIMENT:
        logger.warning(f"Evaluating first {MAX_FACTORS_PER_EXPERIMENT} of {len(discovered)} discovered factors")
    results = []
    for name, factor in list(discovered.items())[:MAX_FACTORS_PER_EXPERIMENT]:
        try:
            series = Factor.as_cs_series(panel, factor.compute(panel))
            metrics = evaluate_factor(series, panel)
            status, dominates_list, dominated_by = pareto_decision(name, metrics)
            results.append({"factor_name": name, "metrics": metrics, "status": status,
                            "dominates": dominates_list, "dominated_by": dominated_by})
        except Exception as exc:
            logger.exception(f"Factor {name} failed")
            results.append({"factor_name": name, "metrics": {m: float("nan") for m in METRIC_NAMES},
                            "status": "crash", "dominates": [], "dominated_by": [], "error": str(exc)})
    return results


def print_results(results: List[Dict]) -> None:
    for result in results:
        metrics = result["metrics"]
        print("---")
        print(f"factor: {result['factor_name']}")
        for key in METRIC_NAMES:
            print(f"{key}: {metrics[key]:.6f}")
        print(f"dominates: {', '.join(result['dominates']) or '(none)'}")
        print(f"dominated_by: {', '.join(result['dominated_by']) or '(none)'}")
        print(f"status: {result['status']}")
        if result.get("error"):
            print(f"error: {result['error']}")


def _timeout_handler(signum, frame) -> None:
    raise TimeoutError("Experiment exceeded wall-clock safety timeout")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Alpha factor evaluation harness (v2)")
    parser.add_argument("--build-cache", action="store_true", help="Build full panel from klines_daily")
    parser.add_argument("--no-archive", action="store_true", help="Evaluate without modifying archive")
    arguments = parser.parse_args(argv)
    if arguments.build_cache:
        return 0 if not build_unified_panel().empty else 1
    panel = load_panel()
    if panel.empty:
        logger.error("No panel data found")
        return 1
    alarm_available = hasattr(signal, "SIGALRM")
    if alarm_available:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(WALL_CLOCK_TIMEOUT)
    try:
        results = evaluate_all_factors(panel)
        print_results(results)
        if not results or any(result["status"] == "crash" for result in results):
            return 1
        if not arguments.no_archive:
            for result in results:
                if result["status"] != "keep":
                    continue
                name, metrics = result["factor_name"], result["metrics"]
                status, removed, _ = pareto_decision(name, metrics)
                if status != "keep":
                    continue
                update_archive({"name": name, **metrics, "description": "", "commit": "",
                                "added": datetime.now(timezone.utc).isoformat(), "formula": ""},
                               dominates=removed)
        return 0
    except TimeoutError:
        logger.error(f"Evaluation exceeded {WALL_CLOCK_TIMEOUT}s")
        return 1
    finally:
        if alarm_available:
            signal.alarm(0)


if __name__ == "__main__":
    sys.exit(main())
