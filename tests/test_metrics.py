# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest


def make_test_panel(n_symbols=5, n_days=50, seed=42):
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
    np.random.seed(seed)
    idx = pd.MultiIndex.from_frame(df[["datetime", "symbol"]], names=["datetime", "symbol"])
    base = df.groupby("symbol")["close"].pct_change().shift(-1).fillna(0)
    noise = np.random.normal(0, 0.01, len(df))
    values = base.values + noise
    return pd.Series(values, index=idx)


class TestMetricComputation:

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
        assert result == 0.0

    def test_turnover_stability_computes(self):
        from prepare import compute_turnover_stability
        df = make_test_panel(n_symbols=5, n_days=50)
        factor = make_factor_series(df)
        result = compute_turnover_stability(factor)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_turnover_stability_constant_factor(self):
        from prepare import compute_turnover_stability
        df = make_test_panel(n_symbols=3, n_days=20)
        idx = pd.MultiIndex.from_frame(df[["datetime", "symbol"]], names=["datetime", "symbol"])
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
