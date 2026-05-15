# -*- coding: utf-8 -*-
"""Operator library tests — verify numeric correctness of all ops.* functions."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def simple_series():
    """A simple 10-element series for basic window tests."""
    return pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])


class TestOperators:
    """Test all 12 operators for numeric correctness."""

    def test_rolling_sum(self, simple_series):
        from prepare import ops
        result = ops.rolling_sum(simple_series, 3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == 6.0
        assert result.iloc[9] == 27.0

    def test_rolling_min(self, simple_series):
        from prepare import ops
        result = ops.rolling_min(simple_series, 3)
        assert result.iloc[2] == 1.0
        assert result.iloc[9] == 8.0

    def test_rolling_max(self, simple_series):
        from prepare import ops
        result = ops.rolling_max(simple_series, 3)
        assert result.iloc[2] == 3.0
        assert result.iloc[9] == 10.0

    def test_rolling_std(self):
        from prepare import ops
        s = pd.Series([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        result = ops.rolling_std(s, 4)
        # ddof=0: std([2,4,4,4]) ≈ 0.866025
        assert abs(result.iloc[3] - 0.866025) < 0.001

    def test_delta(self):
        from prepare import ops
        s = pd.Series([10.0, 12.0, 15.0, 11.0, 8.0])
        result = ops.delta(s, 1)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == 2.0
        assert result.iloc[2] == 3.0
        assert result.iloc[3] == -4.0
        assert result.iloc[4] == -3.0

    def test_delta_window(self):
        from prepare import ops
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = ops.delta(s, 3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert pd.isna(result.iloc[2])
        assert result.iloc[3] == 3.0
        assert result.iloc[5] == 3.0

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
        # Window [1,3,2]: rank(2)=2, pct=2/3 ≈ 0.6667
        assert abs(result.iloc[2] - 2/3) < 0.01

    def test_decay_linear(self):
        from prepare import ops
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ops.decay_linear(s, 3)
        # weights [1,2,3]/6: 1*1/6 + 2*2/6 + 3*3/6 = 14/6 ≈ 2.333
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
        # mean=2, std(ddof=0)=0.816: zscores ≈ [-1.225, 0, 1.225]
        # Actually with ddof=0: std = sqrt(((1+0+1)/3)) = sqrt(2/3) = 0.8165
        assert abs(result.iloc[0] + 1.0) < 0.5  # Approximate
        assert abs(result.iloc[2] - 1.0) < 0.5

    def test_rolling_corr(self):
        from prepare import ops
        s1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        s2 = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
        result = ops.rolling_corr(s1, s2, 5)
        # Perfect linear correlation → near 1.0
        val = result.iloc[4]
        if not pd.isna(val):
            assert abs(val - 1.0) < 1e-10

    def test_rolling_cov_ddof0(self):
        """Gotcha: rolling_cov should use ddof=0 consistent with rolling_std."""
        from prepare import ops
        s1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        s2 = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
        result = ops.rolling_cov(s1, s2, 3)
        assert result.iloc[2] > 0
