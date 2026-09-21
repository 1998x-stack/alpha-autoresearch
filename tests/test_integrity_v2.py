"""Regression tests for cross-symbol contamination and invalid frontier updates."""
import json
import numpy as np
import pandas as pd
import pytest
from prepare import Factor, _make_forward_return, compute_turnover_stability, ops, pareto_decision, update_archive


@pytest.fixture
def panel():
    dates = pd.date_range("2025-01-01", periods=4)
    return pd.DataFrame([(d, symbol, float(value))
                         for symbol, values in (("A", [10, 20, 40, 80]),
                                                ("B", [100, 110, 121, 133.1]))
                         for d, value in zip(dates, values)],
                        columns=["datetime", "symbol", "close"])


def test_forward_return_stays_within_symbol(panel):
    returns = _make_forward_return(panel)
    assert returns.loc[(pd.Timestamp("2025-01-01"), "A")] == pytest.approx(1.0)
    assert returns.loc[(pd.Timestamp("2025-01-03"), "A")] == pytest.approx(1.0)
    assert returns.loc[(pd.Timestamp("2025-01-01"), "B")] == pytest.approx(.1)
    assert np.isnan(returns.loc[(pd.Timestamp("2025-01-04"), "A")])
    assert np.isnan(returns.loc[(pd.Timestamp("2025-01-04"), "B")])


def test_multiday_returns_stay_within_symbol(panel):
    returns = _make_forward_return(panel, horizon=2)
    assert returns.loc[(pd.Timestamp("2025-01-01"), "A")] == pytest.approx(3.0)
    assert returns.loc[(pd.Timestamp("2025-01-01"), "B")] == pytest.approx(.21)
    assert np.isnan(returns.loc[(pd.Timestamp("2025-01-03"), "A")])


def test_time_series_windows_do_not_mix_stocks(panel):
    ordered = panel.sort_values(["datetime", "symbol"])
    index = pd.MultiIndex.from_frame(ordered[["datetime", "symbol"]])
    values = pd.Series(ordered.close.to_numpy(), index=index)
    assert ops.delay(values).loc[(pd.Timestamp("2025-01-02"), "A")] == 10
    assert ops.rolling_sum(values, 2).loc[(pd.Timestamp("2025-01-02"), "B")] == 210
    assert ops.rolling_cov(values, values, 2).loc[(pd.Timestamp("2025-01-02"), "A")] == pytest.approx(25)
    assert ops.ts_rank(values, 2).loc[(pd.Timestamp("2025-01-02"), "B")] == 1.0
    shuffled = values.sample(frac=1, random_state=9)
    pd.testing.assert_series_equal(ops.rolling_sum(values, 2).sort_index(),
                                   ops.rolling_sum(shuffled, 2).sort_index())


def test_factor_output_is_aligned_by_key(panel):
    ordered = panel.sort_values(["datetime", "symbol"]).reset_index(drop=True)
    index = pd.MultiIndex.from_frame(ordered[["datetime", "symbol"]])
    values = pd.Series(np.arange(len(index), dtype=float), index=index).iloc[::-1]
    assert Factor.as_cs_series(ordered, values).iloc[0] == 0
    wrong = pd.Series(np.arange(len(index)), index=pd.MultiIndex.from_tuples(
        list(index[:-1]) + [(pd.Timestamp("2020-01-01"), "X")], names=index.names))
    with pytest.raises(ValueError, match="keys"):
        Factor.as_cs_series(ordered, wrong)


def test_pareto_discards_if_any_member_dominates(tmp_path):
    path = tmp_path / "frontier.json"
    path.write_text(json.dumps({"frontier": [
        {"name": "strong", "rank_ic": .06, "ic_ir": 1.5, "turnover_stability": .9},
        {"name": "unrelated", "rank_ic": .02, "ic_ir": 1., "turnover_stability": .99}
    ], "dominated_count": 0, "total_experiments": 2}))
    candidate = {"rank_ic": .04, "ic_ir": 1., "turnover_stability": .8}
    assert pareto_decision("candidate", candidate, str(path)) == ("discard", [], ["strong"])


def test_archive_deduplicates_and_writes_atomically(tmp_path):
    path = tmp_path / "frontier.json"
    factor = {"name": "candidate", "rank_ic": .04, "ic_ir": 1., "turnover_stability": .8}
    update_archive(factor, str_path=str(path))
    with pytest.raises(ValueError, match="already exists"):
        update_archive(factor, str_path=str(path))
    assert json.loads(path.read_text())["total_experiments"] == 1


def test_single_day_turnover_is_not_artificially_perfect(panel):
    first = panel[panel.datetime == pd.Timestamp("2025-01-01")]
    index = pd.MultiIndex.from_frame(first[["datetime", "symbol"]])
    assert np.isnan(compute_turnover_stability(pd.Series([1., 2.], index=index)))
