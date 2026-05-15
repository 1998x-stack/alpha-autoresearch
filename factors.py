# factors.py — Agent edit surface for alpha_autoresearch
# Each Factor* class is auto-discovered and evaluated.
# Available operators: ops.ts_rank, ops.rolling_corr, ops.rolling_cov,
#   ops.cs_rank, ops.delta, ops.delay, ops.decay_linear, ops.rolling_std,
#   ops.rolling_min, ops.rolling_max, ops.rolling_sum, ops.cs_zscore
# Available columns: open, high, low, close, volume, vwap, returns,
#   adv5, adv10, adv20, adv30, adv40, adv60, adv120, adv150, adv180

from prepare import Factor, ops


class Factor001(Factor):
    name = "decay_range"

    def compute(self, df):
        m = df.set_index(["datetime", "symbol"])
        val = ops.decay_linear((m["high"] - m["low"]) / m["close"], 10)
        return Factor.as_cs_series(df, val)
