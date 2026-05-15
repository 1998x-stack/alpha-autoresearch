# factors.py — Agent edit surface for alpha_autoresearch
# Each Factor* class is auto-discovered and evaluated.
# Available operators: ops.ts_rank, ops.rolling_corr, ops.rolling_cov,
#   ops.cs_rank, ops.delta, ops.delay, ops.decay_linear, ops.rolling_std,
#   ops.rolling_min, ops.rolling_max, ops.rolling_sum, ops.cs_zscore
# Available columns: open, high, low, close, volume, vwap, returns,
#   adv5, adv10, adv20, adv30, adv40, adv60, adv120, adv150, adv180

from prepare import Factor, ops


class Factor001(Factor):
    name = "ts_rank_returns"

    def compute(self, df):
        m = df.set_index(["datetime", "symbol"])
        # TS rank of recent returns — analogous to ts_rank_vol but on returns
        val = -ops.ts_rank(m["returns"], 5)
        return Factor.as_cs_series(df, val)


class Factor002(Factor):
    name = "decay_returns"

    def compute(self, df):
        m = df.set_index(["datetime", "symbol"])
        # Decay-linear weighted returns over 20 days — capture short-term drift
        val = ops.decay_linear(m["returns"], 20)
        return Factor.as_cs_series(df, val)


class Factor003(Factor):
    name = "range_vol_adj"

    def compute(self, df):
        m = df.set_index(["datetime", "symbol"])
        # Volume-adjusted price range — penalizes illiquid stocks
        vol_ratio = m["volume"] / m["adv20"].replace(0, float("nan"))
        vol_scale = ops.rolling_std(vol_ratio, 20).replace(0, float("nan"))
        val = (m["high"] - m["low"]) / (m["close"] * vol_scale + 0.001)
        return Factor.as_cs_series(df, val)


class Factor004(Factor):
    name = "sharpe_ret"

    def compute(self, df):
        m = df.set_index(["datetime", "symbol"])
        # Sharpe-like: decay-weighted returns / long-term volatility
        signal = ops.decay_linear(m["returns"], 10)
        risk = ops.rolling_std(m["returns"], 60).replace(0, float("nan"))
        val = signal / (risk + 0.001)
        return Factor.as_cs_series(df, val)


class Factor005(Factor):
    name = "adv_term_struct"

    def compute(self, df):
        m = df.set_index(["datetime", "symbol"])
        # Volume term structure: short-term vs long-term avg volume
        short = m["adv5"]
        long = m["adv60"].replace(0, float("nan"))
        val = ops.cs_rank(short / long)
        return Factor.as_cs_series(df, val)


class Factor006(Factor):
    name = "close_open_decay"

    def compute(self, df):
        m = df.set_index(["datetime", "symbol"])
        # Smoothed intraday return — decay-weighted close/open ratio change
        intraday_ret = (m["close"] - m["open"]) / m["open"].replace(0, float("nan"))
        val = ops.decay_linear(intraday_ret, 10)
        return Factor.as_cs_series(df, val)


class Factor007(Factor):
    name = "cs_zscore_vol"

    def compute(self, df):
        m = df.set_index(["datetime", "symbol"])
        # Cross-sectional z-score of volume — different distribution than cs_rank
        val = ops.cs_zscore(m["volume"])
        return Factor.as_cs_series(df, val)


class Factor008(Factor):
    name = "combo_range_vwap"

    def compute(self, df):
        m = df.set_index(["datetime", "symbol"])
        # Combines range signal with VWAP deviation — two proven signals
        range_sig = ops.cs_rank((m["high"] - m["low"]) / m["close"].replace(0, float("nan")))
        vwap_sig = ops.cs_rank(m["close"] - m["vwap"])
        val = range_sig * vwap_sig
        return Factor.as_cs_series(df, val)


class Factor009(Factor):
    name = "hl_spread_chg"

    def compute(self, df):
        m = df.set_index(["datetime", "symbol"])
        # Change in high-low spread over 5 days — range expansion/contraction
        spread = (m["high"] - m["low"]) / m["close"].replace(0, float("nan"))
        val = ops.delta(spread, 5)
        return Factor.as_cs_series(df, val)


class Factor010(Factor):
    name = "open_vwap_dev"

    def compute(self, df):
        m = df.set_index(["datetime", "symbol"])
        # Open price deviation from VWAP — gap signal
        val = ops.cs_rank(m["open"] - m["vwap"])
        return Factor.as_cs_series(df, val)
