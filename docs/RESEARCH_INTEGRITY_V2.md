# Research integrity v2: migration and validation

## Why this change is necessary

The original `_make_forward_return` applied `shift(-horizon)` to the **entire** grouped-return vector, causing returns at stock boundaries to be assigned to the wrong stock. The original time-series operators applied rolling windows and delays directly to the date-major MultiIndex, mixing observations of different stocks. The original Pareto decision rejected a candidate only when *every* current frontier member dominated it, rather than when *any* member dominated it. These errors invalidate comparison of old and corrected RankIC/IC IR values.

## Changes

- Forward returns use next-price lookup *within each stock*, even for multi-day horizons.
- Rolling, delay, delta, covariance, correlation, time-series rank and decay operate separately within chronological stock histories. Plain Series retain the existing behavior.
- Factor output is checked for duplicate and mismatched keys, and matched by `(datetime, symbol)` rather than silently matched by position.
- A candidate dominated by any archived member is discarded; archive updates are revalidated and written through atomic replacement.
- Missing turnover transitions return NaN rather than an artificial perfect 1.0. Non-finite metric vectors are rejected.
- New results are saved to `pareto_frontier.v2.json`; the original `pareto_frontier.json` is preserved for historical reference only.

## Verification

Run `uv run pytest tests/ -q` and `uv run python prepare.py --no-archive` before creating the v2 archive. The regression tests specifically cover stock-boundary forward returns, multi-day horizons, shuffled rows, rolling windows, keyed factor alignment, dominance and duplicate archive entries. For the full 495-stock experiment, configure the original data source, rebuild its panel, and rerun experiments with the revised evaluator; historical rankings and published charts are not verified by these unit tests.

## Important remaining work

This is an evaluator correctness patch, **not** a claim of realistic out-of-sample trading performance. Add point-in-time constituents and corporate-action-adjusted inputs, publication-time feature availability, suspension/delisting handling, train/validation/test separation, transaction cost and capacity assumptions, and per-experiment provenance before using factors as trading signals. `run_loop.py` still requires an independent hardening pass for subprocess exit status, immutable experiment snapshots, idempotent resume, and robust result serialization; its hand-written factor list is not a general autonomous model-driven researcher. Generated factors are executable Python, so run untrusted generations in an isolated environment with filesystem/network limits rather than relying on the Unix alarm alone.

This patch intentionally preserves the old dataset and factor definitions and does not rewrite historical metrics as if they had been recomputed.
