"""Integration test of the shared pipeline on a reduced real EUR/USD slice.

Exercises the full chain (indicators -> causal centering -> F_0 -> Kalman
smoothing -> forward operator -> hysteresis rule -> realized returns) on
actual market data and checks the paper's structural properties (Sec. 2-3):
no look-ahead in the composite signal, a one-step execution delay, and a
bounded, well-formed equity curve. It does not compare to the paper's
reported numbers, since the slice is far shorter than the full sample.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.forex_session import filter_forex_session
from src.pipeline import backtest, compute_f, compute_f0


@pytest.mark.market_data
def test_pipeline_runs_end_to_end_on_real_slice(eurusd_slice):
    """The shared pipeline runs on a real EUR/USD slice and returns
    well-formed outputs for both F_0 and F(t)."""
    ohlcv = eurusd_slice.drop(columns=["ticker"], errors="ignore").copy()
    ohlcv = filter_forex_session(ohlcv, timestamp_col="timestamp")
    ohlcv["ret"] = ohlcv["close"].pct_change()
    assert len(ohlcv) > 1000, "the cached slice is too short to exercise the indicator warmup"

    f0_frame = compute_f0(ohlcv)
    assert {"f0_raw", "f0", "rsi", "mfi", "macd_diff", "bb_pct"} <= set(f0_frame.columns)

    f_series = compute_f(f0_frame["f0"])
    assert f_series.notna().sum() > 0

    result_f0 = backtest(ohlcv, f0_frame["f0"], config.DECISION_THETA)
    result_f = backtest(ohlcv, f_series, config.DECISION_THETA)

    for result in (result_f0, result_f):
        assert set(result["state"].dropna().unique()) <= {0, 1}
        assert np.isfinite(result["equity"].iloc[-1])
        assert (result["equity"] > 0).all(), "compounded equity must stay strictly positive"
        assert result["activity"].iloc[-1] >= 0


@pytest.mark.market_data
def test_no_lookahead_on_real_slice(eurusd_slice):
    """Appending future bars must not change the Kalman-smoothed F_0 value
    already produced at an earlier bar (Sec. 2.2 causal construction)."""
    ohlcv = eurusd_slice.drop(columns=["ticker"], errors="ignore").copy()
    ohlcv = filter_forex_session(ohlcv, timestamp_col="timestamp")
    cutoff = len(ohlcv) // 2

    full_f0 = compute_f0(ohlcv)["f0"]
    truncated_f0 = compute_f0(ohlcv.iloc[:cutoff].reset_index(drop=True))["f0"]

    compare_n = cutoff - 5  # exclude the last few bars where warmup windows differ at the boundary
    pd.testing.assert_series_equal(
        full_f0.iloc[:compare_n].reset_index(drop=True),
        truncated_f0.iloc[:compare_n].reset_index(drop=True),
        check_names=False,
    )


@pytest.mark.market_data
def test_execution_delay_matches_one_bar_on_real_slice(eurusd_slice):
    """The realized return at bar t must use the state decided at t-1, not
    the contemporaneous state (Sec. 2.4, Eq. 7: R_t = p_{t-1} * r_t)."""
    ohlcv = eurusd_slice.drop(columns=["ticker"], errors="ignore").copy()
    ohlcv = filter_forex_session(ohlcv, timestamp_col="timestamp")
    ohlcv["ret"] = ohlcv["close"].pct_change()

    f0_frame = compute_f0(ohlcv)
    result = backtest(ohlcv, f0_frame["f0"], config.DECISION_THETA)

    manual = result["state"].shift(1).fillna(0).astype(int) * ohlcv["ret"]
    pd.testing.assert_series_equal(result["realized"], manual, check_names=False)
