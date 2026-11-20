"""Unit tests for performance-metric reporting (src/performance.py)."""

import numpy as np
import pandas as pd

from src.performance import max_drawdown_pct, subperiod_metrics, summary_table, trades_per_month


def test_max_drawdown_is_nonpositive():
    equity = pd.Series([1.0, 1.2, 0.9, 1.1, 0.8, 1.3])
    mdd = max_drawdown_pct(equity)
    assert mdd <= 0.0
    # Peak 1.2 -> trough 0.8: drop of 1/3 = -33.33%.
    assert np.isclose(mdd, (0.8 / 1.2 - 1.0) * 100.0)


def test_max_drawdown_zero_for_monotonic_equity():
    equity = pd.Series([1.0, 1.1, 1.2, 1.3])
    assert np.isclose(max_drawdown_pct(equity), 0.0)


def test_trades_per_month_scales_with_elapsed_time():
    index = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
    activity = pd.Series(np.arange(60, dtype=float))  # 59 trades total
    rate = trades_per_month(activity, pd.DatetimeIndex(index))
    # 59 trades over ~59 days ~ 1 trade/day -> ~30.4 trades/month.
    assert 25.0 < rate < 35.0


def test_summary_table_end_equity_and_return():
    equity = pd.Series([1.0, 1.2, 1.5])
    summary = summary_table(equity)
    assert np.isclose(summary["end_equity"], 1.5)
    assert np.isclose(summary["cum_return_pct"], 50.0)
    assert summary["max_drawdown_pct"] <= 0.0


def test_subperiod_metrics_renormalizes_equity_to_one():
    timestamps = pd.Series(pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"))
    realized = pd.Series(np.full(10, 0.01))
    state = pd.Series(np.ones(10, dtype=int))
    metrics = subperiod_metrics(
        timestamps, realized, state, "sub1", timestamps.iloc[0], timestamps.iloc[-1]
    )
    assert metrics["n_bars"] == 9
    expected_end_equity = 1.01 ** 9
    assert np.isclose(metrics["end_equity"], expected_end_equity)


def test_subperiod_metrics_empty_window_returns_nan():
    timestamps = pd.Series(pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"))
    realized = pd.Series(np.zeros(5))
    state = pd.Series(np.zeros(5, dtype=int))
    metrics = subperiod_metrics(
        timestamps, realized, state, "empty",
        pd.Timestamp("2030-01-01", tz="UTC"), pd.Timestamp("2030-02-01", tz="UTC"),
    )
    assert metrics["n_bars"] == 0
    assert np.isnan(metrics["end_equity"])
