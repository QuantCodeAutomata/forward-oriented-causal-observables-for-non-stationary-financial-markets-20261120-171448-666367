"""Unit tests for the hysteresis decision rule and causal execution
(src/decision_rule.py, paper Sec. 2.4, Eqs. 6-7)."""

import numpy as np
import pandas as pd

from src.decision_rule import (
    compounded_equity,
    hysteresis_state,
    realized_returns,
    trading_activity,
)


def test_hysteresis_enters_long_above_theta():
    signal = pd.Series([0.0, 0.01, 0.07, 0.08, 0.02])
    state = hysteresis_state(signal, theta=0.06)
    assert list(state) == [0, 0, 1, 1, 1]


def test_hysteresis_exits_long_below_minus_theta():
    signal = pd.Series([0.07, 0.07, 0.0, -0.07, -0.01])
    state = hysteresis_state(signal, theta=0.06)
    assert list(state) == [1, 1, 1, 0, 0]


def test_hysteresis_holds_state_inside_band():
    signal = pd.Series([0.07, 0.05, -0.05, 0.03])
    state = hysteresis_state(signal, theta=0.06)
    # Once long (t=0), stays long while |s_t| < theta.
    assert list(state) == [1, 1, 1, 1]


def test_hysteresis_nan_holds_previous_state():
    signal = pd.Series([0.07, np.nan, np.nan, -0.07])
    state = hysteresis_state(signal, theta=0.06)
    assert list(state) == [1, 1, 1, 0]


def test_realized_returns_applies_one_step_delay():
    state = pd.Series([0, 1, 1, 0])
    returns = pd.Series([0.01, 0.02, -0.01, 0.03])
    realized = realized_returns(state, returns, delay=1)
    # R_t = p_{t-1} * r_t: p_{-1} undefined -> 0 by construction.
    expected = pd.Series([0.0, 0.0 * 0.02, 1.0 * -0.01, 1.0 * 0.03])
    pd.testing.assert_series_equal(realized, expected, check_names=False)


def test_realized_returns_flat_state_gives_zero_pnl():
    state = pd.Series(np.zeros(10, dtype=int))
    returns = pd.Series(np.random.default_rng(0).normal(size=10))
    realized = realized_returns(state, returns)
    assert np.allclose(realized, 0.0)


def test_trading_activity_counts_state_changes():
    state = pd.Series([0, 0, 1, 1, 0, 1])
    activity = trading_activity(state)
    # Three transitions: 0->1, 1->0, 0->1.
    assert activity.iloc[-1] == 3


def test_compounded_equity_starts_at_one_and_compounds():
    realized = pd.Series([0.0, 0.1, -0.1, 0.05])
    equity = compounded_equity(realized)
    assert np.isclose(equity.iloc[0], 1.0)
    expected_last = 1.0 * 1.1 * 0.9 * 1.05
    assert np.isclose(equity.iloc[-1], expected_last)


def test_compounded_equity_matches_buy_and_hold_when_always_long():
    simple_returns = pd.Series(np.random.default_rng(1).normal(0, 0.001, size=50))
    always_long = pd.Series(np.ones(50, dtype=int))
    realized = realized_returns(always_long, simple_returns, delay=0)
    equity = compounded_equity(realized)
    buy_and_hold = compounded_equity(simple_returns)
    pd.testing.assert_series_equal(equity, buy_and_hold, check_names=False)
