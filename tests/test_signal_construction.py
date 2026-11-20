"""Unit tests for F_0(t) construction and Kalman stabilization (src/signal_construction.py)."""

import numpy as np
import pandas as pd

from src.signal_construction import (
    build_f0,
    causal_median_center,
    composite_f0_raw,
    forward_operator,
    kalman_1d_filter,
)


def test_causal_median_center_uses_only_past():
    series = pd.Series([1.0, 2.0, 3.0, 100.0, 5.0])
    centered = causal_median_center(series)
    # centered[3] must not be affected by series[3] itself (100.0): the
    # median of {1,2,3} = 2.0, so centered[3] = 100 - 2 = 98.
    assert np.isclose(centered.iloc[3], 100.0 - 2.0)
    # First element has no past data: NaN.
    assert pd.isna(centered.iloc[0])


def test_causal_median_center_prefix_invariant():
    series = pd.Series(np.random.default_rng(1).normal(size=50))
    full = causal_median_center(series)
    prefix = causal_median_center(series.iloc[:30])
    pd.testing.assert_series_equal(full.iloc[:30], prefix, check_exact=False)


def test_kalman_filter_converges_to_constant_observation():
    z = pd.Series(np.full(100, 0.05))
    x = kalman_1d_filter(z)
    assert np.isclose(x.iloc[0], 0.05)
    assert np.isclose(x.iloc[-1], 0.05, atol=1e-6)


def test_kalman_filter_initial_state_equals_first_observation():
    z = pd.Series([0.3, 0.31, 0.29, 0.32])
    x = kalman_1d_filter(z)
    assert np.isclose(x.iloc[0], z.iloc[0])


def test_kalman_filter_smooths_noise():
    rng = np.random.default_rng(2)
    z = pd.Series(0.1 + rng.normal(0.0, 0.5, size=500))
    x = kalman_1d_filter(z)
    assert x.diff().abs().mean() < z.diff().abs().mean()


def test_kalman_filter_prefix_invariant():
    z = pd.Series(np.random.default_rng(3).normal(size=80))
    full = kalman_1d_filter(z)
    prefix = kalman_1d_filter(z.iloc[:40])
    np.testing.assert_allclose(full.iloc[:40].to_numpy(), prefix.to_numpy(), atol=1e-12)


def test_composite_f0_raw_combines_four_indicators():
    n = 100
    indicators = pd.DataFrame(
        {
            "rsi": np.full(n, 50.0),
            "mfi": np.full(n, 50.0),
            "macd_diff": np.zeros(n),
            "bb_pct": np.full(n, 0.5),
        }
    )
    f0 = composite_f0_raw(indicators)
    # With constant indicators the causal median centers to zero after warm-up.
    assert np.isclose(f0.iloc[-1], 0.0, atol=1e-8)


def test_build_f0_returns_raw_and_smoothed_columns():
    rng = np.random.default_rng(4)
    n = 60
    indicators = pd.DataFrame(
        {
            "rsi": 50 + rng.normal(0, 5, n),
            "mfi": 50 + rng.normal(0, 5, n),
            "macd_diff": rng.normal(0, 1e-4, n),
            "bb_pct": 0.5 + rng.normal(0, 0.1, n),
        }
    )
    f0 = build_f0(indicators)
    assert set(f0.columns) == {"f0_raw", "f0"}
    assert len(f0) == n


def test_forward_operator_reduces_to_zero_derivative_when_f0_constant():
    f0 = pd.Series(np.full(50, 0.1))
    f = forward_operator(f0)
    # Derivative is zero for a constant series, so F(t) = c1(t) * F0(t).
    expected_c1 = np.tanh(0.1)
    valid = f.dropna()
    assert np.allclose(valid, expected_c1 * 0.1, atol=1e-10)


def test_forward_operator_prefix_invariant():
    rng = np.random.default_rng(5)
    f0 = pd.Series(rng.normal(0, 0.1, size=120))
    full = forward_operator(f0)
    prefix = forward_operator(f0.iloc[:60])
    pd.testing.assert_series_equal(full.iloc[:60], prefix, check_exact=False)
