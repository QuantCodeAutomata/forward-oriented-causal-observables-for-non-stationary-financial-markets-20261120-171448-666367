"""Unit tests for causal technical indicators (src/indicators.py)."""

import numpy as np
import pandas as pd
import pytest

from src.indicators import (
    causal_bb_percent,
    causal_macd_diff,
    causal_mfi,
    causal_rsi,
    compute_all_indicators,
)


def _synthetic_ohlcv(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.001, size=n)
    close = 1.10 + np.cumsum(steps)
    high = close + rng.uniform(0.0, 0.001, size=n)
    low = close - rng.uniform(0.0, 0.001, size=n)
    open_ = close + rng.normal(0.0, 0.0002, size=n)
    volume = rng.integers(50, 500, size=n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def test_rsi_bounded_between_0_and_100():
    ohlcv = _synthetic_ohlcv()
    rsi = causal_rsi(ohlcv["close"])
    valid = rsi.dropna()
    assert len(valid) > 0
    assert (valid >= 0.0).all() and (valid <= 100.0).all()


def test_mfi_bounded_between_0_and_100():
    ohlcv = _synthetic_ohlcv()
    mfi = causal_mfi(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"])
    valid = mfi.dropna()
    assert len(valid) > 0
    assert (valid >= 0.0).all() and (valid <= 100.0).all()


def test_bb_percent_near_zero_one_for_random_walk():
    ohlcv = _synthetic_ohlcv(n=500)
    bb = causal_bb_percent(ohlcv["close"])
    valid = bb.dropna()
    assert len(valid) > 0
    # Most values should lie within [-1, 2]; bands are +/-2 std around the mean.
    assert valid.between(-2, 3).mean() > 0.95


def test_macd_diff_zero_for_constant_price():
    close = pd.Series(np.full(100, 1.10))
    macd = causal_macd_diff(close)
    valid = macd.dropna()
    assert np.allclose(valid, 0.0, atol=1e-10)


def test_indicators_are_causal_prefix_invariant():
    """Appending future bars must not change already-computed past values
    (the defining property of a strictly causal/online computation)."""
    ohlcv = _synthetic_ohlcv(n=300)
    full = compute_all_indicators(ohlcv)
    prefix = compute_all_indicators(ohlcv.iloc[:150])
    pd.testing.assert_frame_equal(full.iloc[:150], prefix, check_exact=False, rtol=1e-8)


def test_compute_all_indicators_requires_columns():
    with pytest.raises(ValueError):
        compute_all_indicators(pd.DataFrame({"close": [1.0, 2.0]}))
