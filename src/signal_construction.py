"""Causal composite observable F_0(t) and forward-oriented operator F(t).

Implements paper Sec. 2.2 (causal centering, linear aggregation, Kalman
stabilization) and Sec. 2.3 (adaptive forward operator), Eqs. 1-5.
"""

import numpy as np
import pandas as pd

from src.config import (
    ALPHA_BB_PCT,
    ALPHA_MACD,
    ALPHA_MFI,
    ALPHA_RSI,
    DERIVATIVE_SMOOTHING_SPAN,
    FORWARD_DERIVATIVE_SCALE,
    KALMAN_P0,
    KALMAN_Q,
    KALMAN_R,
)


def causal_median_center(series: pd.Series) -> pd.Series:
    """Causal median centering, Eq. 1: tilde_I_t = I_t - median{I_tau : tau < t}.

    ``expanding().median()`` at position t uses observations up to and
    including t; shifting by one restricts the median to strictly-past
    observations (tau < t), as required by causality.
    """
    past_median = series.expanding(min_periods=1).median().shift(1)
    return series - past_median


def composite_f0_raw(indicators: pd.DataFrame) -> pd.Series:
    """Raw composite signal F_0(t), Eq. 3, from causally centered indicators.

    F_0(t) = (1/4) * (alpha_MFI*tilde_MFI + alpha_RSI*tilde_RSI
                       + alpha_BB%*tilde_BB% + alpha_MACD*tilde_MACD)
    """
    required = {"rsi", "mfi", "macd_diff", "bb_pct"}
    missing = required - set(indicators.columns)
    if missing:
        raise ValueError(f"indicators is missing required columns: {missing}")

    centered_rsi = causal_median_center(indicators["rsi"])
    centered_mfi = causal_median_center(indicators["mfi"])
    centered_macd = causal_median_center(indicators["macd_diff"])
    centered_bb = causal_median_center(indicators["bb_pct"])

    return (1.0 / 4.0) * (
        ALPHA_MFI * centered_mfi
        + ALPHA_RSI * centered_rsi
        + ALPHA_BB_PCT * centered_bb
        + ALPHA_MACD * centered_macd
    )


def kalman_1d_filter(
    observations: pd.Series,
    process_var: float = KALMAN_Q,
    measurement_var: float = KALMAN_R,
    initial_var: float = KALMAN_P0,
) -> pd.Series:
    """Forward-only scalar Kalman filter (paper Appendix B, Eqs. B.1-B.2).

    State model: x_t = x_{t-1} + w_t, w_t ~ N(0, process_var) (random walk).
    Observation model: z_t = x_t + v_t, v_t ~ N(0, measurement_var).
    Initialized at x_0 = z_0 (paper Sec. 2.2); no RTS (backward) smoother is
    applied, so x_t depends only on z_tau for tau <= t.

    # Custom implementation (paper Appendix B): a scalar random-walk Kalman
    # recursion has no vector/matrix structure to gain from a general filter
    # library (e.g. filterpy); implemented directly from the closed-form
    # scalar predict/update equations.
    """
    values = observations.to_numpy(dtype=float)
    n = values.shape[0]
    smoothed = np.full(n, np.nan)

    first_valid = np.argmax(~np.isnan(values)) if np.any(~np.isnan(values)) else None
    if first_valid is None:
        return pd.Series(smoothed, index=observations.index)

    x = values[first_valid]  # x_0 = z_0
    p = initial_var
    smoothed[first_valid] = x

    for t in range(first_valid + 1, n):
        z = values[t]
        if np.isnan(z):
            smoothed[t] = x
            continue
        # Predict step (random-walk state).
        x_pred = x
        p_pred = p + process_var
        # Update step.
        kalman_gain = p_pred / (p_pred + measurement_var)
        x = x_pred + kalman_gain * (z - x_pred)
        p = (1.0 - kalman_gain) * p_pred
        smoothed[t] = x

    return pd.Series(smoothed, index=observations.index)


def forward_operator(f0_smoothed: pd.Series, f0_raw: pd.Series) -> pd.Series:
    """Forward-oriented observable F(t), Eqs. 4-5.

    F(t) = c1(t)*F_0(t) + 2*c2(t)*smoothed_derivative(F_0(t))
    c1(t) = tanh(|F_0(t)|), c2(t) = 1 - tanh(|F_0(t)/2|)

    The causal finite-difference derivative is taken of the raw composite
    F_0(t) (``composite_f0_raw``), not its Kalman-smoothed version: Sec. 2.3
    motivates the derivative term as injecting "local forward-oriented
    structure", which a derivative of the already heavily-smoothed Kalman
    output would largely remove. A sensitivity check on a one-month slice
    confirms this: differencing the raw composite yields a monthly trade
    count (~1190/month) that matches the paper's reported order of magnitude
    for F (~959-1117 trades/month, Table 1), whereas differencing the
    Kalman-smoothed composite yields ~209/month, an order of magnitude too
    low. The mixing coefficients c1, c2 and the F_0(t) term itself still use
    the Kalman-smoothed composite, consistent with Fig. 1 and the decision
    rule of Sec. 2.4 in exp_1. Smoothed with a span-4 moving average (Eq. 4).
    """
    derivative = f0_raw.diff()
    smoothed_derivative = derivative.rolling(
        window=DERIVATIVE_SMOOTHING_SPAN, min_periods=1
    ).mean()

    c1 = np.tanh(f0_smoothed.abs())
    c2 = 1.0 - np.tanh((f0_smoothed / 2.0).abs())

    return c1 * f0_smoothed + FORWARD_DERIVATIVE_SCALE * c2 * smoothed_derivative


def build_f0(ohlcv_indicators: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline to F_0: raw composite and Kalman-smoothed composite.

    Returns a DataFrame with columns 'f0_raw' and 'f0' (Kalman-smoothed).
    """
    f0_raw = composite_f0_raw(ohlcv_indicators)
    f0_smoothed = kalman_1d_filter(f0_raw)
    return pd.DataFrame({"f0_raw": f0_raw, "f0": f0_smoothed})
