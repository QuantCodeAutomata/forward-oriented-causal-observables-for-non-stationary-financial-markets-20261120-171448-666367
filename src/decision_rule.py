"""Two-state hysteresis decision rule and causal execution (Sec. 2.4, Eqs. 6-7).

# Custom implementation (paper Eqs. 6-7): the hysteresis state p_t depends on
# p_{t-1}, a first-order recursion with no vectorized equivalent in pandas/
# numpy and no match among the backtesting libraries surveyed (vectorbt signal
# generators assume stateless threshold crossings, not path-dependent
# hysteresis), so it is implemented as an explicit forward loop.
"""

import numpy as np
import pandas as pd

from src.config import DECISION_THETA, EXECUTION_DELAY


def hysteresis_state(signal: pd.Series, theta: float = DECISION_THETA) -> pd.Series:
    """Two-state hysteresis rule, Eq. 6.

    p_t = 1 if p_{t-1}=0 and s_t > theta
    p_t = 0 if p_{t-1}=1 and s_t < -theta
    p_t = p_{t-1} otherwise

    The initial state p_0 is 0 (flat) since no position can be held before
    any signal has been observed.
    """
    s = signal.to_numpy(dtype=float)
    n = s.shape[0]
    state = np.zeros(n, dtype=int)
    current = 0
    for t in range(n):
        value = s[t]
        if np.isnan(value):
            state[t] = current
            continue
        if current == 0 and value > theta:
            current = 1
        elif current == 1 and value < -theta:
            current = 0
        state[t] = current
    return pd.Series(state, index=signal.index, name="state")


def realized_returns(
    state: pd.Series, simple_returns: pd.Series, delay: int = EXECUTION_DELAY
) -> pd.Series:
    """Realized strategy response R_t = p_{t-delay} * r_t, Eq. 7.

    The one-step execution delay enforces strict causality: the decision
    formed from information up to t only affects the return realized from
    t to t+1 (applied here as a shift of the state series).
    """
    delayed_state = state.shift(delay).fillna(0).astype(int)
    return delayed_state * simple_returns


def trading_activity(state: pd.Series) -> pd.Series:
    """Cumulative trading activity: cumulative sum of |Delta p_t|."""
    delta = state.diff().abs().fillna(0.0)
    return delta.cumsum()


def compounded_equity(realized: pd.Series) -> pd.Series:
    """Compounded equity V_t = prod_{tau=1}^{t} (1 + R_tau), with V_0 = 1."""
    return (1.0 + realized.fillna(0.0)).cumprod()
