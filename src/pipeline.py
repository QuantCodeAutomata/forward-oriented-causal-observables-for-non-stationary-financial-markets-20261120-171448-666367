"""Shared end-to-end pipeline: data loading, observable construction, and
backtesting, used identically by both experiments (Sec. 2-3 of the paper).
"""

from __future__ import annotations

import pandas as pd

from data.data_loader import load_ohlcv
from src.config import TICKER, TIMEFRAME
from src.decision_rule import compounded_equity, hysteresis_state, realized_returns, trading_activity
from src.forex_session import filter_forex_session
from src.indicators import compute_all_indicators
from src.performance import subperiod_metrics, summary_table
from src.signal_construction import build_f0, forward_operator


def load_price_data(
    start: str, end: str, timeframe: str = TIMEFRAME, use_cache: bool = True
) -> pd.DataFrame:
    """Load EUR/USD OHLCV bars, restrict to the active Forex session, and
    attach the simple return r_t = (P_t - P_{t-1}) / P_{t-1} (Sec. 2.4).

    Returns a DataFrame sorted by timestamp with columns timestamp, open,
    high, low, close, volume, ret.
    """
    bars, meta = load_ohlcv(
        [TICKER], start, end, timeframe=timeframe, yahoo_backstop=False, use_cache=use_cache
    )
    bars = bars.drop(columns=["ticker"], errors="ignore")
    bars = filter_forex_session(bars, timestamp_col="timestamp")
    bars["ret"] = bars["close"].pct_change()
    bars.attrs["load_meta"] = meta
    return bars


def compute_f0(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Indicators, causally centered composite F_0(t), and its Kalman-smoothed
    version (Sec. 2.2, Eqs. 1-3; Appendix B).

    Returns a DataFrame indexed like ``ohlcv`` with columns rsi, mfi,
    macd_diff, bb_pct, f0_raw, f0.
    """
    indicators = compute_all_indicators(ohlcv)
    f0_frame = build_f0(indicators)
    return pd.concat([indicators, f0_frame], axis=1)


def compute_f(f0_smoothed: pd.Series, f0_raw: pd.Series) -> pd.Series:
    """Forward-oriented observable F(t) (Sec. 2.3, Eqs. 4-5): mixing
    coefficients and the F_0(t) term use the Kalman-smoothed composite
    ``f0_smoothed``; the causal finite-difference derivative is taken of the
    raw composite ``f0_raw`` (see ``forward_operator`` docstring)."""
    return forward_operator(f0_smoothed, f0_raw)


def backtest(ohlcv: pd.DataFrame, signal: pd.Series, theta: float) -> dict:
    """Run the hysteresis decision rule on ``signal`` and compute realized
    strategy performance against buy-and-hold (Sec. 2.4, Eqs. 6-7).

    Returns a dict with keys: state, realized, equity, buy_hold_equity,
    activity. ``equity`` and ``buy_hold_equity`` are compounded V_t series
    with V_0 = 1; ``activity`` is the cumulative count of state changes.
    """
    state = hysteresis_state(signal, theta)
    realized = realized_returns(state, ohlcv["ret"])
    equity = compounded_equity(realized)
    buy_hold_equity = compounded_equity(ohlcv["ret"])
    activity = trading_activity(state)
    return {
        "state": state,
        "realized": realized,
        "equity": equity,
        "buy_hold_equity": buy_hold_equity,
        "activity": activity,
    }


def subperiod_report(
    timestamps: pd.Series,
    realized: pd.Series,
    state: pd.Series,
    splits: list[tuple[str, pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame:
    """Build the Table 1 style subperiod report: end equity, cumulative
    return %, max drawdown %, trades/month, for each (label, start, end)
    window, with equity re-normalized to V_0 = 1 at the start of the window.
    """
    rows = [
        subperiod_metrics(timestamps, realized, state, label, start, end)
        for label, start, end in splits
    ]
    return pd.DataFrame(rows)


def whole_sample_report(equity: pd.Series) -> dict:
    """Whole-sample end equity, cumulative return %, and max drawdown %."""
    return summary_table(equity)
