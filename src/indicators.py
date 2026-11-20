"""Causal (online) technical indicators (paper Appendix A, Eqs. A.1-A.4).

Every indicator below uses only information available up to the current bar:
Wilder/exponential smoothing recursions and rolling windows draw exclusively
on past and present observations, so no indicator value at time t depends on
data at time > t.

# Custom implementation (paper Appendix A): the 'ta' package's RSI/MFI/MACD/
# Bollinger routines compute the same classical formulas, but they operate on
# an already-materialized vector; re-implemented here with pandas' causal
# rolling/ewm primitives to keep full control over the recursion (Wilder
# smoothing choice, EMA adjust=False) and to make causality easy to test.
"""

import numpy as np
import pandas as pd

from src.config import (
    BB_NUM_STD,
    BB_WINDOW,
    MACD_FAST_PERIOD,
    MACD_SIGNAL_PERIOD,
    MACD_SLOW_PERIOD,
    MFI_PERIOD,
    RSI_PERIOD,
)


def causal_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Relative Strength Index with Wilder smoothing (Eq. A.1).

    G_t and L_t are Wilder-smoothed (exponential, alpha=1/period) averages of
    gains and losses, matching the original definition in Wilder (1978) cited
    by the paper. Only past price changes up to t enter the recursion.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    # Avoid division by zero; when avg_loss == 0, RSI saturates at 100.
    ratio = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 * (1.0 - 1.0 / (1.0 + ratio))
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    return rsi


def causal_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = MFI_PERIOD,
) -> pd.Series:
    """Money Flow Index (Eq. A.2), volume-weighted momentum indicator.

    Typical price TP_t = (H_t+L_t+C_t)/3, raw money flow RMF_t = TP_t*V_t.
    Positive/negative money flow are summed over a trailing window of
    ``period`` bars according to the sign of TP_t - TP_{t-1}; all quantities
    at time t use only bars up to and including t.
    """
    typical_price = (high + low + close) / 3.0
    raw_money_flow = typical_price * volume
    price_change = typical_price.diff()
    positive_flow = raw_money_flow.where(price_change > 0, 0.0)
    negative_flow = raw_money_flow.where(price_change < 0, 0.0)
    positive_sum = positive_flow.rolling(window=period, min_periods=period).sum()
    negative_sum = negative_flow.rolling(window=period, min_periods=period).sum()
    money_flow_ratio = positive_sum / negative_sum.replace(0.0, np.nan)
    mfi = 100.0 * (1.0 - 1.0 / (1.0 + money_flow_ratio))
    mfi = mfi.where(negative_sum != 0.0, 100.0)
    return mfi


def causal_macd_diff(
    close: pd.Series,
    fast_period: int = MACD_FAST_PERIOD,
    slow_period: int = MACD_SLOW_PERIOD,
    signal_period: int = MACD_SIGNAL_PERIOD,
) -> pd.Series:
    """MACD difference (histogram), Eq. A.3: (EMA_fast - EMA_slow) - Signal."""
    ema_fast = close.ewm(span=fast_period, adjust=False, min_periods=fast_period).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False, min_periods=slow_period).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(
        span=signal_period, adjust=False, min_periods=signal_period
    ).mean()
    return macd_line - signal_line


def causal_bb_percent(
    close: pd.Series, window: int = BB_WINDOW, num_std: float = BB_NUM_STD
) -> pd.Series:
    """Bollinger Band percent (Eq. A.4): location of price within the bands.

    mu_t and sigma_t are the trailing rolling mean/standard deviation over
    ``window`` bars up to and including t.
    """
    rolling_mean = close.rolling(window=window, min_periods=window).mean()
    rolling_std = close.rolling(window=window, min_periods=window).std(ddof=0)
    lower_band = rolling_mean - num_std * rolling_std
    upper_band = rolling_mean + num_std * rolling_std
    band_width = (upper_band - lower_band).replace(0.0, np.nan)
    return (close - lower_band) / band_width


def compute_all_indicators(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute RSI, MFI, MACD difference, and BB% from an OHLCV frame.

    Parameters
    ----------
    ohlcv : DataFrame with columns 'open', 'high', 'low', 'close', 'volume',
        sorted by increasing time.

    Returns
    -------
    DataFrame with columns 'rsi', 'mfi', 'macd_diff', 'bb_pct' aligned to
    ``ohlcv``'s index.
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(ohlcv.columns)
    if missing:
        raise ValueError(f"ohlcv is missing required columns: {missing}")
    return pd.DataFrame(
        {
            "rsi": causal_rsi(ohlcv["close"]),
            "mfi": causal_mfi(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]),
            "macd_diff": causal_macd_diff(ohlcv["close"]),
            "bb_pct": causal_bb_percent(ohlcv["close"]),
        },
        index=ohlcv.index,
    )
