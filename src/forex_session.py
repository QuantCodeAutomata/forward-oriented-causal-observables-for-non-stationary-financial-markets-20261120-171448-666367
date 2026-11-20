"""Active Forex trading-hour filter (Sec. 3, paragraph preceding Figure 3).

"Saturdays are removed entirely, as well as Sundays before 18:00 and Fridays
after 18:00 (local time)." This keeps indicators computed from contiguous
market activity, avoiding weekend-closure distortions.
"""

import pandas as pd

from src.config import FOREX_SESSION_CUTOFF_HOUR, FOREX_SESSION_TZ


def filter_forex_session(ohlcv: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Drop bars outside the active Forex session (paper Sec. 3).

    Parameters
    ----------
    ohlcv : DataFrame with a timezone-aware UTC timestamp column.
    timestamp_col : name of the timestamp column.

    Returns
    -------
    DataFrame restricted to active trading hours, sorted by timestamp, with
    the original index reset.
    """
    if timestamp_col not in ohlcv.columns:
        raise ValueError(f"ohlcv is missing the '{timestamp_col}' column")

    local_time = pd.to_datetime(ohlcv[timestamp_col], utc=True).dt.tz_convert(FOREX_SESSION_TZ)
    weekday = local_time.dt.weekday  # Monday=0 ... Sunday=6
    hour = local_time.dt.hour

    is_saturday = weekday == 5
    is_sunday_before_open = (weekday == 6) & (hour < FOREX_SESSION_CUTOFF_HOUR)
    is_friday_after_close = (weekday == 4) & (hour >= FOREX_SESSION_CUTOFF_HOUR)

    keep = ~(is_saturday | is_sunday_before_open | is_friday_after_close)
    filtered = ohlcv.loc[keep].sort_values(timestamp_col).reset_index(drop=True)
    return filtered
