"""Unit tests for the active Forex trading-hour filter (src/forex_session.py)."""

import pandas as pd
import pytest

from src.forex_session import filter_forex_session


def _bars(timestamps_utc):
    n = len(timestamps_utc)
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps_utc, utc=True),
            "open": range(n),
            "high": range(n),
            "low": range(n),
            "close": range(n),
            "volume": range(n),
        }
    )


def test_saturday_rows_removed():
    # 2024-01-06 is a Saturday.
    bars = _bars(["2024-01-05T12:00:00Z", "2024-01-06T12:00:00Z", "2024-01-07T20:00:00Z"])
    filtered = filter_forex_session(bars)
    kept_dates = pd.to_datetime(filtered["timestamp"]).dt.date.astype(str).tolist()
    assert "2024-01-06" not in kept_dates


def test_sunday_before_open_removed_after_open_kept():
    # 2024-01-07 is a Sunday; cutoff is 18:00 America/New_York (UTC-5 in Jan).
    # 2024-01-07T17:00:00 UTC-5 => 22:00 UTC, i.e. after the 18:00 local open.
    bars = _bars([
        "2024-01-07T15:00:00Z",  # 10:00 local Sunday -> before open, dropped
        "2024-01-07T23:30:00Z",  # 18:30 local Sunday -> after open, kept
    ])
    filtered = filter_forex_session(bars)
    assert len(filtered) == 1
    kept_hour_utc = pd.to_datetime(filtered["timestamp"]).dt.hour.iloc[0]
    assert kept_hour_utc == 23


def test_friday_after_close_removed_before_close_kept():
    # 2024-01-05 is a Friday; 18:00 local (UTC-5) = 23:00 UTC.
    bars = _bars([
        "2024-01-05T20:00:00Z",  # 15:00 local Friday -> before close, kept
        "2024-01-06T00:30:00Z",  # 19:30 local Friday -> after close, dropped
    ])
    filtered = filter_forex_session(bars)
    assert len(filtered) == 1
    kept_hour_utc = pd.to_datetime(filtered["timestamp"]).dt.hour.iloc[0]
    assert kept_hour_utc == 20


def test_weekday_rows_all_kept():
    bars = _bars(["2024-01-02T10:00:00Z", "2024-01-03T14:00:00Z", "2024-01-04T09:00:00Z"])
    filtered = filter_forex_session(bars)
    assert len(filtered) == 3


def test_missing_timestamp_column_raises():
    with pytest.raises(ValueError):
        filter_forex_session(pd.DataFrame({"close": [1.0, 2.0]}))
