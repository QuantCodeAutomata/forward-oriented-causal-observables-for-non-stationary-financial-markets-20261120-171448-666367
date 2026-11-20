"""Shared pytest fixtures: marker registration and a cached market-data slice."""

import os

import pandas as pd
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "market_data: test needs a live QuantX Dataloader call (skipped without DATALOADER_API_KEY)"
    )


@pytest.fixture(scope="session")
def eurusd_slice(request):
    """A reduced EUR/USD 1-minute slice loaded once per test session and cached
    on disk between runs (pytest cache), so repeated test runs do not
    re-download it. A few weeks is enough to exercise the full pipeline
    (indicator warmup windows are at most 26 bars) while keeping tests fast.
    """
    if not os.environ.get("DATALOADER_API_KEY") or not os.environ.get("MASSIVE_API_KEY"):
        pytest.skip("DATALOADER_API_KEY / MASSIVE_API_KEY not set; skipping market_data test")

    cache_dir = request.config.cache.mkdir("eurusd_slice")
    cache_file = cache_dir / "eurusd_1m_2024H1_slice.csv"
    if cache_file.exists():
        bars = pd.read_csv(cache_file)
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
        return bars

    from data.data_loader import load_ohlcv

    bars, _ = load_ohlcv(
        ["C:EURUSD"], "2024-01-02", "2024-02-15", timeframe="1m", yahoo_backstop=False
    )
    bars.to_csv(cache_file, index=False)
    return bars
