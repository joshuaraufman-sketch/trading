from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")


def _get_client() -> StockHistoricalDataClient:
    """
    Create an authenticated Alpaca historical-data client.
    """

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise RuntimeError(
            "Alpaca credentials were not found. "
            "Create a local .env file containing "
            "ALPACA_API_KEY and ALPACA_SECRET_KEY."
        )

    return StockHistoricalDataClient(
        api_key,
        secret_key,
    )


def get_daily_bars(
    symbols: str | Iterable[str],
    start: str,
    end: str,
    *,
    use_cache: bool = True,
    feed: DataFeed = DataFeed.SIP,
) -> pd.DataFrame:
    """
    Retrieve adjusted daily bars from Alpaca.

    SIP remains the default for historical research.
    IEX can be supplied for recent/live signal generation.
    """

    if isinstance(symbols, str):
        symbols = [symbols]

    symbols = sorted(
        {
            symbol.upper()
            for symbol in symbols
        }
    )

    cache_name = (
        f"{'-'.join(symbols)}_"
        f"{start}_"
        f"{end}_"
        f"1Day_{feed.value}_adjusted.csv"
    )

    cache_path = CACHE_DIR / cache_name

    if use_cache and cache_path.exists():
        return pd.read_csv(
            cache_path,
            parse_dates=["timestamp"],
        )

    client = _get_client()

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=pd.Timestamp(
            start,
            tz="America/New_York",
        ),
        end=pd.Timestamp(
            end,
            tz="America/New_York",
        ),
        feed=feed,
        adjustment=Adjustment.ALL,
    )

    bars = client.get_stock_bars(
        request
    )

    df = bars.df.reset_index()

    if df.empty:
        raise ValueError(
            f"Alpaca returned no data for "
            f"{symbols} between {start} and {end}."
        )

    expected_columns = [
        "symbol",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "vwap",
    ]

    available = [
        column
        for column in expected_columns
        if column in df.columns
    ]

    df = df[available].copy()

    df["symbol"] = (
        df["symbol"]
        .astype(str)
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = (
        df
        .sort_values(
            ["symbol", "timestamp"]
        )
        .drop_duplicates(
            ["symbol", "timestamp"]
        )
        .reset_index(drop=True)
    )

    df.to_csv(
        cache_path,
        index=False,
    )

    return df