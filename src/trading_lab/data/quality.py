from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {
    "symbol",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
}


def validate_bars(df: pd.DataFrame) -> dict:
    """
    Run basic integrity checks on historical OHLCV data.
    """

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    duplicate_count = int(
        df.duplicated(["symbol", "timestamp"]).sum()
    )

    null_counts = (
        df[list(REQUIRED_COLUMNS)]
        .isna()
        .sum()
        .to_dict()
    )

    invalid_ohlc = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )

    nonpositive_prices = (
        (df[["open", "high", "low", "close"]] <= 0)
        .any(axis=1)
    )

    negative_volume = df["volume"] < 0

    report = {
        "rows": len(df),
        "symbols": sorted(df["symbol"].unique().tolist()),
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
        "duplicates": duplicate_count,
        "nulls": null_counts,
        "invalid_ohlc_rows": int(invalid_ohlc.sum()),
        "nonpositive_price_rows": int(nonpositive_prices.sum()),
        "negative_volume_rows": int(negative_volume.sum()),
    }

    report["passed"] = (
        duplicate_count == 0
        and sum(null_counts.values()) == 0
        and report["invalid_ohlc_rows"] == 0
        and report["nonpositive_price_rows"] == 0
        and report["negative_volume_rows"] == 0
    )

    return report