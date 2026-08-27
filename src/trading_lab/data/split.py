from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DataSplit:
    name: str
    start: str
    end: str


DEVELOPMENT = DataSplit(
    name="development",
    start="2017-01-01",
    end="2022-12-31",
)

VALIDATION = DataSplit(
    name="validation",
    start="2023-01-01",
    end="2024-12-31",
)

HOLDOUT = DataSplit(
    name="holdout",
    start="2025-01-01",
    end="2025-12-31",
)


def _filter_dates(
    df: pd.DataFrame,
    split: DataSplit,
) -> pd.DataFrame:
    """
    Return only rows inside the requested chronological split.
    """

    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must contain a timestamp column.")

    timestamps = pd.to_datetime(df["timestamp"], utc=True)

    start = pd.Timestamp(split.start, tz="UTC")
    end = pd.Timestamp(split.end, tz="UTC") + pd.Timedelta(days=1)

    mask = (timestamps >= start) & (timestamps < end)

    return df.loc[mask].copy().reset_index(drop=True)


def get_development_data(df: pd.DataFrame) -> pd.DataFrame:
    return _filter_dates(df, DEVELOPMENT)


def get_validation_data(df: pd.DataFrame) -> pd.DataFrame:
    return _filter_dates(df, VALIDATION)


def get_holdout_data(
    df: pd.DataFrame,
    *,
    parameters_frozen: bool = False,
) -> pd.DataFrame:
    """
    Holdout access is blocked until strategy parameters are frozen.
    """

    if not parameters_frozen:
        raise PermissionError(
            "Holdout data is sealed. "
            "Set parameters_frozen=True only after strategy "
            "parameters have been finalized."
        )

    return _filter_dates(df, HOLDOUT)