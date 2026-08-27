from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PENDING_DIR = PROJECT_ROOT / "reports" / "pending_signals"


def save_pending_signal(
    *,
    symbol: str,
    signal_time,
    signal_reference_price: float,
    strategy_name: str,
    parameters: dict[str, Any],
) -> Path:
    """
    Persist a signal for execution during the next
    eligible market-opening window.
    """

    if not symbol:
        raise ValueError("symbol is required")

    if signal_reference_price <= 0:
        raise ValueError(
            "signal_reference_price must be positive"
        )

    PENDING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    signal_date = (
        signal_time
        .to_pydatetime()
        .date()
        .isoformat()
        if hasattr(signal_time, "to_pydatetime")
        else signal_time.date().isoformat()
    )

    record = {
        "symbol": symbol,
        "signal_time": str(signal_time),
        "signal_date": signal_date,
        "signal_reference_price": (
            signal_reference_price
        ),
        "strategy_name": strategy_name,
        "parameters": parameters,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "pending",
    }

    path = (
        PENDING_DIR
        / f"{signal_date}_{symbol}.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            record,
            file,
            indent=2,
        )

    return path


def load_pending_signals() -> list[dict[str, Any]]:
    """
    Load all currently pending signals.
    """

    if not PENDING_DIR.exists():
        return []

    records = []

    for path in sorted(
        PENDING_DIR.glob("*.json")
    ):
        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                record = json.load(file)
        except (
            OSError,
            json.JSONDecodeError,
        ):
            continue

        if record.get("status") != "pending":
            continue

        record["_path"] = str(path)

        records.append(record)

    return records


def mark_signal_processed(
    path: str | Path,
    *,
    status: str,
    order_id: str | None = None,
) -> None:
    """
    Mark a pending signal as processed, blocked,
    expired, or submitted.
    """

    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        record = json.load(file)

    record["status"] = status
    record["processed_at_utc"] = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    if order_id is not None:
        record["order_id"] = order_id

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            record,
            file,
            indent=2,
        )