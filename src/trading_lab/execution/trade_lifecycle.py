from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRADE_DIR = PROJECT_ROOT / "reports" / "forward_trades"


def _now_utc() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def _trade_path(
    entry_order_id: str,
    *,
    trade_dir: str | Path = TRADE_DIR,
) -> Path:
    if not entry_order_id:
        raise ValueError(
            "entry_order_id is required"
        )

    return (
        Path(trade_dir)
        / f"{entry_order_id}.json"
    )


def create_trade_record(
    *,
    entry_order_id: str,
    symbol: str,
    strategy_name: str,
    signal_date: str,
    signal_time: str,
    reference_price: float,
    quantity: float,
    planned_stop_price: float,
    holding_days: int,
    trade_dir: str | Path = TRADE_DIR,
) -> Path:
    if not symbol:
        raise ValueError("symbol is required")

    if reference_price <= 0:
        raise ValueError(
            "reference_price must be positive"
        )

    if quantity <= 0:
        raise ValueError(
            "quantity must be positive"
        )

    if planned_stop_price <= 0:
        raise ValueError(
            "planned_stop_price must be positive"
        )

    if holding_days <= 0:
        raise ValueError(
            "holding_days must be positive"
        )

    path = _trade_path(
        entry_order_id,
        trade_dir=trade_dir,
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        raise FileExistsError(
            f"Trade record already exists: {path}"
        )

    record: dict[str, Any] = {
        "schema_version": 1,
        "entry_order_id": entry_order_id,
        "symbol": symbol,
        "strategy_name": strategy_name,
        "signal_date": signal_date,
        "signal_time": signal_time,
        "status": "submitted",
        "created_at_utc": _now_utc(),
        "reference_price": reference_price,
        "quantity_planned": quantity,
        "planned_stop_price": (
            planned_stop_price
        ),
        "holding_days": holding_days,
        "entry": {
            "status": "submitted",
            "filled_qty": 0.0,
            "filled_avg_price": None,
            "filled_at_utc": None,
        },
        "exit": {
            "order_id": None,
            "reason": None,
            "status": None,
            "filled_qty": 0.0,
            "filled_avg_price": None,
            "filled_at_utc": None,
        },
        "realized_pnl": None,
        "realized_return_pct": None,
    }

    path.write_text(
        json.dumps(
            record,
            indent=2,
        ),
        encoding="utf-8",
    )

    return path


def load_trade_record(
    entry_order_id: str,
    *,
    trade_dir: str | Path = TRADE_DIR,
) -> dict[str, Any]:
    path = _trade_path(
        entry_order_id,
        trade_dir=trade_dir,
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def update_entry_fill(
    *,
    entry_order_id: str,
    status: str,
    filled_qty: float,
    filled_avg_price: float | None,
    trade_dir: str | Path = TRADE_DIR,
) -> Path:
    if filled_qty < 0:
        raise ValueError(
            "filled_qty cannot be negative"
        )

    path = _trade_path(
        entry_order_id,
        trade_dir=trade_dir,
    )

    record = load_trade_record(
        entry_order_id,
        trade_dir=trade_dir,
    )

    record["entry"]["status"] = status
    record["entry"]["filled_qty"] = (
        filled_qty
    )
    record["entry"]["filled_avg_price"] = (
        filled_avg_price
    )

    if (
        filled_qty > 0
        and filled_avg_price is not None
    ):
        record["entry"]["filled_at_utc"] = (
            _now_utc()
        )
        record["status"] = "open"

    record["updated_at_utc"] = _now_utc()

    path.write_text(
        json.dumps(
            record,
            indent=2,
        ),
        encoding="utf-8",
    )

    return path


def close_trade_record(
    *,
    entry_order_id: str,
    exit_order_id: str,
    exit_reason: str,
    filled_qty: float,
    filled_avg_price: float,
    trade_dir: str | Path = TRADE_DIR,
) -> Path:
    if filled_qty <= 0:
        raise ValueError(
            "filled_qty must be positive"
        )

    if filled_avg_price <= 0:
        raise ValueError(
            "filled_avg_price must be positive"
        )

    record = load_trade_record(
        entry_order_id,
        trade_dir=trade_dir,
    )

    entry_price = record["entry"][
        "filled_avg_price"
    ]

    if entry_price is None:
        raise ValueError(
            "Entry must be filled before "
            "the trade can be closed."
        )

    realized_pnl = (
        filled_avg_price - entry_price
    ) * filled_qty

    realized_return_pct = (
        filled_avg_price / entry_price - 1
    ) * 100

    record["exit"] = {
        "order_id": exit_order_id,
        "reason": exit_reason,
        "status": "filled",
        "filled_qty": filled_qty,
        "filled_avg_price": (
            filled_avg_price
        ),
        "filled_at_utc": _now_utc(),
    }

    record["status"] = "closed"
    record["realized_pnl"] = realized_pnl
    record["realized_return_pct"] = (
        realized_return_pct
    )
    record["updated_at_utc"] = _now_utc()

    path = _trade_path(
        entry_order_id,
        trade_dir=trade_dir,
    )

    path.write_text(
        json.dumps(
            record,
            indent=2,
        ),
        encoding="utf-8",
    )

    return path
