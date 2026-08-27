from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"


def save_experiment(
    *,
    strategy_name: str,
    parameters: dict[str, Any],
    metrics: dict[str, Any],
    evaluation: dict[str, Any],
    dataset: str,
) -> Path:
    """
    Save one research experiment as JSON.
    """

    EXPERIMENT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    filename = (
        f"{timestamp}_"
        f"{strategy_name}.json"
    )

    path = EXPERIMENT_DIR / filename

    record = {
        "experiment_time_utc": timestamp,
        "strategy": strategy_name,
        "dataset": dataset,
        "parameters": parameters,
        "metrics": metrics,
        "evaluation": evaluation,
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            record,
            file,
            indent=2,
            default=str,
        )

    return path