from __future__ import annotations

import hashlib
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

    # Second resolution collides: a parameter sweep runs dozens of
    # experiments per second, and each collision silently overwrote the
    # previous record while the summary CSV kept pointing at the path.
    # The parameter digest makes the filename a function of what was
    # actually run, so identical filenames now mean identical runs.
    digest = hashlib.sha256(
        json.dumps(
            {
                "strategy": strategy_name,
                "dataset": dataset,
                "parameters": parameters,
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:10]

    filename = (
        f"{timestamp}_"
        f"{strategy_name}_"
        f"{digest}.json"
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
