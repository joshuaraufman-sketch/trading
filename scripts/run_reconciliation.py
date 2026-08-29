"""
Evening reconciliation: did the account end up where the plan intended?

Reads the most recent daily run log, the current account state, and the
broker's own order record, then reports any gap between intended and
actual exposure.

This is the check that makes the system safe to leave alone during the
day. Without it, a rejected or partially filled order leaves the account
holding something other than what the strategy decided, and nothing
notices.

    python scripts/run_reconciliation.py

Exit code 0 means clean, 1 means something needs attention -- so a
scheduler can alert on it.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from trading_lab.execution.alpaca_account import get_account_state
from trading_lab.execution.equity_baseline import (
    account_fingerprint,
    load_or_create_baseline,
)
from trading_lab.execution.position_check import (
    classify_orders,
    reconcile_positions,
    summarize_reconciliation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT_ROOT / "reports" / "daily_runs"
RECON_DIR = PROJECT_ROOT / "reports" / "reconciliation"
BASELINE_PATH = PROJECT_ROOT / "state" / "equity_baseline.json"


def latest_run() -> dict | None:
    if not RUN_DIR.exists():
        return None

    paths = sorted(RUN_DIR.glob("*_daily_run.json"))

    if not paths:
        return None

    with paths[-1].open("r", encoding="utf-8") as file:
        record = json.load(file)

    record["_path"] = paths[-1].name
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tolerance", type=float, default=0.02)
    args = parser.parse_args()

    run = latest_run()

    if run is None:
        print("No daily run found. Nothing to reconcile.")
        sys.exit(0)

    account_state = get_account_state()

    baseline, _ = load_or_create_baseline(
        BASELINE_PATH,
        account_id=account_state.account_id,
        equity=account_state.equity,
    )

    print("EVENING RECONCILIATION")
    print("=" * 66)
    print(f"run log            {run['_path']}")
    print(f"session            {run.get('session')}")
    print(f"submitted          {run.get('submitted')}")
    print(f"account            "
          f"{account_fingerprint(account_state.account_id)}")
    print(f"equity vs baseline "
          f"{baseline.ratio(account_state.equity):.4f}x")

    logged = run.get("account_fingerprint", "")
    current = account_fingerprint(account_state.account_id)

    if logged and logged != current:
        print()
        print("ACCOUNT MISMATCH: the run log was written against a")
        print(f"different account ({logged}). Reconciling the wrong")
        print("account would produce meaningless results.")
        sys.exit(1)

    planned = run.get("orders", [])

    broker_orders = [
        {
            "symbol": order["symbol"],
            "status": order["status"],
            "qty": order.get("qty"),
            "filled_qty": order.get("filled_qty"),
        }
        for order in account_state.open_orders
    ]

    order_issues = (
        classify_orders(planned, broker_orders)
        if run.get("submitted")
        else []
    )

    prices = {
        position["symbol"]: (
            position["market_value"] / position["qty"]
            if position["qty"]
            else 0.0
        )
        for position in account_state.positions
    }

    reference = run.get("diagnostics", {}).get("reference_price")

    for symbol in run.get("target_weights", {}):
        if symbol not in prices and reference:
            prices[symbol] = float(reference)

    report = reconcile_positions(
        intended_weights=run.get("target_weights", {}),
        actual_quantities={
            p["symbol"]: p["qty"] for p in account_state.positions
        },
        prices=prices,
        equity=account_state.equity,
        submitted=bool(run.get("submitted")),
        order_issues=order_issues,
        tolerance=args.tolerance,
    )

    print()
    print(summarize_reconciliation(report))

    RECON_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = RECON_DIR / f"{stamp}_reconciliation.json"

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "checked_utc": datetime.now(timezone.utc).isoformat(),
                "run_log": run["_path"],
                "session": run.get("session"),
                "account_fingerprint": current,
                "equity_ratio": baseline.ratio(account_state.equity),
                "clean": report.clean,
                "order_issues": report.order_issues,
                "discrepancies": [
                    {
                        "symbol": d.symbol,
                        "intended_weight": d.intended_weight,
                        "actual_weight": d.actual_weight,
                        "drift": d.drift,
                        "severity": d.severity,
                        "likely_cause": d.likely_cause,
                    }
                    for d in report.discrepancies
                ],
            },
            file, indent=2, default=str,
        )

    print()
    print(f"saved: {path.relative_to(PROJECT_ROOT)}")

    sys.exit(0 if report.clean else 1)


if __name__ == "__main__":
    main()
