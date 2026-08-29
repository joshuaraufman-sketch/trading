"""
Confirm the account is safe and expected before anything trades.

Run this after any credential change, and as the first step of any
scheduled run. It places no orders and makes no modifications.

    python scripts/preflight_check.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trading_lab.execution.alpaca_account import get_account_state
from trading_lab.execution.preflight import (
    AccountExpectations,
    run_preflight,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACCOUNT_PATH = PROJECT_ROOT / "config" / "account.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quiet", action="store_true",
        help="Print only failures. Suitable for a scheduled job.",
    )
    args = parser.parse_args()

    expectations = AccountExpectations.from_yaml(ACCOUNT_PATH)
    state = get_account_state()
    result = run_preflight(state, expectations)

    if not args.quiet:
        print("PREFLIGHT CHECK")
        print("=" * 66)
        print(f"account id       {state.account_id or 'unknown'}")
        print(f"account number   {state.account_number or 'unknown'}")
        print(f"equity           ${state.equity:,.2f}")
        print(f"cash             ${state.cash:,.2f}")
        print(f"buying power     ${state.buying_power:,.2f}")
        print(f"positions        {len(state.positions)}")
        print(f"open orders      {len(state.open_orders)}")

        if state.positions:
            print()
            for position in state.positions:
                print(
                    f"  {position['symbol']:<8}"
                    f"{position['qty']:>12,.4f} sh"
                    f"{position['market_value']:>14,.2f}"
                )

        print()
        print("CHECKS")
        print("-" * 66)

        for check in result.checks:
            if check.passed:
                mark = "PASS"
            elif check.blocking:
                mark = "FAIL"
            else:
                mark = "WARN"

            print(f"  {mark:<6}{check.name:<26}{check.detail}")

        print()

    if result.passed:
        if not args.quiet:
            print("PREFLIGHT PASSED - safe to proceed")

        if not expectations.expected_account_id:
            print()
            print("Account identity is not yet pinned. Record it now:")
            print()
            print(f'  expected_account_id: "{state.account_id}"')
            print()
            print("in config/account.yaml. Until then a credential swap")
            print("will not be detected.")
    else:
        print("PREFLIGHT FAILED - do not trade")
        print()
        for check in result.failures:
            if check.blocking:
                print(f"  {check.name}: {check.detail}")
        sys.exit(1)


if __name__ == "__main__":
    main()
