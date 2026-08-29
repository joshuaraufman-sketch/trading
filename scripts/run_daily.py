"""
Post-close daily run.

Computes the target weight from today's close and queues orders for
tomorrow's open. This matches the backtest exactly: a weight knowable at
the close of t-1 is what is held into session t.

DRY RUN IS THE DEFAULT. Nothing is submitted unless --submit is passed.
Get in the habit of running without it first.

    python scripts/run_daily.py                 # decide and report only
    python scripts/run_daily.py --submit        # actually place orders

Intended schedule: shortly after the 16:00 ET close, on trading days.
Run it before the close and the stale-bar check will refuse, which is
the correct behaviour rather than an inconvenience.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from trading_lab.data.alpaca import get_daily_bars
from trading_lab.execution.alpaca_account import get_account_state
from trading_lab.execution.daily_runner import (
    StrategyConfig,
    build_daily_plan,
    summarize_plan,
)
from trading_lab.execution.equity_baseline import (
    account_fingerprint,
    load_or_create_baseline,
    normalize_order,
)
from trading_lab.execution.preflight import AccountExpectations
from trading_lab.execution.rebalance import ExecutionPolicy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACCOUNT_PATH = PROJECT_ROOT / "config" / "account.yaml"
POLICY_PATH = PROJECT_ROOT / "config" / "execution_policy.yaml"
STRATEGY_PATH = PROJECT_ROOT / "config" / "live_strategy.yaml"
RUN_DIR = PROJECT_ROOT / "reports" / "daily_runs"
# Local only, gitignored. The single place a dollar figure lives.
BASELINE_PATH = PROJECT_ROOT / "state" / "equity_baseline.json"

NEW_YORK = ZoneInfo("America/New_York")


def load_strategy() -> StrategyConfig:
    with STRATEGY_PATH.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}

    unknown = set(payload) - set(StrategyConfig.__dataclass_fields__)

    if unknown:
        raise ValueError(f"unknown strategy keys: {sorted(unknown)}")

    return StrategyConfig(**payload)


def completed_sessions() -> set:
    """
    Sessions already processed, read from the run log.

    Idempotency matters for a scheduled job: a retry after a transient
    failure must not double-submit.
    """

    done = set()

    if not RUN_DIR.exists():
        return done

    for path in RUN_DIR.glob("*_daily_run.json"):
        try:
            with path.open("r", encoding="utf-8") as file:
                record = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue

        if not record.get("submitted"):
            continue

        session = record.get("session")

        if session:
            done.add(datetime.fromisoformat(session).date())

    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submit", action="store_true",
        help="Actually place orders. Omit for a dry run.",
    )
    parser.add_argument(
        "--session", default=None,
        help="Override the session date (YYYY-MM-DD). For testing.",
    )
    args = parser.parse_args()

    expectations = AccountExpectations.from_yaml(ACCOUNT_PATH)
    policy = ExecutionPolicy.from_yaml(POLICY_PATH)
    strategy = load_strategy()

    now_et = datetime.now(NEW_YORK)

    if args.session:
        session = datetime.fromisoformat(args.session).date()
    else:
        session = now_et.date()

    print("DAILY RUN" + ("" if args.submit else "  (DRY RUN)"))
    print("=" * 66)
    print(f"now (ET)           {now_et:%Y-%m-%d %H:%M}")
    print(f"trading session    {session}")
    print(f"strategy           {strategy.symbol} vol-target "
          f"{strategy.target_volatility:.0%} / {strategy.lookback}d")

    if now_et.hour < 16 and not args.session:
        print()
        print("NOTE: it is before the 16:00 ET close. Today's bar is")
        print("incomplete and the stale-bar check will likely refuse.")

    account_state = get_account_state()

    baseline, baseline_created = load_or_create_baseline(
        BASELINE_PATH,
        account_id=account_state.account_id,
        equity=account_state.equity,
    )
    equity_ratio = baseline.ratio(account_state.equity)

    print(f"account            {account_fingerprint(account_state.account_id)}")
    print(f"equity vs baseline {equity_ratio:.4f}x"
          + ("  (baseline established this run)" if baseline_created else ""))

    bars = get_daily_bars(
        symbols=[strategy.symbol],
        start="2024-01-01",
        end=session.isoformat(),
    )

    plan = build_daily_plan(
        account_state=account_state,
        bars=bars,
        expected_session=session,
        expectations=expectations,
        policy=policy,
        strategy=strategy,
        completed_sessions=completed_sessions(),
    )

    print()
    print(summarize_plan(plan))

    submitted = []

    if plan.blocked:
        exit_code = 1
    elif not plan.should_trade:
        exit_code = 0
    elif not args.submit:
        print()
        print("DRY RUN - nothing submitted. Re-run with --submit to place.")
        exit_code = 0
    else:
        from trading_lab.execution.alpaca_orders import (
            submit_paper_market_order,
        )

        print()
        print("SUBMITTING")
        print("-" * 66)

        for order in plan.rebalance.orders:
            result = submit_paper_market_order(
                symbol=order.symbol,
                quantity=order.quantity,
                side=order.side,
            )
            submitted.append(
                {
                    "symbol": order.symbol,
                    "side": order.side,
                    "notional_pct": (
                        order.notional / account_state.equity
                    ),
                    "accepted": result is not None,
                }
            )
            print(f"  {order.side.upper():<5}{order.symbol:<8}"
                  f"{order.quantity:>12,.4f} sh  submitted")

        exit_code = 0

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = RUN_DIR / f"{stamp}_daily_run.json"

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "session": session.isoformat(),
                "run_time_utc": datetime.now(timezone.utc).isoformat(),
                # Stamped so a credential swap can never again leave
                # history silently spanning two accounts.
                # Fingerprint, not the id: detects a credential swap
                # just as well without publishing the identifier.
                "account_fingerprint": account_fingerprint(
                    account_state.account_id
                ),
                # Ratio, not dollars. This log is public.
                "equity_ratio": equity_ratio,
                "baseline_reset": baseline_created,
                "submitted": bool(submitted),
                "dry_run": not args.submit,
                "blocked_reason": plan.blocked_reason,
                "target_weights": plan.target_weights,
                "diagnostics": plan.diagnostics,
                # Share counts and dollar notionals are omitted: they
                # state the account size as plainly as an equity line.
                # Alpaca's own order history is the source of truth for
                # fills and is not public.
                "orders": [
                    normalize_order(o, account_state.equity)
                    for o in (
                        plan.rebalance.orders if plan.rebalance else []
                    )
                ],
                "skipped": (
                    plan.rebalance.skipped if plan.rebalance else {}
                ),
                "submitted_orders": submitted,
                "preflight": [
                    {
                        "name": c.name,
                        "passed": c.passed,
                        "detail": c.detail,
                        "blocking": c.blocking,
                    }
                    for c in (
                        plan.preflight.checks if plan.preflight else []
                    )
                ],
            },
            file, indent=2, default=str,
        )

    print()
    print(f"run logged: {path.relative_to(PROJECT_ROOT)}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
