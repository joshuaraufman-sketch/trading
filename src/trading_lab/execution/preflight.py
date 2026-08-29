"""
Pre-run safety checks.

The first piece of the operational layer, and the one that matters most
for a system that trades while its owner is at work: confirm you are
about to act on the account you think you are, in the state you think it
is in.

This exists because the paper account was replaced mid-project. Alpaca
offers no reset, so a new account was created and credentials swapped.
Nothing in the codebase noticed. Order counts and the forward-test clock
silently began spanning two accounts with different starting equity and
unrelated positions.

Every check operates on a plain ``AccountState``, so the logic is
testable without credentials or a network call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from trading_lab.execution.alpaca_account import AccountState


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str
    blocking: bool = True


@dataclass
class PreflightResult:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if c.blocking)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and not c.blocking]


@dataclass(frozen=True)
class AccountExpectations:
    """
    What the account should look like before trading.

    ``expected_account_id`` empty means "not yet pinned". The preflight
    reports the observed id so it can be recorded, and warns rather than
    blocking. Once set, a mismatch is fatal.
    """

    expected_account_id: str = ""
    universe: tuple[str, ...] = ()
    minimum_equity: float = 0.0
    require_paper: bool = True
    allow_open_orders: bool = False

    @classmethod
    def from_yaml(cls, path: Path) -> "AccountExpectations":
        with Path(path).open("r", encoding="utf-8") as file:
            payload = yaml.safe_load(file) or {}

        known = set(cls.__dataclass_fields__)
        unknown = set(payload) - known

        if unknown:
            raise ValueError(
                f"unknown account expectation keys: {sorted(unknown)}"
            )

        if "universe" in payload and payload["universe"] is not None:
            payload["universe"] = tuple(payload["universe"])

        return cls(**payload)



def normalize_status(value: str) -> str:
    """
    Reduce "AccountStatus.ACTIVE" or "ACTIVE" to "ACTIVE".

    Applied at the check rather than trusting the caller, so a raw
    stringified enum from anywhere still compares correctly.
    """

    text = str(value or "").strip()

    if "." in text:
        text = text.rsplit(".", 1)[-1]

    return text.upper()


def run_preflight(
    state: AccountState,
    expectations: AccountExpectations,
) -> PreflightResult:
    """
    Check an account is safe and expected before any order is placed.
    """

    result = PreflightResult()

    result.checks.append(
        CheckResult(
            "paper mode",
            passed=state.is_paper or not expectations.require_paper,
            detail=(
                "paper account confirmed"
                if state.is_paper
                else "NOT A PAPER ACCOUNT - real money at risk"
            ),
        )
    )

    result.checks.append(
        CheckResult(
            "trading enabled",
            passed=not state.trading_blocked,
            detail=(
                "trading blocked on this account"
                if state.trading_blocked
                else "trading permitted"
            ),
        )
    )

    # Normalized defensively: alpaca-py enums stringify as
    # "AccountStatus.ACTIVE", and any future field could arrive the same
    # way. Belt and braces is appropriate for a check whose false
    # positives train people to skip it.
    status = normalize_status(state.status)

    result.checks.append(
        CheckResult(
            "account active",
            passed=status in {"ACTIVE", ""},
            detail=f"status: {status or 'unknown'}",
        )
    )

    if expectations.expected_account_id:
        matches = state.account_id == expectations.expected_account_id
        result.checks.append(
            CheckResult(
                "account identity",
                passed=matches,
                detail=(
                    f"matches pinned account {state.account_id}"
                    if matches
                    else (
                        f"MISMATCH - credentials point at "
                        f"{state.account_id or 'unknown'}, expected "
                        f"{expectations.expected_account_id}"
                    )
                ),
            )
        )
    else:
        result.checks.append(
            CheckResult(
                "account identity",
                passed=False,
                detail=(
                    f"not pinned. Observed id {state.account_id or '?'} "
                    f"- record it in config/account.yaml"
                ),
                blocking=False,
            )
        )

    universe = set(expectations.universe)
    held = {p["symbol"] for p in state.positions}
    unexpected = sorted(held - universe) if universe else sorted(held)

    result.checks.append(
        CheckResult(
            "no unexpected positions",
            passed=not unexpected,
            detail=(
                f"holdings outside the strategy universe: {unexpected}"
                if unexpected
                else f"{len(held)} position(s), all within universe"
            ),
        )
    )

    result.checks.append(
        CheckResult(
            "no stale open orders",
            passed=(
                not state.open_orders or expectations.allow_open_orders
            ),
            detail=(
                f"{len(state.open_orders)} open order(s) outstanding"
                if state.open_orders
                else "no open orders"
            ),
        )
    )

    result.checks.append(
        CheckResult(
            "sufficient equity",
            passed=state.equity >= expectations.minimum_equity,
            detail=(
                f"equity ${state.equity:,.2f} "
                f"(minimum ${expectations.minimum_equity:,.2f})"
            ),
        )
    )

    # Informational: under $25k a margin account is capped at three day
    # trades per five business days. A daily-rebalancing strategy can
    # trip this without anyone noticing.
    result.checks.append(
        CheckResult(
            "pattern day trader flag",
            passed=not state.pattern_day_trader,
            detail=(
                "flagged as a pattern day trader"
                if state.pattern_day_trader
                else "not flagged"
            ),
            blocking=False,
        )
    )

    return result
