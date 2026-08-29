"""
Tests for pre-run account safety checks.

The account-swap tests are the point. Credentials were replaced
mid-project and nothing detected it; these make that class of mistake
loud rather than silent.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from trading_lab.execution.alpaca_account import AccountState
from trading_lab.execution.preflight import (
    AccountExpectations,
    normalize_status,
    run_preflight,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACCOUNT_PATH = PROJECT_ROOT / "config" / "account.yaml"


def _state(**overrides) -> AccountState:
    base = {
        "equity": 100_000.0,
        "cash": 100_000.0,
        "buying_power": 100_000.0,
        "positions": [],
        "open_orders": [],
        "account_id": "acct-new",
        "account_number": "PA123",
        "status": "ACTIVE",
        "pattern_day_trader": False,
        "trading_blocked": False,
        "is_paper": True,
    }
    base.update(overrides)
    return AccountState(**base)


def _expect(**overrides) -> AccountExpectations:
    base = {
        "expected_account_id": "acct-new",
        "universe": ("SPY",),
        "minimum_equity": 1000.0,
        "require_paper": True,
        "allow_open_orders": False,
    }
    base.update(overrides)
    return AccountExpectations(**base)


def test_clean_account_passes():
    assert run_preflight(_state(), _expect()).passed


def test_wrong_account_is_fatal():
    """
    The exact failure that occurred: credentials swapped to a different
    account and nothing noticed.
    """

    result = run_preflight(_state(account_id="acct-old"), _expect())

    assert not result.passed
    detail = next(
        c.detail for c in result.checks if c.name == "account identity"
    )
    assert "MISMATCH" in detail


def test_unpinned_identity_warns_but_does_not_block():
    result = run_preflight(_state(), _expect(expected_account_id=""))

    assert result.passed
    assert any(c.name == "account identity" for c in result.warnings)


def test_live_account_is_fatal():
    result = run_preflight(_state(is_paper=False), _expect())

    assert not result.passed
    detail = next(
        c.detail for c in result.checks if c.name == "paper mode"
    )
    assert "real money" in detail.lower()


def test_manual_positions_outside_universe_block_the_run():
    """
    The original account held manual BUD, QQQ and QQQM positions, which
    blocked signals and contaminated forward-test results for weeks.
    """

    contaminated = _state(
        positions=[
            {"symbol": "BUD", "qty": 1.0, "market_value": 60.0,
             "avg_entry_price": 60.0, "unrealized_pl": 0.0},
            {"symbol": "QQQM", "qty": 10.0, "market_value": 2000.0,
             "avg_entry_price": 200.0, "unrealized_pl": 0.0},
        ]
    )

    result = run_preflight(contaminated, _expect())

    assert not result.passed
    detail = next(
        c.detail for c in result.checks
        if c.name == "no unexpected positions"
    )
    assert "BUD" in detail and "QQQM" in detail


def test_positions_inside_universe_are_fine():
    held = _state(
        positions=[
            {"symbol": "SPY", "qty": 100.0, "market_value": 40_000.0,
             "avg_entry_price": 400.0, "unrealized_pl": 0.0},
        ]
    )

    assert run_preflight(held, _expect()).passed


def test_stringified_enum_status_still_passes():
    """
    Regression. alpaca-py returns an enum whose str() is
    "AccountStatus.ACTIVE", and the first version of this check compared
    it against "ACTIVE" and failed on a perfectly healthy account.

    A safety check that cries wolf on good input gets disabled, so this
    class of false positive is worse than a missing check.
    """

    for form in ("ACTIVE", "AccountStatus.ACTIVE", "active"):
        result = run_preflight(_state(status=form), _expect())
        assert result.passed, f"healthy account rejected for {form!r}"


def test_normalize_status_handles_both_forms():
    assert normalize_status("AccountStatus.ACTIVE") == "ACTIVE"
    assert normalize_status("ACTIVE") == "ACTIVE"
    assert normalize_status("") == ""
    assert normalize_status(None) == ""


def test_inactive_status_is_still_caught_in_enum_form():
    assert not run_preflight(
        _state(status="AccountStatus.ACCOUNT_CLOSED"), _expect()
    ).passed


def test_blocked_trading_and_inactive_status_are_fatal():
    assert not run_preflight(
        _state(trading_blocked=True), _expect()
    ).passed
    assert not run_preflight(
        _state(status="ACCOUNT_CLOSED"), _expect()
    ).passed


def test_stale_open_orders_block_unless_permitted():
    stale = _state(
        open_orders=[
            {"id": "1", "symbol": "SPY", "side": "buy", "qty": 10.0,
             "status": "new", "type": "market"}
        ]
    )

    assert not run_preflight(stale, _expect()).passed
    assert run_preflight(stale, _expect(allow_open_orders=True)).passed


def test_insufficient_equity_is_fatal():
    assert not run_preflight(
        _state(equity=500.0), _expect(minimum_equity=1000.0)
    ).passed


def test_pattern_day_trader_warns_only():
    """
    Under $25k a margin account is capped at three day trades per five
    business days. Worth surfacing, not worth blocking on.
    """

    result = run_preflight(_state(pattern_day_trader=True), _expect())

    assert result.passed
    assert any(
        c.name == "pattern day trader flag" for c in result.warnings
    )


def test_config_file_matches_the_dataclass():
    with ACCOUNT_PATH.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)

    assert set(payload) == set(
        AccountExpectations.__dataclass_fields__
    )

    expectations = AccountExpectations.from_yaml(ACCOUNT_PATH)
    assert expectations.universe == tuple(payload["universe"])


def test_unknown_config_keys_are_rejected():
    tmp = PROJECT_ROOT / "config" / "_tmp_account.yaml"
    tmp.write_text('expected_account_id: "x"\nmystery: 1\n')

    try:
        with pytest.raises(ValueError, match="unknown account"):
            AccountExpectations.from_yaml(tmp)
    finally:
        tmp.unlink()
