"""
Tests for normalized equity and order reporting.

The run log is committed to a PUBLIC repository. These tests assert that
nothing in it states the size of the account.
"""

from __future__ import annotations

import json

import pytest

from trading_lab.execution.equity_baseline import (
    EquityBaseline,
    account_fingerprint,
    load_or_create_baseline,
    normalize_order,
)
from trading_lab.execution.rebalance import RebalanceOrder


def test_fingerprint_is_stable_and_hides_the_id():
    raw = "69b01b31-aa3d-475d-8b2c-fd14f6d206d7"
    tag = account_fingerprint(raw)

    assert tag == account_fingerprint(raw)
    assert raw not in tag
    assert len(tag) == 12
    assert account_fingerprint("") == ""


def test_different_accounts_produce_different_fingerprints():
    assert account_fingerprint("acct-a") != account_fingerprint("acct-b")


def test_baseline_is_created_then_reused(tmp_path):
    path = tmp_path / "equity_baseline.json"

    first, created = load_or_create_baseline(
        path, account_id="acct-a", equity=500_000.0
    )
    assert created
    assert first.baseline_equity == 500_000.0

    second, created_again = load_or_create_baseline(
        path, account_id="acct-a", equity=525_000.0
    )
    assert not created_again
    assert second.baseline_equity == 500_000.0
    assert second.ratio(525_000.0) == pytest.approx(1.05)


def test_baseline_resets_on_account_change(tmp_path):
    """
    Carrying a baseline across an account swap would report a fictional
    cumulative return. The paper-account replacement already caused this
    class of error once.
    """

    path = tmp_path / "equity_baseline.json"

    load_or_create_baseline(path, account_id="old", equity=100_000.0)

    fresh, created = load_or_create_baseline(
        path, account_id="new", equity=500_000.0
    )

    assert created
    assert fresh.baseline_equity == 500_000.0
    assert fresh.account_fingerprint == account_fingerprint("new")


def test_corrupt_baseline_is_reset_not_trusted(tmp_path):
    path = tmp_path / "equity_baseline.json"
    path.write_text("{ not valid json")

    baseline, created = load_or_create_baseline(
        path, account_id="acct-a", equity=250_000.0
    )

    assert created
    assert baseline.baseline_equity == 250_000.0


def test_ratio_is_the_only_equity_figure_exposed():
    baseline = EquityBaseline("abc123", 500_000.0, "2026-08-28T00:00:00Z")

    assert baseline.ratio(500_000.0) == pytest.approx(1.0)
    assert baseline.ratio(450_000.0) == pytest.approx(0.9)


def test_normalized_order_hides_account_size():
    """
    900 shares of SPY at $600 states the account size as plainly as an
    equity line. Neither share count nor dollar notional may appear.
    """

    order = RebalanceOrder(
        symbol="SPY",
        side="buy",
        quantity=900.0,
        reference_price=600.0,
        current_weight=0.0,
        target_weight=0.72,
    )

    record = normalize_order(order, equity=750_000.0)

    assert "quantity" not in record
    assert "reference_price" not in record
    assert record["notional_pct"] == pytest.approx(0.72)
    assert record["weight_delta"] == pytest.approx(0.72)

    serialized = json.dumps(record)

    for leak in ("900", "600", "750000", "540000"):
        assert leak not in serialized, f"{leak} leaked account size"


def test_invalid_inputs_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="equity must be positive"):
        load_or_create_baseline(
            tmp_path / "b.json", account_id="a", equity=0.0
        )

    order = RebalanceOrder("SPY", "buy", 1.0, 100.0, 0.0, 0.1)

    with pytest.raises(ValueError, match="equity must be positive"):
        normalize_order(order, equity=0.0)


def test_state_directory_is_gitignored():
    """
    The baseline file is the one place a dollar figure exists. If it
    were committed, normalizing everything else would be pointless.
    """

    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    ignored = (root / ".gitignore").read_text()

    assert "state/" in ignored
