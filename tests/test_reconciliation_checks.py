"""
Tests for evening reconciliation.

The question these answer is not "how good was the fill" but "did the
account end up where the plan intended". For a system running while its
owner is at work, silent divergence is the failure that matters.
"""

from __future__ import annotations

import pytest

from trading_lab.execution.position_check import (
    classify_orders,
    reconcile_positions,
    summarize_reconciliation,
)
from trading_lab.execution.reconciliation import reconcile_fill


# ---------------------------------------------------------------------
# Slippage sign convention
# ---------------------------------------------------------------------


def test_buy_above_reference_is_worse():
    result = reconcile_fill(
        order_id="1", symbol="SPY", reference_price=100.0,
        status="filled", filled_avg_price=100.5, filled_qty=10,
        side="buy",
    )

    assert result.slippage_per_share == pytest.approx(0.5)
    assert result.slippage_bps == pytest.approx(50.0)


def test_sell_above_reference_is_better():
    """
    Regression. The original convention assumed buys, so every good sell
    was reported as a loss. Volatility targeting reduces exposure by
    selling, so this would have been wrong on roughly half of all
    orders.
    """

    result = reconcile_fill(
        order_id="1", symbol="SPY", reference_price=100.0,
        status="filled", filled_avg_price=100.5, filled_qty=10,
        side="sell",
    )

    assert result.slippage_per_share == pytest.approx(-0.5)
    assert result.slippage_bps == pytest.approx(-50.0)


def test_sell_below_reference_is_worse():
    result = reconcile_fill(
        order_id="1", symbol="SPY", reference_price=100.0,
        status="filled", filled_avg_price=99.0, filled_qty=10,
        side="sell",
    )

    assert result.slippage_per_share == pytest.approx(1.0)


def test_invalid_side_is_rejected():
    with pytest.raises(ValueError, match="side must be"):
        reconcile_fill(
            order_id="1", symbol="SPY", reference_price=100.0,
            status="filled", filled_avg_price=100.0, filled_qty=1,
            side="sideways",
        )


# ---------------------------------------------------------------------
# Order classification
# ---------------------------------------------------------------------


def test_missing_broker_record_is_critical():
    """
    The worst case: the system believed it submitted and the broker has
    no trace.
    """

    issues = classify_orders(
        planned=[{"symbol": "SPY", "side": "buy"}],
        broker_orders=[],
    )

    assert len(issues) == 1
    assert issues[0]["severity"] == "critical"
    assert "no broker record" in issues[0]["issue"]


def test_rejected_order_is_critical():
    issues = classify_orders(
        planned=[{"symbol": "SPY", "side": "buy"}],
        broker_orders=[{"symbol": "SPY", "status": "rejected"}],
    )

    assert issues[0]["severity"] == "critical"


def test_partial_fill_is_a_warning():
    issues = classify_orders(
        planned=[{"symbol": "SPY", "side": "buy"}],
        broker_orders=[
            {"symbol": "SPY", "status": "partially_filled",
             "qty": 100, "filled_qty": 40}
        ],
    )

    assert issues[0]["severity"] == "warning"
    assert "40" in issues[0]["detail"]


def test_filled_order_raises_nothing():
    assert classify_orders(
        planned=[{"symbol": "SPY", "side": "buy"}],
        broker_orders=[{"symbol": "SPY", "status": "filled"}],
    ) == []


# ---------------------------------------------------------------------
# Position reconciliation
# ---------------------------------------------------------------------


def test_matching_position_reconciles_clean():
    report = reconcile_positions(
        intended_weights={"SPY": 0.72},
        actual_quantities={"SPY": 180.0},
        prices={"SPY": 400.0},
        equity=100_000.0,
        submitted=True,
    )

    assert report.clean
    assert report.discrepancies[0].severity == "ok"


def test_overnight_drift_is_tolerated():
    """
    Price movement between the decision close and the next session
    shifts the held weight slightly. That is expected, not a failure.
    """

    report = reconcile_positions(
        intended_weights={"SPY": 0.72},
        actual_quantities={"SPY": 180.0},
        prices={"SPY": 407.0},        # ~1.75% overnight move
        equity=100_000.0,
        submitted=True,
    )

    assert report.clean


def test_unexplained_large_gap_is_major():
    report = reconcile_positions(
        intended_weights={"SPY": 0.72},
        actual_quantities={},
        prices={"SPY": 400.0},
        equity=100_000.0,
        submitted=True,
    )

    assert not report.clean
    assert report.problems[0].severity == "major"
    assert "unexplained" in report.problems[0].likely_cause


def test_dry_run_gap_is_not_an_error():
    """
    A dry run leaves the account untouched by design. Flagging that
    every evening would train the reader to ignore the report.
    """

    report = reconcile_positions(
        intended_weights={"SPY": 0.72},
        actual_quantities={},
        prices={"SPY": 400.0},
        equity=100_000.0,
        submitted=False,
    )

    assert report.clean
    assert "dry run" in report.discrepancies[0].likely_cause


def test_gap_is_attributed_to_a_known_order_issue():
    issues = classify_orders(
        planned=[{"symbol": "SPY", "side": "buy"}],
        broker_orders=[{"symbol": "SPY", "status": "rejected"}],
    )

    report = reconcile_positions(
        intended_weights={"SPY": 0.72},
        actual_quantities={},
        prices={"SPY": 400.0},
        equity=100_000.0,
        submitted=True,
        order_issues=issues,
    )

    assert not report.clean
    assert "explained by an order issue" in (
        report.problems[0].likely_cause
    )


def test_unexpected_holding_is_caught():
    """
    A position the strategy never intended must be surfaced, not
    ignored because it is absent from the target set.
    """

    report = reconcile_positions(
        intended_weights={"SPY": 0.72},
        actual_quantities={"SPY": 180.0, "BUD": 100.0},
        prices={"SPY": 400.0, "BUD": 60.0},
        equity=100_000.0,
        submitted=True,
    )

    assert not report.clean
    assert any(d.symbol == "BUD" for d in report.problems)


def test_missing_price_is_flagged_as_unknown():
    report = reconcile_positions(
        intended_weights={"SPY": 0.72},
        actual_quantities={"SPY": 180.0},
        prices={},
        equity=100_000.0,
        submitted=True,
    )

    assert report.discrepancies[0].severity == "unknown"


def test_summary_reports_clean_and_dirty_states():
    clean = reconcile_positions(
        intended_weights={"SPY": 0.72},
        actual_quantities={"SPY": 180.0},
        prices={"SPY": 400.0},
        equity=100_000.0,
        submitted=True,
    )
    assert "RECONCILED CLEAN" in summarize_reconciliation(clean)

    dirty = reconcile_positions(
        intended_weights={"SPY": 0.72},
        actual_quantities={},
        prices={"SPY": 400.0},
        equity=100_000.0,
        submitted=True,
    )
    text = summarize_reconciliation(dirty)
    assert "FOUND ISSUES" in text and "major" in text
