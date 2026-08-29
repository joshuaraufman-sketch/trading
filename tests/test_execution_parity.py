"""
Backtest/live parity tests.

The guarantee is architectural: both paths call
``compute_rebalance_orders``. These tests confirm the shared function
behaves correctly and that the policy loaded from config is the one
actually applied, so parity cannot be quietly lost by editing one side.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from trading_lab.execution.rebalance import (
    ExecutionPolicy,
    clamp_weights,
    compute_rebalance_orders,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = PROJECT_ROOT / "config" / "execution_policy.yaml"


def _policy(**overrides) -> ExecutionPolicy:
    base = {
        "max_position_pct": 1.0,
        "max_gross_exposure": 1.0,
        "min_order_notional": 100.0,
        "rebalance_band": 0.05,
        "allow_short": False,
        "allow_fractional_shares": True,
        "max_orders_per_session": 10,
    }
    base.update(overrides)
    return ExecutionPolicy(**base)


def test_shipped_policy_permits_the_validated_strategy():
    """
    The regression that motivated this module. Volatility targeting
    averages 72% exposure and reduces itself by SELLING. The previous
    live rules capped positions at 25% and rejected every sell, so not
    one of its orders could have been placed.
    """

    policy = ExecutionPolicy.from_yaml(POLICY_PATH)

    assert policy.max_position_pct >= 0.75, (
        "policy would block the validated ~72% average exposure"
    )
    assert not policy.uses_leverage, (
        "leverage must remain off until deliberately enabled"
    )

    plan = compute_rebalance_orders(
        target_weights={"SPY": 0.45},
        current_quantities={"SPY": 150.0},
        prices={"SPY": 400.0},
        equity=100_000.0,
        policy=policy,
    )

    assert len(plan.orders) == 1
    assert plan.orders[0].side == "sell"


def test_reducing_exposure_produces_a_sell():
    plan = compute_rebalance_orders(
        target_weights={"SPY": 0.40},
        current_quantities={"SPY": 200.0},
        prices={"SPY": 400.0},
        equity=100_000.0,
        policy=_policy(),
    )

    order = plan.orders[0]

    assert order.side == "sell"
    assert order.current_weight == pytest.approx(0.80)
    assert order.target_weight == pytest.approx(0.40)
    assert order.quantity == pytest.approx(100.0)


def test_rebalance_band_suppresses_small_drift():
    plan = compute_rebalance_orders(
        target_weights={"SPY": 0.72},
        current_quantities={"SPY": 175.0},   # 0.70 of equity
        prices={"SPY": 400.0},
        equity=100_000.0,
        policy=_policy(rebalance_band=0.05),
    )

    assert plan.orders == []
    assert "within rebalance band" in plan.skipped["SPY"]


def test_minimum_notional_is_reported_not_silently_dropped():
    """
    A skipped order must record why. Unexplained gaps between intended
    and actual exposure are how live diverges from backtest unnoticed.
    """

    plan = compute_rebalance_orders(
        target_weights={"SPY": 0.5006},
        current_quantities={"SPY": 125.0},
        prices={"SPY": 400.0},
        equity=100_000.0,
        policy=_policy(rebalance_band=0.0, min_order_notional=100.0),
    )

    assert plan.orders == []
    assert "below minimum notional" in plan.skipped["SPY"]


def test_position_cap_clamps_and_gross_cap_scales():
    clamped = clamp_weights(
        {"SPY": 0.9, "QQQ": 0.9},
        _policy(max_position_pct=0.6, max_gross_exposure=1.0),
    )

    assert clamped["SPY"] == pytest.approx(0.5)
    assert clamped["QQQ"] == pytest.approx(0.5)
    assert sum(clamped.values()) == pytest.approx(1.0)


def test_gross_scaling_preserves_relative_shape():
    """
    Scaling rather than truncating matters: truncation would silently
    change which instrument dominates the book.
    """

    clamped = clamp_weights(
        {"SPY": 1.2, "QQQ": 0.6},
        _policy(max_position_pct=2.0, max_gross_exposure=0.9),
    )

    assert clamped["SPY"] / clamped["QQQ"] == pytest.approx(2.0)
    assert sum(clamped.values()) == pytest.approx(0.9)


def test_shorts_are_zeroed_when_disallowed():
    clamped = clamp_weights({"SPY": -0.5}, _policy(allow_short=False))
    assert clamped["SPY"] == 0.0

    allowed = clamp_weights({"SPY": -0.5}, _policy(allow_short=True))
    assert allowed["SPY"] == pytest.approx(-0.5)


def test_whole_share_rounding_when_fractional_disabled():
    plan = compute_rebalance_orders(
        target_weights={"SPY": 0.5},
        current_quantities={},
        prices={"SPY": 333.0},
        equity=100_000.0,
        policy=_policy(allow_fractional_shares=False),
    )

    assert plan.orders[0].quantity == pytest.approx(150.0)


def test_order_cap_keeps_largest_and_is_order_independent():
    """
    Truncating in dictionary order would make live results depend on
    iteration order, which a backtest cannot reproduce.
    """

    targets = {f"S{i}": 0.2 for i in range(6)}
    prices = {f"S{i}": 100.0 for i in range(6)}

    plan = compute_rebalance_orders(
        target_weights=targets,
        current_quantities={},
        prices=prices,
        equity=100_000.0,
        policy=_policy(max_position_pct=0.2, max_gross_exposure=1.2,
                       max_orders_per_session=3),
    )

    assert len(plan.orders) == 3
    notionals = [o.notional for o in plan.orders]
    assert notionals == sorted(notionals, reverse=True)
    assert len(plan.skipped) == 3


def test_missing_price_is_flagged_not_assumed():
    plan = compute_rebalance_orders(
        target_weights={"SPY": 0.5},
        current_quantities={"SPY": 100.0},
        prices={},
        equity=100_000.0,
        policy=_policy(),
    )

    assert plan.orders == []
    assert plan.skipped["SPY"] == "no valid price"


def test_liquidating_a_dropped_symbol_still_generates_a_sell():
    """
    A symbol that leaves the target set must be sold, not forgotten.
    """

    plan = compute_rebalance_orders(
        target_weights={"SPY": 1.0},
        current_quantities={"SPY": 150.0, "QQQ": 100.0},
        prices={"SPY": 400.0, "QQQ": 300.0},
        equity=100_000.0,
        policy=_policy(),
    )

    sides = {o.symbol: o.side for o in plan.orders}
    assert sides["QQQ"] == "sell"


def test_config_file_matches_the_dataclass():
    """
    Guards against the failure that hit research_rules.yaml: a config
    file whose keys nothing reads.
    """

    with POLICY_PATH.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)

    assert set(payload) == set(ExecutionPolicy.__dataclass_fields__)

    policy = ExecutionPolicy.from_yaml(POLICY_PATH)

    for key, value in payload.items():
        assert getattr(policy, key) == value


def test_unknown_policy_keys_are_rejected():
    tmp = PROJECT_ROOT / "config" / "_tmp_policy.yaml"
    tmp.write_text("max_position_pct: 1.0\nmystery_setting: 3\n")

    try:
        with pytest.raises(ValueError, match="unknown execution policy"):
            ExecutionPolicy.from_yaml(tmp)
    finally:
        tmp.unlink()


def test_invalid_policy_values_are_rejected():
    with pytest.raises(ValueError, match="max_position_pct"):
        _policy(max_position_pct=0.0)

    with pytest.raises(ValueError, match="rebalance_band"):
        _policy(rebalance_band=-0.1)
