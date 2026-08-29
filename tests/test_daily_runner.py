"""
Tests for the post-close daily run.

Every test here is about REFUSING. A runner that trades when it should
not is far more dangerous than one that fails to trade, because the
failure is silent and the orders look reasonable.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from trading_lab.execution.alpaca_account import AccountState
from trading_lab.execution.daily_runner import (
    StrategyConfig,
    build_daily_plan,
    latest_session,
    summarize_plan,
)
from trading_lab.execution.preflight import AccountExpectations
from trading_lab.execution.rebalance import ExecutionPolicy


POLICY = ExecutionPolicy(
    max_position_pct=1.0,
    max_gross_exposure=1.0,
    min_order_notional=100.0,
    rebalance_band=0.05,
    allow_short=False,
    allow_fractional_shares=True,
    max_orders_per_session=10,
)

EXPECT = AccountExpectations(
    expected_account_id="acct-new",
    universe=("SPY",),
    minimum_equity=1000.0,
)

STRATEGY = StrategyConfig(symbol="SPY", target_volatility=0.10,
                          lookback=20, max_weight=1.0)


def _bars(n=200, end="2026-08-28", vol=0.01, seed=5):
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(end=end, periods=n, tz="UTC")
    closes = 400 * np.cumprod(1 + rng.normal(0.0003, vol, n))

    return pd.DataFrame(
        {
            "symbol": "SPY",
            "timestamp": sessions + pd.Timedelta(hours=5),
            "close": closes,
        }
    )


def _account(**overrides) -> AccountState:
    base = {
        "equity": 100_000.0,
        "cash": 100_000.0,
        "buying_power": 100_000.0,
        "positions": [],
        "open_orders": [],
        "account_id": "acct-new",
        "account_number": "PA1",
        "status": "ACTIVE",
        "pattern_day_trader": False,
        "trading_blocked": False,
        "is_paper": True,
    }
    base.update(overrides)
    return AccountState(**base)


def _plan(**overrides):
    kwargs = {
        "account_state": _account(),
        "bars": _bars(),
        "expected_session": date(2026, 8, 28),
        "expectations": EXPECT,
        "policy": POLICY,
        "strategy": STRATEGY,
        "completed_sessions": set(),
    }
    kwargs.update(overrides)
    return build_daily_plan(**kwargs)


def test_clean_run_produces_a_buy_from_flat():
    plan = _plan()

    assert not plan.blocked
    assert plan.should_trade
    assert plan.rebalance.orders[0].side == "buy"
    assert 0.0 < plan.target_weights["SPY"] <= 1.0


def test_stale_bars_block_the_run():
    """
    THE CHECK THAT MATTERS MOST. Acting on yesterday's close while
    believing it is today's shifts the whole strategy one session. The
    resulting orders look entirely reasonable in a log -- they are just
    answering the wrong question.
    """

    plan = _plan(
        bars=_bars(end="2026-08-27"),
        expected_session=date(2026, 8, 28),
    )

    assert plan.blocked
    assert "stale or mismatched data" in plan.blocked_reason
    assert "2026-08-27" in plan.blocked_reason


def test_future_dated_bars_also_block():
    plan = _plan(
        bars=_bars(end="2026-09-04"),
        expected_session=date(2026, 8, 28),
    )

    assert plan.blocked
    assert "stale or mismatched" in plan.blocked_reason


def test_duplicate_session_is_refused():
    plan = _plan(completed_sessions={date(2026, 8, 28)})

    assert plan.blocked
    assert "already processed" in plan.blocked_reason
    assert plan.rebalance is None


def test_preflight_failure_stops_everything():
    plan = _plan(account_state=_account(account_id="acct-wrong"))

    assert plan.blocked
    assert "preflight failed" in plan.blocked_reason
    assert plan.rebalance is None
    assert plan.target_weights == {}


def test_unexpected_position_stops_the_run():
    """
    The contamination that sat in the retired account for weeks.
    """

    contaminated = _account(
        positions=[
            {"symbol": "BUD", "qty": 1.0, "market_value": 60.0,
             "avg_entry_price": 60.0, "unrealized_pl": 0.0},
        ]
    )

    assert _plan(account_state=contaminated).blocked


def test_insufficient_history_blocks_rather_than_guessing():
    plan = _plan(bars=_bars(n=10))

    assert plan.blocked
    assert "insufficient history" in plan.blocked_reason


def test_missing_symbol_blocks():
    bars = _bars()
    bars["symbol"] = "QQQ"

    plan = _plan(bars=bars)

    assert plan.blocked
    assert "no bars supplied" in plan.blocked_reason


def test_no_action_when_already_at_target():
    """
    Holding roughly the right weight must produce NO orders, not a
    token rebalance. Churn is the enemy of a 2.69x turnover budget.
    """

    calm = _bars(vol=0.006, seed=9)
    probe = _plan(bars=calm)
    target = probe.target_weights["SPY"]
    price = probe.diagnostics["reference_price"]

    holding = _account(
        positions=[
            {
                "symbol": "SPY",
                "qty": target * 100_000.0 / price,
                "market_value": target * 100_000.0,
                "avg_entry_price": price,
                "unrealized_pl": 0.0,
            }
        ]
    )

    plan = _plan(bars=calm, account_state=holding)

    assert not plan.blocked
    assert not plan.should_trade
    assert "within rebalance band" in plan.rebalance.skipped["SPY"]


def test_high_volatility_reduces_the_target_weight():
    calm = _plan(bars=_bars(vol=0.006, seed=3))
    wild = _plan(bars=_bars(vol=0.030, seed=3))

    assert wild.target_weights["SPY"] < calm.target_weights["SPY"]


def test_weight_never_exceeds_the_policy_cap():
    plan = _plan(bars=_bars(vol=0.001, seed=4))

    assert plan.target_weights["SPY"] <= STRATEGY.max_weight
    assert plan.rebalance.applied_weights["SPY"] <= POLICY.max_position_pct


def test_latest_session_reads_the_market_date():
    assert latest_session(_bars(end="2026-08-28")) == date(2026, 8, 28)


def test_summary_states_blocked_reason_plainly():
    blocked = _plan(bars=_bars(end="2026-08-27"))
    text = summarize_plan(blocked)

    assert "BLOCKED" in text
    assert "stale" in text

    active = summarize_plan(_plan())
    assert "BUY" in active and "SPY" in active


def test_live_strategy_config_matches_the_dataclass():
    """
    Guards the failure that hit research_rules.yaml: a config file whose
    keys nothing reads.
    """

    import yaml
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "config" / "live_strategy.yaml"
    )

    payload = yaml.safe_load(path.read_text())

    assert set(payload) == set(StrategyConfig.__dataclass_fields__)

    config = StrategyConfig(**payload)

    assert config.max_weight <= 1.0, (
        "max_weight above 1.0 is leverage and must be a deliberate, "
        "recorded decision"
    )


def test_deployed_parameters_match_what_was_validated():
    """
    The deployed values must be the ones research validated. Drift here
    means trading a strategy that was never tested, while the reports
    describe a different one.
    """

    import yaml
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "config" / "live_strategy.yaml"
    )
    config = StrategyConfig(**yaml.safe_load(path.read_text()))

    assert config.symbol == "SPY"
    assert config.target_volatility == pytest.approx(0.10)
    assert config.lookback == 20


def test_invalid_strategy_parameters_are_rejected():
    with pytest.raises(ValueError, match="target_volatility"):
        StrategyConfig(target_volatility=0.0)

    with pytest.raises(ValueError, match="lookback"):
        StrategyConfig(lookback=1)
