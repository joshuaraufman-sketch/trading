import pytest

from trading_lab.execution.trade_lifecycle import (
    close_trade_record,
    create_trade_record,
    load_trade_record,
    update_entry_fill,
)


def test_create_trade_record(tmp_path):
    path = create_trade_record(
        entry_order_id="order-123",
        symbol="SPY",
        strategy_name="sma_crossover",
        signal_date="2026-08-27",
        signal_time=(
            "2026-08-27 04:00:00+00:00"
        ),
        reference_price=770.0,
        quantity=10,
        planned_stop_price=754.6,
        holding_days=10,
        trade_dir=tmp_path,
    )

    assert path.exists()

    record = load_trade_record(
        "order-123",
        trade_dir=tmp_path,
    )

    assert record["status"] == "submitted"
    assert record["symbol"] == "SPY"
    assert record["entry"]["filled_qty"] == 0


def test_entry_fill_opens_trade(tmp_path):
    create_trade_record(
        entry_order_id="order-123",
        symbol="SPY",
        strategy_name="sma_crossover",
        signal_date="2026-08-27",
        signal_time="test",
        reference_price=770.0,
        quantity=10,
        planned_stop_price=754.6,
        holding_days=10,
        trade_dir=tmp_path,
    )

    update_entry_fill(
        entry_order_id="order-123",
        status="filled",
        filled_qty=10,
        filled_avg_price=771.0,
        trade_dir=tmp_path,
    )

    record = load_trade_record(
        "order-123",
        trade_dir=tmp_path,
    )

    assert record["status"] == "open"
    assert (
        record["entry"]["filled_avg_price"]
        == 771.0
    )


def test_close_trade_calculates_pnl(
    tmp_path,
):
    create_trade_record(
        entry_order_id="order-123",
        symbol="SPY",
        strategy_name="sma_crossover",
        signal_date="2026-08-27",
        signal_time="test",
        reference_price=100.0,
        quantity=10,
        planned_stop_price=98.0,
        holding_days=10,
        trade_dir=tmp_path,
    )

    update_entry_fill(
        entry_order_id="order-123",
        status="filled",
        filled_qty=10,
        filled_avg_price=100.0,
        trade_dir=tmp_path,
    )

    close_trade_record(
        entry_order_id="order-123",
        exit_order_id="exit-456",
        exit_reason="holding_period",
        filled_qty=10,
        filled_avg_price=105.0,
        trade_dir=tmp_path,
    )

    record = load_trade_record(
        "order-123",
        trade_dir=tmp_path,
    )

    assert record["status"] == "closed"
    assert record["realized_pnl"] == 50.0
    assert record["realized_return_pct"] == pytest.approx(
        5.0
    )


def test_duplicate_trade_is_blocked(
    tmp_path,
):
    kwargs = {
        "entry_order_id": "order-123",
        "symbol": "SPY",
        "strategy_name": "sma_crossover",
        "signal_date": "2026-08-27",
        "signal_time": "test",
        "reference_price": 100.0,
        "quantity": 10,
        "planned_stop_price": 98.0,
        "holding_days": 10,
        "trade_dir": tmp_path,
    }

    create_trade_record(**kwargs)

    with pytest.raises(FileExistsError):
        create_trade_record(**kwargs)
