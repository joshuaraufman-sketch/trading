from datetime import datetime, timezone

import trading_lab.execution.pending_signals as pending


def test_save_and_load_pending_signal(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        pending,
        "PENDING_DIR",
        tmp_path,
    )

    signal_time = datetime(
        2026,
        8,
        27,
        20,
        0,
        tzinfo=timezone.utc,
    )

    path = pending.save_pending_signal(
        symbol="SPY",
        signal_time=signal_time,
        signal_reference_price=100.0,
        strategy_name="sma_crossover",
        parameters={
            "sma_window": 10,
            "holding_days": 10,
            "stop_loss_pct": 0.02,
            "risk_pct": 0.005,
        },
    )

    assert path.exists()

    records = (
        pending.load_pending_signals()
    )

    assert len(records) == 1
    assert records[0]["symbol"] == "SPY"
    assert records[0]["status"] == "pending"


def test_processed_signal_is_not_loaded(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        pending,
        "PENDING_DIR",
        tmp_path,
    )

    signal_time = datetime(
        2026,
        8,
        27,
        20,
        0,
        tzinfo=timezone.utc,
    )

    path = pending.save_pending_signal(
        symbol="SPY",
        signal_time=signal_time,
        signal_reference_price=100.0,
        strategy_name="sma_crossover",
        parameters={},
    )

    pending.mark_signal_processed(
        path,
        status="submitted",
        order_id="test-order-id",
    )

    records = (
        pending.load_pending_signals()
    )

    assert records == []


def test_rejects_invalid_reference_price(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        pending,
        "PENDING_DIR",
        tmp_path,
    )

    signal_time = datetime(
        2026,
        8,
        27,
        20,
        0,
        tzinfo=timezone.utc,
    )

    try:
        pending.save_pending_signal(
            symbol="SPY",
            signal_time=signal_time,
            signal_reference_price=0,
            strategy_name="sma_crossover",
            parameters={},
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError"
        )