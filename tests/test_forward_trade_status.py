import json

from trading_lab.validation.forward_status import (
    summarize_forward_trades,
)


def _write(path, record):
    path.write_text(
        json.dumps(record),
        encoding="utf-8",
    )


def test_trade_lifecycle_summary(tmp_path):
    _write(
        tmp_path / "submitted.json",
        {
            "status": "submitted",
            "realized_pnl": None,
        },
    )

    _write(
        tmp_path / "open.json",
        {
            "status": "open",
            "realized_pnl": None,
        },
    )

    _write(
        tmp_path / "winner.json",
        {
            "status": "closed",
            "realized_pnl": 25.0,
        },
    )

    _write(
        tmp_path / "loser.json",
        {
            "status": "closed",
            "realized_pnl": -10.0,
        },
    )

    (
        tmp_path / "bad.json"
    ).write_text(
        "not json",
        encoding="utf-8",
    )

    result = summarize_forward_trades(
        tmp_path
    )

    assert result.files == 5
    assert result.valid_records == 4
    assert result.invalid_records == 1
    assert result.submitted_trades == 1
    assert result.open_trades == 1
    assert result.completed_trades == 2
    assert result.winning_trades == 1
    assert result.losing_trades == 1
    assert result.total_realized_pnl == 15.0


def test_empty_trade_directory(tmp_path):
    result = summarize_forward_trades(
        tmp_path
    )

    assert result.files == 0
    assert result.completed_trades == 0
    assert result.open_trades == 0
    assert result.total_realized_pnl == 0.0
