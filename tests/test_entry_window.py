from unittest.mock import patch

from trading_lab.risk.entry_window import (
    NEW_YORK,
    check_next_open_entry_window,
)


def make_datetime(
    year,
    month,
    day,
    hour,
    minute,
):
    from datetime import datetime

    return datetime(
        year,
        month,
        day,
        hour,
        minute,
        tzinfo=NEW_YORK,
    )


@patch(
    "trading_lab.risk.entry_window.datetime"
)
def test_entry_window_passes_at_935(
    mock_datetime,
):
    mock_datetime.now.return_value = (
        make_datetime(
            2026,
            8,
            27,
            9,
            35,
        )
    )

    decision = (
        check_next_open_entry_window()
    )

    assert decision.approved is True


@patch(
    "trading_lab.risk.entry_window.datetime"
)
def test_entry_window_blocks_before_open(
    mock_datetime,
):
    mock_datetime.now.return_value = (
        make_datetime(
            2026,
            8,
            27,
            9,
            20,
        )
    )

    decision = (
        check_next_open_entry_window()
    )

    assert decision.approved is False


@patch(
    "trading_lab.risk.entry_window.datetime"
)
def test_entry_window_blocks_late(
    mock_datetime,
):
    mock_datetime.now.return_value = (
        make_datetime(
            2026,
            8,
            27,
            10,
            0,
        )
    )

    decision = (
        check_next_open_entry_window()
    )

    assert decision.approved is False