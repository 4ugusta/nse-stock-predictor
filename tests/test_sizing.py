"""Position sizing beside a signal, as the dashboard and the CLI both show it."""

from decimal import Decimal

from stock_predictor.presentation.sizing import position_sizes


def test_decimal_prices_size_without_error():
    # Entry and stop arrive as Decimal; a float risk raised TypeError here on every BUY or SELL.
    rows = position_sizes(Decimal("266.00"), Decimal("256.29"))
    assert [r.shares for r in rows] == [102, 514, 1029]
    assert rows[0].max_loss <= Decimal("1000")
    assert rows[0].cost == 102 * Decimal("266.00")


def test_a_short_signal_sizes_the_same_way():
    assert [r.shares for r in position_sizes(Decimal("100"), Decimal("110"))] == [100, 500, 1000]


def test_no_rows_without_a_risk_or_for_capital_too_small_for_one_share():
    assert position_sizes(Decimal("100"), Decimal("100")) == []
    assert [r.capital for r in position_sizes(Decimal("5000"), Decimal("10"), capitals=(10_000, 1_000_000))] == [1_000_000]
