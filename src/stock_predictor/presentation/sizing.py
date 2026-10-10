"""Position sizing shown beside a signal, shared by the dashboard and the CLI."""

from decimal import Decimal
from typing import NamedTuple


class SizingRow(NamedTuple):
    capital: int
    shares: int
    cost: Decimal
    max_loss: Decimal


def position_sizes(entry_price: Decimal, stop_loss: Decimal, capitals=(100_000, 500_000, 1_000_000), risk=Decimal("0.01")) -> list[SizingRow]:
    """Shares to buy at each capital so a stop-out loses `risk` of it.

    Prices arrive as Decimal; the risk stays Decimal too (a float here raised TypeError on every signal).
    Capitals too small to buy one share are left out.
    """
    risk_per_share = abs(Decimal(entry_price) - Decimal(stop_loss))
    if risk_per_share <= 0:
        return []
    rows = []
    for capital in capitals:
        shares = int(capital * risk / risk_per_share)
        if shares > 0:
            rows.append(SizingRow(capital, shares, shares * Decimal(entry_price), shares * risk_per_share))
    return rows
