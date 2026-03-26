"""Stock-related domain entities."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

import pandas as pd


@dataclass
class Stock:
    """Represents a stock/security on NSE."""

    symbol: str
    name: str
    exchange: str = "NSE"
    sector: str | None = None
    industry: str | None = None
    isin: str | None = None

    @property
    def yahoo_symbol(self) -> str:
        """Get Yahoo Finance compatible symbol."""
        return f"{self.symbol}.NS"


@dataclass
class OHLCV:
    """Open-High-Low-Close-Volume data point."""

    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "volume": self.volume,
        }


@dataclass
class Quote:
    """Real-time quote for a stock."""

    symbol: str
    last_price: Decimal
    change: Decimal
    change_percent: Decimal
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    timestamp: datetime
    bid: Decimal | None = None
    ask: Decimal | None = None
    bid_qty: int | None = None
    ask_qty: int | None = None

    @property
    def is_positive(self) -> bool:
        """Check if price change is positive."""
        return self.change >= 0


@dataclass
class StockData:
    """Complete stock data with OHLCV history."""

    symbol: str
    name: str
    data: pd.DataFrame  # OHLCV dataframe
    quote: Quote | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Ensure dataframe has required columns."""
        required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        if not required_cols.issubset(set(self.data.columns)):
            missing = required_cols - set(self.data.columns)
            raise ValueError(f"Missing required columns: {missing}")

    @property
    def latest_price(self) -> float:
        """Get the most recent closing price."""
        if self.quote:
            return float(self.quote.last_price)
        return float(self.data["close"].iloc[-1])

    @property
    def period_high(self) -> float:
        """Get the highest price in the data period."""
        return float(self.data["high"].max())

    @property
    def period_low(self) -> float:
        """Get the lowest price in the data period."""
        return float(self.data["low"].min())


@dataclass
class IndexData:
    """Market index data (Nifty 50, Bank Nifty, etc.)."""

    symbol: str
    name: str
    value: Decimal
    change: Decimal
    change_percent: Decimal
    open: Decimal
    high: Decimal
    low: Decimal
    previous_close: Decimal
    timestamp: datetime
    advances: int | None = None
    declines: int | None = None
    unchanged: int | None = None
