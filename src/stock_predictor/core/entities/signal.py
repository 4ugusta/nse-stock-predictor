"""Trading signal domain entities."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum


class SignalType(Enum):
    """Type of trading signal."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    NO_EDGE = "no_edge"  # Insufficient edge to trade - stay out
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class SignalStrength(Enum):
    """Strength/confidence of the signal."""

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


class Timeframe(Enum):
    """Trading timeframe."""

    INTRADAY = "intraday"  # Same day
    SWING = "swing"  # 2-10 days
    POSITIONAL = "positional"  # Weeks to months


class TradingStyle(Enum):
    """Trading style for signal generation."""

    INTRADAY = "intraday"
    SWING = "swing"
    POSITIONAL = "positional"


@dataclass
class IndicatorSignal:
    """Signal from a single indicator."""

    indicator: str
    value: float
    signal: SignalType
    score: float  # -1.0 to 1.0, negative = bearish, positive = bullish
    reason: str


@dataclass
class TradingSignal:
    """Complete trading signal with entry/exit levels."""

    symbol: str
    signal_type: SignalType
    strength: SignalStrength
    timeframe: Timeframe
    entry_price: Decimal
    target_price: Decimal | None = None
    stop_loss: Decimal | None = None
    confidence: float = 0.0  # 0.0 to 1.0
    reasons: list[str] = field(default_factory=list)
    indicator_signals: list[IndicatorSignal] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)

    @property
    def risk_reward_ratio(self) -> float | None:
        """Calculate risk-reward ratio if targets are set."""
        if self.target_price is None or self.stop_loss is None:
            return None

        risk = abs(float(self.entry_price - self.stop_loss))
        reward = abs(float(self.target_price - self.entry_price))

        if risk == 0:
            return None

        return reward / risk

    @property
    def is_bullish(self) -> bool:
        """Check if signal is bullish."""
        return self.signal_type in (SignalType.BUY, SignalType.STRONG_BUY)

    @property
    def is_bearish(self) -> bool:
        """Check if signal is bearish."""
        return self.signal_type in (SignalType.SELL, SignalType.STRONG_SELL)

    def to_dict(self) -> dict:
        """Convert to dictionary for display."""
        return {
            "symbol": self.symbol,
            "signal": self.signal_type.value,
            "strength": self.strength.value,
            "timeframe": self.timeframe.value,
            "entry": float(self.entry_price),
            "target": float(self.target_price) if self.target_price else None,
            "stop_loss": float(self.stop_loss) if self.stop_loss else None,
            "confidence": f"{self.confidence:.1%}",
            "risk_reward": f"{self.risk_reward_ratio:.2f}" if self.risk_reward_ratio else "N/A",
            "reasons": self.reasons,
        }


@dataclass
class ScreenerMatch:
    """A stock that matches screener criteria."""

    symbol: str
    name: str
    current_price: Decimal
    change_percent: Decimal
    score: float  # Ranking score
    matching_criteria: list[str]
    signal: TradingSignal | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for display."""
        return {
            "symbol": self.symbol,
            "name": self.name,
            "price": float(self.current_price),
            "change": f"{self.change_percent:+.2f}%",
            "score": f"{self.score:.2f}",
            "criteria": ", ".join(self.matching_criteria),
        }
