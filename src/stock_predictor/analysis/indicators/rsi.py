"""Relative Strength Index (RSI) indicator calculations."""

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from stock_predictor.infrastructure.config.constants import INDICATOR_DEFAULTS


class RSIZone(Enum):
    """RSI zone classification."""

    OVERBOUGHT = "overbought"
    OVERSOLD = "oversold"
    NEUTRAL = "neutral"


@dataclass
class RSIResult:
    """RSI calculation result."""

    value: float
    zone: RSIZone
    signal_strength: float  # -1.0 to 1.0

    @property
    def is_overbought(self) -> bool:
        return self.zone == RSIZone.OVERBOUGHT

    @property
    def is_oversold(self) -> bool:
        return self.zone == RSIZone.OVERSOLD


@dataclass
class RSIDivergence:
    """RSI divergence detection result."""

    divergence_type: str  # "bullish", "bearish", or "none"
    price_trend: str  # "higher_high", "lower_low", etc.
    rsi_trend: str  # "lower_high", "higher_low", etc.
    strength: float  # 0.0 to 1.0


class RSICalculator:
    """Calculator for Relative Strength Index."""

    def __init__(
        self,
        period: int | None = None,
        overbought: float | None = None,
        oversold: float | None = None,
    ) -> None:
        """Initialize RSI calculator.

        Args:
            period: RSI period (default: 14)
            overbought: Overbought threshold (default: 70)
            oversold: Oversold threshold (default: 30)
        """
        self.period = period or INDICATOR_DEFAULTS["rsi_period"]
        self.overbought = overbought or INDICATOR_DEFAULTS["rsi_overbought"]
        self.oversold = oversold or INDICATOR_DEFAULTS["rsi_oversold"]

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate RSI values.

        Args:
            data: DataFrame with 'close' column

        Returns:
            Series with RSI values
        """
        delta = data["close"].diff()

        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=self.period, min_periods=self.period).mean()
        avg_loss = loss.rolling(window=self.period, min_periods=self.period).mean()

        # Use EMA for subsequent calculations (Wilder's smoothing)
        for i in range(self.period, len(gain)):
            avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (self.period - 1) + gain.iloc[i]) / self.period
            avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (self.period - 1) + loss.iloc[i]) / self.period

        # Explicit edge cases: all-gains and all-losses
        rsi = pd.Series(index=data.index, dtype=float)
        for i in range(len(avg_gain)):
            ag = avg_gain.iloc[i]
            al = avg_loss.iloc[i]
            if pd.isna(ag) or pd.isna(al):
                rsi.iloc[i] = float('nan')
            elif al == 0 and ag == 0:
                rsi.iloc[i] = 50.0  # No movement = neutral
            elif al == 0:
                rsi.iloc[i] = 100.0  # All gains, no losses
            elif ag == 0:
                rsi.iloc[i] = 0.0  # All losses, no gains
            else:
                rs = ag / al
                rsi.iloc[i] = 100 - (100 / (1 + rs))

        return rsi

    def get_zone(self, rsi_value: float) -> RSIZone:
        """Determine RSI zone.

        Args:
            rsi_value: RSI value

        Returns:
            RSI zone classification
        """
        if rsi_value >= self.overbought:
            return RSIZone.OVERBOUGHT
        elif rsi_value <= self.oversold:
            return RSIZone.OVERSOLD
        return RSIZone.NEUTRAL

    def get_signal_strength(self, rsi_value: float) -> float:
        """Calculate signal strength based on RSI.

        Args:
            rsi_value: RSI value

        Returns:
            Signal strength from -1.0 (very bearish) to 1.0 (very bullish)
        """
        # Extreme oversold (RSI < 20) = strong bullish
        if rsi_value <= 20:
            return 0.8 + (20 - rsi_value) / 100
        # Oversold (RSI 20-30) = moderate bullish
        elif rsi_value <= self.oversold:
            return 0.3 + (self.oversold - rsi_value) / (self.overbought - self.oversold)
        # Extreme overbought (RSI > 80) = strong bearish
        elif rsi_value >= 80:
            return -0.8 - (rsi_value - 80) / 100
        # Overbought (RSI 70-80) = moderate bearish
        elif rsi_value >= self.overbought:
            return -0.3 - (rsi_value - self.overbought) / (self.overbought - self.oversold)
        # Neutral zone - slight bias based on position
        else:
            # Center at 50, range -0.2 to 0.2
            return (50 - rsi_value) / 100

    def analyze(self, data: pd.DataFrame) -> RSIResult:
        """Analyze RSI and return result.

        Args:
            data: DataFrame with 'close' column

        Returns:
            RSIResult with value, zone, and signal strength
        """
        # Warmup enforcement: need at least period + 10 bars for meaningful RSI
        if len(data) < self.period + 10:
            return RSIResult(
                value=50.0,
                zone=RSIZone.NEUTRAL,
                signal_strength=0.0,
            )

        rsi_series = self.calculate(data)
        current_rsi = rsi_series.iloc[-1]

        # NaN guard: if current RSI is NaN, return neutral
        if pd.isna(current_rsi):
            current_rsi = 50.0

        return RSIResult(
            value=current_rsi,
            zone=self.get_zone(current_rsi),
            signal_strength=self.get_signal_strength(current_rsi),
        )

    def detect_divergence(self, data: pd.DataFrame, lookback: int = 28) -> RSIDivergence:
        """Detect RSI divergence (price vs RSI disagreement).

        Improved with:
        - Dynamic lookback scaled to data length
        - Pivot-based detection (split into two halves)
        - Relative strength calculation using RSI range

        Args:
            data: DataFrame with 'close' column
            lookback: Number of periods to look back for divergence

        Returns:
            RSIDivergence result
        """
        rsi = self.calculate(data)

        # Scale lookback to available data
        effective_lookback = min(lookback, len(data) // 3)
        if effective_lookback < 10 or len(data) < effective_lookback * 2:
            return RSIDivergence(
                divergence_type="none",
                price_trend="insufficient_data",
                rsi_trend="insufficient_data",
                strength=0.0,
            )

        # Split into two halves for comparison
        half = effective_lookback // 2
        first_half_prices = data["close"].iloc[-effective_lookback:-half]
        second_half_prices = data["close"].iloc[-half:]
        first_half_rsi = rsi.iloc[-effective_lookback:-half].dropna()
        second_half_rsi = rsi.iloc[-half:].dropna()

        if len(first_half_rsi) < 3 or len(second_half_rsi) < 3:
            return RSIDivergence(
                divergence_type="none",
                price_trend="insufficient_data",
                rsi_trend="insufficient_data",
                strength=0.0,
            )

        # Find lows and highs in each half
        price_low_1 = first_half_prices.min()
        price_low_2 = second_half_prices.min()
        price_high_1 = first_half_prices.max()
        price_high_2 = second_half_prices.max()

        rsi_low_1 = first_half_rsi.min()
        rsi_low_2 = second_half_rsi.min()
        rsi_high_1 = first_half_rsi.max()
        rsi_high_2 = second_half_rsi.max()

        # RSI range for relative strength calculation
        rsi_range = max(rsi_high_1, rsi_high_2) - min(rsi_low_1, rsi_low_2)
        if rsi_range < 5:  # RSI barely moved — no meaningful divergence
            return RSIDivergence(
                divergence_type="none",
                price_trend="normal",
                rsi_trend="normal",
                strength=0.0,
            )

        # Bullish divergence: price makes lower low, RSI makes higher low
        if price_low_2 < price_low_1 and rsi_low_2 > rsi_low_1:
            strength = (rsi_low_2 - rsi_low_1) / rsi_range
            return RSIDivergence(
                divergence_type="bullish",
                price_trend="lower_low",
                rsi_trend="higher_low",
                strength=min(1.0, max(0.0, strength)),
            )

        # Bearish divergence: price makes higher high, RSI makes lower high
        if price_high_2 > price_high_1 and rsi_high_2 < rsi_high_1:
            strength = (rsi_high_1 - rsi_high_2) / rsi_range
            return RSIDivergence(
                divergence_type="bearish",
                price_trend="higher_high",
                rsi_trend="lower_high",
                strength=min(1.0, max(0.0, strength)),
            )

        return RSIDivergence(
            divergence_type="none",
            price_trend="normal",
            rsi_trend="normal",
            strength=0.0,
        )

    def add_to_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add RSI as a column to dataframe.

        Args:
            data: DataFrame with 'close' column

        Returns:
            DataFrame with RSI column added
        """
        df = data.copy()
        df[f"RSI_{self.period}"] = self.calculate(data)
        return df
