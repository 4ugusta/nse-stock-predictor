"""Average True Range (ATR) indicator for volatility measurement.

ATR measures market volatility by decomposing the entire range of a price
for a given period. It is used for:
- Volatility-adjusted stop losses (instead of fixed percentage)
- Position sizing (lower ATR = larger position, higher ATR = smaller position)
- Detecting volatility regime changes
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class VolatilityRegime(Enum):
    """Classification of volatility conditions."""

    VERY_LOW = "very_low"  # ATR ratio < 0.5x average
    LOW = "low"  # ATR ratio 0.5x-0.8x average
    NORMAL = "normal"  # ATR ratio 0.8x-1.2x average
    HIGH = "high"  # ATR ratio 1.2x-2.0x average
    EXTREME = "extreme"  # ATR ratio > 2.0x average


@dataclass
class ATRResult:
    """ATR calculation result."""

    current_atr: float
    atr_percent: float  # ATR as percentage of price
    avg_atr_20: float  # 20-period average ATR
    atr_ratio: float  # current ATR / avg ATR (volatility expansion/contraction)
    volatility_regime: VolatilityRegime


@dataclass
class VolatilityAdjustedLevels:
    """Stop loss and target levels adjusted by ATR."""

    entry_price: float
    stop_loss: float
    target_1: float  # 2x ATR target
    target_2: float  # 3x ATR target
    atr_value: float
    atr_multiplier_stop: float
    atr_multiplier_target: float
    risk_per_share: float
    reward_per_share: float
    risk_reward_ratio: float


class ATRCalculator:
    """Calculator for Average True Range and volatility metrics."""

    def __init__(self, period: int = 14) -> None:
        """Initialize ATR calculator.

        Args:
            period: Period for ATR calculation (default 14)
        """
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate ATR values.

        True Range = max of:
        1. Current High - Current Low
        2. abs(Current High - Previous Close)
        3. abs(Current Low - Previous Close)

        ATR = Rolling mean of True Range

        Args:
            data: DataFrame with 'high', 'low', 'close' columns

        Returns:
            Series with ATR values
        """
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # True Range components
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Wilder's smoothing (same as used in RSI)
        atr = pd.Series(index=data.index, dtype=float)
        atr.iloc[self.period - 1] = true_range.iloc[: self.period].mean()

        for i in range(self.period, len(true_range)):
            atr.iloc[i] = (atr.iloc[i - 1] * (self.period - 1) + true_range.iloc[i]) / self.period

        return atr

    def get_volatility_regime(self, atr_ratio: float) -> VolatilityRegime:
        """Classify current volatility regime.

        Args:
            atr_ratio: Current ATR / Average ATR

        Returns:
            VolatilityRegime classification
        """
        if atr_ratio < 0.5:
            return VolatilityRegime.VERY_LOW
        elif atr_ratio < 0.8:
            return VolatilityRegime.LOW
        elif atr_ratio < 1.2:
            return VolatilityRegime.NORMAL
        elif atr_ratio < 2.0:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME

    def analyze(self, data: pd.DataFrame) -> ATRResult:
        """Perform complete ATR analysis.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            ATRResult with volatility metrics
        """
        atr_series = self.calculate(data)
        current_atr = float(atr_series.iloc[-1])
        current_price = float(data["close"].iloc[-1])

        # Guard against zero/negative price
        if current_price <= 0:
            return ATRResult(
                current_atr=current_atr,
                atr_percent=0.0,
                avg_atr_20=current_atr,
                atr_ratio=1.0,
                volatility_regime=VolatilityRegime.NORMAL,
            )

        atr_percent = (current_atr / current_price) * 100

        # Average ATR over last 20 periods for comparison
        recent_atr = atr_series.dropna().iloc[-20:]
        avg_atr_20 = float(recent_atr.mean()) if len(recent_atr) >= 20 else current_atr

        atr_ratio = current_atr / avg_atr_20 if avg_atr_20 > 0 else 1.0
        volatility_regime = self.get_volatility_regime(atr_ratio)

        return ATRResult(
            current_atr=current_atr,
            atr_percent=atr_percent,
            avg_atr_20=avg_atr_20,
            atr_ratio=atr_ratio,
            volatility_regime=volatility_regime,
        )

    def get_volatility_adjusted_levels(
        self,
        data: pd.DataFrame,
        is_long: bool = True,
        stop_multiplier: float = 1.5,
        target_multiplier: float = 3.0,
    ) -> VolatilityAdjustedLevels:
        """Calculate volatility-adjusted stop loss and target levels.

        Instead of fixed percentage stops, this uses ATR to adapt to
        each stock's actual volatility.

        Args:
            data: DataFrame with OHLCV data
            is_long: True for long positions, False for short
            stop_multiplier: ATR multiplier for stop loss (default 1.5)
            target_multiplier: ATR multiplier for target (default 3.0)

        Returns:
            VolatilityAdjustedLevels with entry, stop, and targets
        """
        atr_result = self.analyze(data)
        entry_price = float(data["close"].iloc[-1])
        atr = atr_result.current_atr

        if is_long:
            stop_loss = entry_price - (atr * stop_multiplier)
            target_1 = entry_price + (atr * 2.0)
            target_2 = entry_price + (atr * target_multiplier)
        else:
            stop_loss = entry_price + (atr * stop_multiplier)
            target_1 = entry_price - (atr * 2.0)
            target_2 = entry_price - (atr * target_multiplier)

        risk_per_share = abs(entry_price - stop_loss)
        reward_per_share = abs(target_1 - entry_price)
        risk_reward = reward_per_share / risk_per_share if risk_per_share > 0 else 0

        return VolatilityAdjustedLevels(
            entry_price=entry_price,
            stop_loss=round(stop_loss, 2),
            target_1=round(target_1, 2),
            target_2=round(target_2, 2),
            atr_value=atr,
            atr_multiplier_stop=stop_multiplier,
            atr_multiplier_target=target_multiplier,
            risk_per_share=round(risk_per_share, 2),
            reward_per_share=round(reward_per_share, 2),
            risk_reward_ratio=round(risk_reward, 2),
        )

    def get_position_size(
        self,
        data: pd.DataFrame,
        capital: float,
        risk_percent: float = 1.0,
        stop_multiplier: float = 1.5,
    ) -> dict:
        """Calculate position size based on ATR and risk tolerance.

        Uses fixed fractional position sizing: risk a fixed percentage of
        capital on each trade, adjusted by the stock's volatility.

        Args:
            data: DataFrame with OHLCV data
            capital: Total available capital
            risk_percent: Maximum risk per trade as percentage of capital
            stop_multiplier: ATR multiplier for stop loss

        Returns:
            Dictionary with position sizing details
        """
        atr_result = self.analyze(data)
        entry_price = float(data["close"].iloc[-1])

        # Penny stock guard: too risky for systematic trading
        if entry_price < 5:
            return {
                "shares": 0,
                "position_value": 0,
                "risk_amount": 0,
                "risk_per_share": 0,
                "allocation_percent": 0,
                "reason": "Penny stock (price < 5) - too risky for systematic trading",
            }

        risk_amount = capital * (risk_percent / 100)
        risk_per_share = atr_result.current_atr * stop_multiplier

        if risk_per_share <= 0:
            return {
                "shares": 0,
                "position_value": 0,
                "risk_amount": risk_amount,
                "risk_per_share": 0,
                "allocation_percent": 0,
                "reason": "ATR is zero or negative",
            }

        shares = int(risk_amount / risk_per_share)
        position_value = shares * entry_price
        allocation_percent = (position_value / capital) * 100

        return {
            "shares": shares,
            "position_value": round(position_value, 2),
            "risk_amount": round(risk_amount, 2),
            "risk_per_share": round(risk_per_share, 2),
            "allocation_percent": round(allocation_percent, 2),
            "entry_price": entry_price,
            "atr": round(atr_result.current_atr, 2),
            "atr_percent": round(atr_result.atr_percent, 2),
            "volatility_regime": atr_result.volatility_regime.value,
        }
