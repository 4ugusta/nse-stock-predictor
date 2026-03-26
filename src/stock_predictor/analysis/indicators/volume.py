"""Volume analysis indicators for confirmation of price moves."""

from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np


class VolumeSignal(Enum):
    """Volume signal classification."""

    STRONG_BULLISH = "strong_bullish"  # High volume on up move
    BULLISH = "bullish"  # Above average volume on up move
    NEUTRAL = "neutral"  # Average volume
    BEARISH = "bearish"  # Above average volume on down move
    STRONG_BEARISH = "strong_bearish"  # High volume on down move
    DISTRIBUTION = "distribution"  # High volume with price weakness
    ACCUMULATION = "accumulation"  # High volume with price strength


@dataclass
class VolumeAnalysis:
    """Result of volume analysis."""

    current_volume: int
    avg_volume_20: float
    avg_volume_50: float
    volume_ratio: float  # current / avg20
    volume_trend: str  # "increasing", "decreasing", "stable"
    signal: VolumeSignal
    is_volume_confirmed: bool  # True if volume supports the price move
    on_balance_volume_trend: str  # "bullish", "bearish", "neutral"


class VolumeAnalyzer:
    """Analyzer for volume-based indicators."""

    def __init__(self) -> None:
        """Initialize volume analyzer."""
        pass

    def calculate_volume_ratio(self, data: pd.DataFrame, period: int = 20) -> float:
        """Calculate current volume relative to average.

        Args:
            data: DataFrame with OHLCV data
            period: Period for average calculation

        Returns:
            Volume ratio (current / average)
        """
        # Use median instead of mean to resist volume spike distortion
        median_volume = data["volume"].iloc[-period:].median()
        current_volume = data["volume"].iloc[-1]
        # Dead stock: zero median volume means NO liquidity, not average
        return current_volume / median_volume if median_volume > 0 else 0.0

    def calculate_notional_turnover(self, data: pd.DataFrame, period: int = 20) -> float:
        """Calculate average daily notional turnover in INR.

        Notional volume = price * shares traded. More meaningful than raw
        share count because 50K shares at Rs 5 is very different from
        50K shares at Rs 5000.

        Args:
            data: DataFrame with OHLCV data
            period: Period for average calculation

        Returns:
            Average daily notional turnover in INR
        """
        if len(data) < period:
            period = len(data)
        recent = data.iloc[-period:]
        daily_turnover = recent["close"] * recent["volume"]
        return float(daily_turnover.mean())

    def calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume (OBV).

        OBV adds volume on up days and subtracts on down days.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Series with OBV values
        """
        obv = pd.Series(index=data.index, dtype=float)
        obv.iloc[0] = 0

        for i in range(1, len(data)):
            close_curr = data["close"].iloc[i]
            close_prev = data["close"].iloc[i - 1]
            vol = data["volume"].iloc[i]

            # NaN guard: skip NaN values, carry forward previous OBV
            if pd.isna(close_curr) or pd.isna(close_prev) or pd.isna(vol):
                obv.iloc[i] = obv.iloc[i - 1]
            elif close_curr > close_prev:
                obv.iloc[i] = obv.iloc[i - 1] + vol
            elif close_curr < close_prev:
                obv.iloc[i] = obv.iloc[i - 1] - vol
            else:
                obv.iloc[i] = obv.iloc[i - 1]

        return obv

    def get_obv_trend(self, data: pd.DataFrame, period: int = 20) -> str:
        """Determine OBV trend direction.

        Args:
            data: DataFrame with OHLCV data
            period: Period for trend calculation

        Returns:
            "bullish", "bearish", or "neutral"
        """
        obv = self.calculate_obv(data)
        obv_sma = obv.rolling(window=period).mean()

        if obv.iloc[-1] > obv_sma.iloc[-1] * 1.10:
            return "bullish"
        elif obv.iloc[-1] < obv_sma.iloc[-1] * 0.90:
            return "bearish"
        else:
            return "neutral"

    def get_volume_trend(self, data: pd.DataFrame, period: int = 10) -> str:
        """Determine if volume is increasing or decreasing.

        Args:
            data: DataFrame with OHLCV data
            period: Period for trend calculation

        Returns:
            "increasing", "decreasing", or "stable"
        """
        recent_avg = data["volume"].iloc[-period:].mean()
        prior_avg = data["volume"].iloc[-2 * period : -period].mean()

        if recent_avg > prior_avg * 1.2:
            return "increasing"
        elif recent_avg < prior_avg * 0.8:
            return "decreasing"
        else:
            return "stable"

    def analyze(self, data: pd.DataFrame) -> VolumeAnalysis:
        """Perform complete volume analysis.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            VolumeAnalysis with all volume metrics
        """
        current_volume = int(data["volume"].iloc[-1])
        avg_volume_20 = data["volume"].iloc[-20:].mean()
        avg_volume_50 = data["volume"].iloc[-50:].mean() if len(data) >= 50 else avg_volume_20

        volume_ratio = self.calculate_volume_ratio(data)
        volume_trend = self.get_volume_trend(data)
        obv_trend = self.get_obv_trend(data)

        # Determine price direction
        price_change = data["close"].iloc[-1] - data["close"].iloc[-2]
        price_up = price_change > 0

        # Determine volume signal
        signal = self._determine_signal(volume_ratio, price_up, obv_trend)

        # Check if volume confirms price action
        is_confirmed = self._is_volume_confirmed(volume_ratio, price_up, obv_trend)

        return VolumeAnalysis(
            current_volume=current_volume,
            avg_volume_20=avg_volume_20,
            avg_volume_50=avg_volume_50,
            volume_ratio=volume_ratio,
            volume_trend=volume_trend,
            signal=signal,
            is_volume_confirmed=is_confirmed,
            on_balance_volume_trend=obv_trend,
        )

    def _determine_signal(
        self, volume_ratio: float, price_up: bool, obv_trend: str
    ) -> VolumeSignal:
        """Determine volume signal based on price and volume relationship.

        Args:
            volume_ratio: Current volume / average volume
            price_up: True if price went up
            obv_trend: OBV trend direction

        Returns:
            VolumeSignal classification
        """
        high_volume = volume_ratio >= 1.5
        above_avg = volume_ratio >= 1.0

        if high_volume and price_up and obv_trend == "bullish":
            return VolumeSignal.STRONG_BULLISH
        elif high_volume and not price_up and obv_trend == "bearish":
            return VolumeSignal.STRONG_BEARISH
        elif above_avg and price_up:
            return VolumeSignal.BULLISH
        elif above_avg and not price_up:
            return VolumeSignal.BEARISH
        elif high_volume and not price_up and obv_trend != "bearish":
            return VolumeSignal.DISTRIBUTION
        elif high_volume and price_up and obv_trend != "bullish":
            return VolumeSignal.ACCUMULATION
        else:
            return VolumeSignal.NEUTRAL

    def _is_volume_confirmed(
        self, volume_ratio: float, price_up: bool, obv_trend: str
    ) -> bool:
        """Check if volume confirms the price move.

        Asymmetric thresholds: selling pressure typically shows higher volume
        than buying pressure, so bearish confirmation requires more volume.

        Args:
            volume_ratio: Current volume / average volume
            price_up: True if price went up
            obv_trend: OBV trend direction

        Returns:
            True if volume confirms price action
        """
        if price_up:
            # Bullish: 0.9x average volume is sufficient with OBV confirmation
            return volume_ratio >= 0.9 and obv_trend == "bullish"
        else:
            # Bearish: require higher volume (1.2x) as panic selling spikes volume
            return volume_ratio >= 1.2 and obv_trend == "bearish"

    def is_safe_to_buy(self, data: pd.DataFrame) -> tuple[bool, str]:
        """Check if volume supports a buy decision.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Tuple of (is_safe, reason)
        """
        analysis = self.analyze(data)

        # Avoid distribution patterns
        if analysis.signal == VolumeSignal.DISTRIBUTION:
            return False, "Distribution pattern detected (selling pressure)"

        # Avoid strong bearish volume
        if analysis.signal == VolumeSignal.STRONG_BEARISH:
            return False, "Heavy selling volume"

        # Require at least neutral OBV
        if analysis.on_balance_volume_trend == "bearish":
            return False, f"OBV trending bearish (money leaving)"

        # Check for declining volume on rallies (weak rally)
        if analysis.volume_trend == "decreasing" and analysis.volume_ratio < 0.5:
            return False, "Very low volume - lack of conviction"

        return True, f"Volume confirmed (ratio: {analysis.volume_ratio:.1f}x, OBV: {analysis.on_balance_volume_trend})"
