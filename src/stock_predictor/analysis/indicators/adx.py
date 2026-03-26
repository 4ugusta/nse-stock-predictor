"""ADX (Average Directional Index) indicator for trend strength measurement.

Improvements over basic ADX:
1. Gradual regime blending instead of binary cliff transitions
2. ADX slope detection (strengthening vs weakening trends)
3. Volatile regime support for high-ADX close-DI markets
4. Regime persistence to avoid daily strategy flipping
"""

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import pandas_ta as ta


class TrendStrength(Enum):
    """Classification of trend strength based on ADX value."""

    NO_TREND = "no_trend"  # ADX < 20
    WEAK = "weak"  # ADX 20-25
    MODERATE = "moderate"  # ADX 25-35
    STRONG = "strong"  # ADX 35-50
    VERY_STRONG = "very_strong"  # ADX > 50


class MarketRegime(Enum):
    """Overall market regime classification."""

    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"  # High ADX but close DI lines — choppy
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"


@dataclass
class ADXAnalysis:
    """Result of ADX analysis."""

    adx_value: float
    plus_di: float  # +DI (bullish directional indicator)
    minus_di: float  # -DI (bearish directional indicator)
    trend_strength: TrendStrength
    trend_direction: str  # "bullish", "bearish", or "neutral"
    is_trending: bool  # True if ADX > 25
    regime: MarketRegime
    adx_slope: float = 0.0  # Positive = strengthening, negative = weakening
    regime_confidence: float = 1.0  # 0-1, how confident we are in the regime


class ADXCalculator:
    """Calculator for ADX and related directional indicators."""

    # Regime persistence: require N consecutive days in new regime before switching
    REGIME_PERSISTENCE_DAYS = 3

    def __init__(self, period: int = 14) -> None:
        """Initialize ADX calculator.

        Args:
            period: Period for ADX calculation (default 14)
        """
        self.period = period
        self._prev_regime: MarketRegime | None = None
        self._regime_streak: int = 0

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate ADX, +DI, and -DI.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with ADX, +DI, -DI columns
        """
        adx_df = ta.adx(data["high"], data["low"], data["close"], length=self.period)
        return adx_df

    def get_trend_strength(self, adx_value: float) -> TrendStrength:
        """Classify trend strength based on ADX value.

        Args:
            adx_value: Current ADX value

        Returns:
            TrendStrength classification
        """
        if adx_value < 20:
            return TrendStrength.NO_TREND
        elif adx_value < 25:
            return TrendStrength.WEAK
        elif adx_value < 35:
            return TrendStrength.MODERATE
        elif adx_value < 50:
            return TrendStrength.STRONG
        else:
            return TrendStrength.VERY_STRONG

    def _calculate_adx_slope(self, adx_df: pd.DataFrame) -> float:
        """Calculate ADX slope (direction of trend strength change).

        Positive slope = trend strengthening, negative = trend weakening.
        Uses 5-period slope for smoothness.

        Returns:
            Slope value (roughly -5 to +5 range)
        """
        adx_col = f"ADX_{self.period}"
        adx_series = adx_df[adx_col].dropna()
        if len(adx_series) < 5:
            return 0.0

        recent = adx_series.iloc[-5:]
        # Simple linear regression slope
        x = range(len(recent))
        x_mean = sum(x) / len(x)
        y_mean = float(recent.mean())
        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, recent))
        denominator = sum((xi - x_mean) ** 2 for xi in x)
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def analyze(self, data: pd.DataFrame) -> ADXAnalysis:
        """Perform complete ADX analysis.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            ADXAnalysis with trend strength and direction
        """
        adx_df = self.calculate(data)

        # Get latest values
        adx_col = f"ADX_{self.period}"
        dmp_col = f"DMP_{self.period}"  # +DI
        dmn_col = f"DMN_{self.period}"  # -DI

        adx_value = float(adx_df[adx_col].iloc[-1])
        plus_di = float(adx_df[dmp_col].iloc[-1])
        minus_di = float(adx_df[dmn_col].iloc[-1])

        # NaN guard: if ADX value is NaN, return default neutral analysis
        if pd.isna(adx_value) or pd.isna(plus_di) or pd.isna(minus_di):
            return ADXAnalysis(
                adx_value=0.0,
                plus_di=0.0,
                minus_di=0.0,
                trend_strength=TrendStrength.NO_TREND,
                trend_direction="neutral",
                is_trending=False,
                regime=MarketRegime.SIDEWAYS,
                adx_slope=0.0,
                regime_confidence=0.0,
            )

        # Determine trend direction
        if plus_di > minus_di:
            trend_direction = "bullish"
        elif minus_di > plus_di:
            trend_direction = "bearish"
        else:
            trend_direction = "neutral"

        # Get trend strength
        trend_strength = self.get_trend_strength(adx_value)
        is_trending = adx_value >= 25

        # Calculate ADX slope (strengthening vs weakening)
        adx_slope = self._calculate_adx_slope(adx_df)

        # Determine market regime with gradual blending
        raw_regime, regime_confidence = self._determine_regime(
            adx_value, plus_di, minus_di
        )

        # Apply regime persistence to avoid daily flipping
        regime = self._apply_persistence(raw_regime)

        return ADXAnalysis(
            adx_value=adx_value,
            plus_di=plus_di,
            minus_di=minus_di,
            trend_strength=trend_strength,
            trend_direction=trend_direction,
            is_trending=is_trending,
            regime=regime,
            adx_slope=adx_slope,
            regime_confidence=regime_confidence,
        )

    def _determine_regime(
        self,
        adx_value: float,
        plus_di: float,
        minus_di: float,
    ) -> tuple[MarketRegime, float]:
        """Determine overall market regime with gradual transition zones.

        Instead of binary cliff transitions, uses a blending zone around
        thresholds. Returns both the regime and a confidence score (0-1)
        indicating how firmly we are in that regime.

        Returns:
            Tuple of (MarketRegime, confidence 0-1)
        """
        # Transition zone: 18-22 is the gradual boundary for sideways
        SIDEWAYS_LOWER = 18.0
        SIDEWAYS_UPPER = 22.0
        STRONG_THRESHOLD = 40.0

        # Pure sideways
        if adx_value < SIDEWAYS_LOWER:
            return MarketRegime.SIDEWAYS, 1.0

        di_diff = plus_di - minus_di
        di_sum = plus_di + minus_di
        relative_diff = abs(di_diff) / di_sum * 100 if di_sum > 0 else 0

        # Transition zone: blend confidence
        if adx_value < SIDEWAYS_UPPER:
            # Linearly interpolate confidence between sideways and trending
            blend = (adx_value - SIDEWAYS_LOWER) / (SIDEWAYS_UPPER - SIDEWAYS_LOWER)
            # In transition zone, still classify direction but with low confidence
            if relative_diff <= 10:
                return MarketRegime.SIDEWAYS, 1.0 - blend * 0.5
            # Has direction but weak ADX
            if di_diff > 0:
                return MarketRegime.UPTREND, blend * 0.5
            else:
                return MarketRegime.DOWNTREND, blend * 0.5

        # ADX >= 22: clearly above sideways threshold
        # HIGH ADX but close DI lines = VOLATILE regime (not sideways)
        if relative_diff <= 10 and adx_value >= 25:
            # High ADX with close DI: volatile/choppy, not sideways
            return MarketRegime.VOLATILE, min(1.0, adx_value / 40)

        # Gradual DI separation confidence: 10-20% relative_diff is transition zone
        if relative_diff <= 10:
            return MarketRegime.SIDEWAYS, 0.6

        # Direction is clear (relative_diff > 10)
        di_confidence = min(1.0, (relative_diff - 10) / 15)  # Ramp from 10% to 25%
        adx_confidence = min(1.0, adx_value / 35)  # Ramp from 0 to 35
        confidence = di_confidence * adx_confidence

        if di_diff > 0:  # Bullish
            if adx_value >= STRONG_THRESHOLD:
                return MarketRegime.STRONG_UPTREND, confidence
            return MarketRegime.UPTREND, confidence
        else:  # Bearish
            if adx_value >= STRONG_THRESHOLD:
                return MarketRegime.STRONG_DOWNTREND, confidence
            return MarketRegime.DOWNTREND, confidence

    def _apply_persistence(self, new_regime: MarketRegime) -> MarketRegime:
        """Apply regime persistence to prevent daily flipping.

        Requires REGIME_PERSISTENCE_DAYS consecutive days in a new regime
        before switching away from the current regime.
        """
        if self._prev_regime is None:
            self._prev_regime = new_regime
            self._regime_streak = 1
            return new_regime

        if new_regime == self._prev_regime:
            self._regime_streak += 1
            return new_regime

        # New regime differs — increment streak but don't switch until threshold
        self._regime_streak += 1
        if self._regime_streak >= self.REGIME_PERSISTENCE_DAYS:
            self._prev_regime = new_regime
            self._regime_streak = 1
            return new_regime

        # Not enough consecutive days — stick with previous regime
        return self._prev_regime

    def is_safe_to_buy(self, data: pd.DataFrame) -> tuple[bool, str]:
        """Check if it's safe to buy based on ADX analysis.

        This is a key filter to avoid buying in weak/choppy markets.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Tuple of (is_safe, reason)
        """
        analysis = self.analyze(data)

        # Avoid sideways markets
        if analysis.regime == MarketRegime.SIDEWAYS:
            return False, f"Sideways market (ADX: {analysis.adx_value:.1f})"

        # Avoid volatile/choppy markets
        if analysis.regime == MarketRegime.VOLATILE:
            return False, f"Volatile/choppy market (ADX: {analysis.adx_value:.1f}, DI lines close)"

        # Avoid strong downtrends
        if analysis.regime in (MarketRegime.DOWNTREND, MarketRegime.STRONG_DOWNTREND):
            return False, f"Downtrend detected (+DI: {analysis.plus_di:.1f}, -DI: {analysis.minus_di:.1f})"

        # Require minimum trend strength for buys
        if analysis.adx_value < 20:
            return False, f"No clear trend (ADX: {analysis.adx_value:.1f})"

        # Warn if ADX is weakening (trend losing steam)
        if analysis.adx_slope < -1.0:
            return True, (
                f"Uptrend confirmed but weakening (ADX: {analysis.adx_value:.1f}, "
                f"slope: {analysis.adx_slope:.1f}, +DI: {analysis.plus_di:.1f})"
            )

        # Good to buy
        return True, f"Uptrend confirmed (ADX: {analysis.adx_value:.1f}, +DI: {analysis.plus_di:.1f})"
