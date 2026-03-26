"""Candlestick pattern recognition for leading price signals.

Unlike lagging indicators (MA, MACD, RSI), candlestick patterns are LEADING
indicators that identify potential reversals and continuations before they
are confirmed by other indicators.

Patterns detected:
- Reversal: Hammer, Inverted Hammer, Engulfing, Doji, Morning/Evening Star
- Continuation: Three White Soldiers, Three Black Crows, Marubozu
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class PatternType(Enum):
    """Type of candlestick pattern."""

    BULLISH_REVERSAL = "bullish_reversal"
    BEARISH_REVERSAL = "bearish_reversal"
    BULLISH_CONTINUATION = "bullish_continuation"
    BEARISH_CONTINUATION = "bearish_continuation"
    INDECISION = "indecision"


class PatternStrength(Enum):
    """Reliability of the pattern."""

    STRONG = "strong"  # High reliability (e.g., engulfing with volume)
    MODERATE = "moderate"
    WEAK = "weak"  # Low reliability (e.g., single candle patterns)


@dataclass
class CandlestickPattern:
    """A detected candlestick pattern."""

    name: str
    pattern_type: PatternType
    strength: PatternStrength
    score: float  # -1.0 (strongly bearish) to 1.0 (strongly bullish)
    description: str
    candles_used: int  # Number of candles in the pattern


@dataclass
class PatternAnalysis:
    """Complete candlestick pattern analysis."""

    patterns_found: list[CandlestickPattern]
    net_score: float  # Combined score of all patterns
    dominant_signal: str  # "bullish", "bearish", or "neutral"
    summary: str


class CandlestickPatternDetector:
    """Detects candlestick patterns in OHLCV data."""

    # Empirical reliability weights (based on research, 1.0 = baseline)
    PATTERN_RELIABILITY: dict[str, float] = {
        "Hammer": 1.2,
        "Inverted Hammer": 0.8,
        "Bullish Engulfing": 1.3,
        "Bearish Engulfing": 1.3,
        "Morning Star": 1.4,
        "Evening Star": 1.4,
        "Doji": 0.6,  # Doji alone is weak
        "Dragonfly Doji": 0.9,
        "Gravestone Doji": 0.9,
        "Spinning Top": 0.5,
        "Bullish Marubozu": 1.1,
        "Bearish Marubozu": 1.1,
        "Shooting Star": 1.0,
        "Three White Soldiers": 1.3,
        "Three Black Crows": 1.3,
    }

    def __init__(self, body_threshold: float = 0.3) -> None:
        """Initialize detector.

        Args:
            body_threshold: Minimum body-to-range ratio for "real body" candles
        """
        self.body_threshold = body_threshold

    def _body_size(self, row: pd.Series) -> float:
        """Calculate absolute body size."""
        return abs(row["close"] - row["open"])

    def _candle_range(self, row: pd.Series) -> float:
        """Calculate total candle range (high - low)."""
        return row["high"] - row["low"]

    def _body_ratio(self, row: pd.Series) -> float:
        """Calculate body to range ratio."""
        rng = self._candle_range(row)
        # Zero range (high == low): return 0.5 (neutral) instead of 0.0
        return self._body_size(row) / rng if rng > 0 else 0.5

    def _is_bullish(self, row: pd.Series) -> bool:
        """Check if candle is bullish (close > open)."""
        return row["close"] > row["open"]

    def _upper_shadow(self, row: pd.Series) -> float:
        """Calculate upper shadow length."""
        return row["high"] - max(row["open"], row["close"])

    def _lower_shadow(self, row: pd.Series) -> float:
        """Calculate lower shadow length."""
        return min(row["open"], row["close"]) - row["low"]

    def detect_doji(self, data: pd.DataFrame) -> CandlestickPattern | None:
        """Detect Doji pattern (indecision candle).

        A Doji has a very small body relative to its range, indicating
        market indecision and potential reversal.
        """
        row = data.iloc[-1]
        rng = self._candle_range(row)
        if rng == 0:
            return None

        body_ratio = self._body_ratio(row)

        if body_ratio < 0.1:  # Body is less than 10% of range
            return CandlestickPattern(
                name="Doji",
                pattern_type=PatternType.INDECISION,
                strength=PatternStrength.MODERATE,
                score=0.0,
                description="Doji: Market indecision, potential reversal",
                candles_used=1,
            )
        return None

    def detect_hammer(self, data: pd.DataFrame) -> CandlestickPattern | None:
        """Detect Hammer pattern (bullish reversal after downtrend).

        Hammer has a small body at the top, long lower shadow (2x+ body),
        and minimal upper shadow. Must appear after a decline.
        """
        if len(data) < 5:
            return None

        row = data.iloc[-1]
        rng = self._candle_range(row)
        if rng == 0:
            return None

        body = self._body_size(row)
        lower_shadow = self._lower_shadow(row)
        upper_shadow = self._upper_shadow(row)

        # Hammer criteria
        is_hammer = (
            lower_shadow >= body * 2
            and upper_shadow <= body * 0.5
            and body > 0
        )

        if not is_hammer:
            return None

        # Verify downtrend context (price declining over last 10 candles + below short MA)
        prices = data["close"].iloc[-11:-1]
        sma_10 = data["close"].iloc[-11:].mean()
        is_downtrend = len(prices) >= 5 and prices.iloc[-1] < prices.iloc[0] and row["close"] < sma_10

        if is_downtrend:
            return CandlestickPattern(
                name="Hammer",
                pattern_type=PatternType.BULLISH_REVERSAL,
                strength=PatternStrength.MODERATE,
                score=0.5,
                description="Hammer: Buyers rejected lower prices, potential bullish reversal",
                candles_used=1,
            )
        return None

    def detect_inverted_hammer(self, data: pd.DataFrame) -> CandlestickPattern | None:
        """Detect Inverted Hammer (bullish reversal after downtrend).

        Long upper shadow, small body at bottom, minimal lower shadow.
        """
        if len(data) < 5:
            return None

        row = data.iloc[-1]
        rng = self._candle_range(row)
        if rng == 0:
            return None

        body = self._body_size(row)
        upper_shadow = self._upper_shadow(row)
        lower_shadow = self._lower_shadow(row)

        is_inverted = (
            upper_shadow >= body * 2
            and lower_shadow <= body * 0.5
            and body > 0
        )

        if not is_inverted:
            return None

        prices = data["close"].iloc[-11:-1]
        sma_10 = data["close"].iloc[-11:].mean()
        is_downtrend = len(prices) >= 5 and prices.iloc[-1] < prices.iloc[0] and row["close"] < sma_10

        if is_downtrend:
            return CandlestickPattern(
                name="Inverted Hammer",
                pattern_type=PatternType.BULLISH_REVERSAL,
                strength=PatternStrength.WEAK,
                score=0.3,
                description="Inverted Hammer: Buying pressure emerging in downtrend",
                candles_used=1,
            )
        return None

    def detect_shooting_star(self, data: pd.DataFrame) -> CandlestickPattern | None:
        """Detect Shooting Star (bearish reversal after uptrend).

        Long upper shadow, small body at bottom, minimal lower shadow.
        Same shape as inverted hammer but in an uptrend context.
        """
        if len(data) < 5:
            return None

        row = data.iloc[-1]
        rng = self._candle_range(row)
        if rng == 0:
            return None

        body = self._body_size(row)
        upper_shadow = self._upper_shadow(row)
        lower_shadow = self._lower_shadow(row)

        is_star = (
            upper_shadow >= body * 2
            and lower_shadow <= body * 0.5
            and body > 0
        )

        if not is_star:
            return None

        prices = data["close"].iloc[-11:-1]
        sma_10 = data["close"].iloc[-11:].mean()
        is_uptrend = len(prices) >= 5 and prices.iloc[-1] > prices.iloc[0] and row["close"] > sma_10

        if is_uptrend:
            return CandlestickPattern(
                name="Shooting Star",
                pattern_type=PatternType.BEARISH_REVERSAL,
                strength=PatternStrength.MODERATE,
                score=-0.5,
                description="Shooting Star: Sellers rejected higher prices, potential bearish reversal",
                candles_used=1,
            )
        return None

    def detect_bullish_engulfing(self, data: pd.DataFrame) -> CandlestickPattern | None:
        """Detect Bullish Engulfing pattern.

        A bearish candle followed by a larger bullish candle that completely
        engulfs the previous candle's body. Strong reversal signal.
        """
        if len(data) < 3:
            return None

        prev = data.iloc[-2]
        curr = data.iloc[-1]

        is_prev_bearish = prev["close"] < prev["open"]
        is_curr_bullish = curr["close"] > curr["open"]

        if not (is_prev_bearish and is_curr_bullish):
            return None

        # Current body must engulf previous body
        engulfs = curr["open"] <= prev["close"] and curr["close"] >= prev["open"]

        if engulfs:
            # Stronger if in downtrend context
            prices = data["close"].iloc[-6:-2]
            is_downtrend = len(prices) >= 2 and prices.iloc[-1] < prices.iloc[0]
            strength = PatternStrength.STRONG if is_downtrend else PatternStrength.MODERATE
            score = 0.7 if is_downtrend else 0.5

            return CandlestickPattern(
                name="Bullish Engulfing",
                pattern_type=PatternType.BULLISH_REVERSAL,
                strength=strength,
                score=score,
                description="Bullish Engulfing: Buyers overwhelmed sellers, strong reversal",
                candles_used=2,
            )
        return None

    def detect_bearish_engulfing(self, data: pd.DataFrame) -> CandlestickPattern | None:
        """Detect Bearish Engulfing pattern.

        A bullish candle followed by a larger bearish candle that completely
        engulfs the previous candle's body. Strong reversal signal.
        """
        if len(data) < 3:
            return None

        prev = data.iloc[-2]
        curr = data.iloc[-1]

        is_prev_bullish = prev["close"] > prev["open"]
        is_curr_bearish = curr["close"] < curr["open"]

        if not (is_prev_bullish and is_curr_bearish):
            return None

        engulfs = curr["open"] >= prev["close"] and curr["close"] <= prev["open"]

        if engulfs:
            prices = data["close"].iloc[-6:-2]
            is_uptrend = len(prices) >= 2 and prices.iloc[-1] > prices.iloc[0]
            strength = PatternStrength.STRONG if is_uptrend else PatternStrength.MODERATE
            score = -0.7 if is_uptrend else -0.5

            return CandlestickPattern(
                name="Bearish Engulfing",
                pattern_type=PatternType.BEARISH_REVERSAL,
                strength=strength,
                score=score,
                description="Bearish Engulfing: Sellers overwhelmed buyers, strong reversal",
                candles_used=2,
            )
        return None

    def detect_morning_star(self, data: pd.DataFrame) -> CandlestickPattern | None:
        """Detect Morning Star pattern (3-candle bullish reversal).

        1. Large bearish candle
        2. Small body candle (gap down)
        3. Large bullish candle closing above midpoint of candle 1
        """
        if len(data) < 3:
            return None

        c1 = data.iloc[-3]
        c2 = data.iloc[-2]
        c3 = data.iloc[-1]

        c1_bearish = c1["close"] < c1["open"]
        c1_large = self._body_ratio(c1) > 0.5
        c2_small = self._body_ratio(c2) < 0.3
        c3_bullish = c3["close"] > c3["open"]
        c3_large = self._body_ratio(c3) > 0.5

        midpoint_c1 = (c1["open"] + c1["close"]) / 2
        c3_above_mid = c3["close"] > midpoint_c1

        # Gap tolerance: in Indian markets, perfect gaps are rare on daily charts.
        # Accept if c2 body center is below c1 close (approximate gap down).
        c2_body_center = (c2["open"] + c2["close"]) / 2
        has_gap_or_near = c2_body_center <= c1["close"]

        if c1_bearish and c1_large and c2_small and c3_bullish and c3_large and c3_above_mid and has_gap_or_near:
            return CandlestickPattern(
                name="Morning Star",
                pattern_type=PatternType.BULLISH_REVERSAL,
                strength=PatternStrength.STRONG,
                score=0.8,
                description="Morning Star: 3-candle bullish reversal, high reliability",
                candles_used=3,
            )
        return None

    def detect_evening_star(self, data: pd.DataFrame) -> CandlestickPattern | None:
        """Detect Evening Star pattern (3-candle bearish reversal).

        1. Large bullish candle
        2. Small body candle (gap up)
        3. Large bearish candle closing below midpoint of candle 1
        """
        if len(data) < 3:
            return None

        c1 = data.iloc[-3]
        c2 = data.iloc[-2]
        c3 = data.iloc[-1]

        c1_bullish = c1["close"] > c1["open"]
        c1_large = self._body_ratio(c1) > 0.5
        c2_small = self._body_ratio(c2) < 0.3
        c3_bearish = c3["close"] < c3["open"]
        c3_large = self._body_ratio(c3) > 0.5

        midpoint_c1 = (c1["open"] + c1["close"]) / 2
        c3_below_mid = c3["close"] < midpoint_c1

        # Gap tolerance: accept if c2 body center is above c1 close (approximate gap up)
        c2_body_center = (c2["open"] + c2["close"]) / 2
        has_gap_or_near = c2_body_center >= c1["close"]

        if c1_bullish and c1_large and c2_small and c3_bearish and c3_large and c3_below_mid and has_gap_or_near:
            return CandlestickPattern(
                name="Evening Star",
                pattern_type=PatternType.BEARISH_REVERSAL,
                strength=PatternStrength.STRONG,
                score=-0.8,
                description="Evening Star: 3-candle bearish reversal, high reliability",
                candles_used=3,
            )
        return None

    def detect_three_white_soldiers(self, data: pd.DataFrame) -> CandlestickPattern | None:
        """Detect Three White Soldiers (bullish continuation).

        Three consecutive bullish candles with higher closes and small upper shadows.
        """
        if len(data) < 3:
            return None

        c1 = data.iloc[-3]
        c2 = data.iloc[-2]
        c3 = data.iloc[-1]

        all_bullish = all(
            data.iloc[i]["close"] > data.iloc[i]["open"] for i in [-3, -2, -1]
        )
        higher_closes = c3["close"] > c2["close"] > c1["close"]
        higher_opens = c3["open"] > c2["open"] > c1["open"]

        # Each candle should have small upper shadow
        small_shadows = all(
            self._upper_shadow(data.iloc[i]) < self._body_size(data.iloc[i]) * 0.5
            for i in [-3, -2, -1]
        )

        if all_bullish and higher_closes and higher_opens and small_shadows:
            return CandlestickPattern(
                name="Three White Soldiers",
                pattern_type=PatternType.BULLISH_CONTINUATION,
                strength=PatternStrength.STRONG,
                score=0.7,
                description="Three White Soldiers: Strong bullish momentum continuation",
                candles_used=3,
            )
        return None

    def detect_three_black_crows(self, data: pd.DataFrame) -> CandlestickPattern | None:
        """Detect Three Black Crows (bearish continuation).

        Three consecutive bearish candles with lower closes and small lower shadows.
        """
        if len(data) < 3:
            return None

        c1 = data.iloc[-3]
        c2 = data.iloc[-2]
        c3 = data.iloc[-1]

        all_bearish = all(
            data.iloc[i]["close"] < data.iloc[i]["open"] for i in [-3, -2, -1]
        )
        lower_closes = c3["close"] < c2["close"] < c1["close"]
        lower_opens = c3["open"] < c2["open"] < c1["open"]

        small_shadows = all(
            self._lower_shadow(data.iloc[i]) < self._body_size(data.iloc[i]) * 0.5
            for i in [-3, -2, -1]
        )

        if all_bearish and lower_closes and lower_opens and small_shadows:
            return CandlestickPattern(
                name="Three Black Crows",
                pattern_type=PatternType.BEARISH_CONTINUATION,
                strength=PatternStrength.STRONG,
                score=-0.7,
                description="Three Black Crows: Strong bearish momentum continuation",
                candles_used=3,
            )
        return None

    def detect_marubozu(self, data: pd.DataFrame) -> CandlestickPattern | None:
        """Detect Marubozu (strong conviction candle).

        A candle with no (or very small) shadows, indicating strong conviction.
        """
        row = data.iloc[-1]
        rng = self._candle_range(row)
        if rng == 0:
            return None

        body_ratio = self._body_ratio(row)
        is_bullish = self._is_bullish(row)

        if body_ratio >= 0.75:  # Body is 75%+ of total range
            if is_bullish:
                return CandlestickPattern(
                    name="Bullish Marubozu",
                    pattern_type=PatternType.BULLISH_CONTINUATION,
                    strength=PatternStrength.MODERATE,
                    score=0.5,
                    description="Bullish Marubozu: Strong buying conviction, no pullback",
                    candles_used=1,
                )
            else:
                return CandlestickPattern(
                    name="Bearish Marubozu",
                    pattern_type=PatternType.BEARISH_CONTINUATION,
                    strength=PatternStrength.MODERATE,
                    score=-0.5,
                    description="Bearish Marubozu: Strong selling conviction, no bounce",
                    candles_used=1,
                )
        return None

    def analyze(self, data: pd.DataFrame) -> PatternAnalysis:
        """Run all pattern detectors and return combined analysis.

        Args:
            data: DataFrame with OHLCV data (at least 5 rows)

        Returns:
            PatternAnalysis with all detected patterns
        """
        if len(data) < 5:
            return PatternAnalysis(
                patterns_found=[],
                net_score=0.0,
                dominant_signal="neutral",
                summary="Insufficient data for pattern detection",
            )

        detectors = [
            self.detect_doji,
            self.detect_hammer,
            self.detect_inverted_hammer,
            self.detect_shooting_star,
            self.detect_bullish_engulfing,
            self.detect_bearish_engulfing,
            self.detect_morning_star,
            self.detect_evening_star,
            self.detect_three_white_soldiers,
            self.detect_three_black_crows,
            self.detect_marubozu,
        ]

        patterns = []
        for detector in detectors:
            result = detector(data)
            if result is not None:
                patterns.append(result)

        if not patterns:
            return PatternAnalysis(
                patterns_found=[],
                net_score=0.0,
                dominant_signal="neutral",
                summary="No significant candlestick patterns detected",
            )

        # Scale pattern scores by empirical reliability weights
        for p in patterns:
            reliability = self.PATTERN_RELIABILITY.get(p.name, 1.0)
            p.score = p.score * reliability

        # Volume confirmation: boost patterns with above-average volume,
        # dampen patterns on low volume (unreliable without conviction)
        if "volume" in data.columns and len(data) >= 20:
            avg_vol = data["volume"].iloc[-20:].mean()
            current_vol = data["volume"].iloc[-1]
            if avg_vol > 0:
                vol_ratio = current_vol / avg_vol
                for p in patterns:
                    if vol_ratio >= 1.5:
                        p.score *= 1.3  # Strong volume confirmation
                        p.strength = PatternStrength.STRONG
                    elif vol_ratio >= 1.0:
                        pass  # Normal volume — no adjustment
                    elif vol_ratio >= 0.5:
                        p.score *= 0.7  # Below-average volume — weaker signal
                    else:
                        p.score *= 0.4  # Very low volume — unreliable pattern
                        p.strength = PatternStrength.WEAK

        # Calculate net score weighted by pattern strength
        strength_weights = {
            PatternStrength.STRONG: 1.0,
            PatternStrength.MODERATE: 0.6,
            PatternStrength.WEAK: 0.3,
        }

        total_weight = sum(strength_weights[p.strength] for p in patterns)
        net_score = sum(
            p.score * strength_weights[p.strength] for p in patterns
        ) / total_weight if total_weight > 0 else 0.0

        # Clamp to [-1, 1]
        net_score = max(-1.0, min(1.0, net_score))

        if net_score > 0.2:
            dominant = "bullish"
        elif net_score < -0.2:
            dominant = "bearish"
        else:
            dominant = "neutral"

        pattern_names = [p.name for p in patterns]
        summary = f"Detected: {', '.join(pattern_names)} (net score: {net_score:+.2f})"

        return PatternAnalysis(
            patterns_found=patterns,
            net_score=net_score,
            dominant_signal=dominant,
            summary=summary,
        )
