"""MACD (Moving Average Convergence Divergence) indicator calculations."""

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from stock_predictor.infrastructure.config.constants import INDICATOR_DEFAULTS


class MACDSignalType(Enum):
    """Type of MACD signal."""

    BULLISH_CROSSOVER = "bullish_crossover"  # MACD crosses above signal
    BEARISH_CROSSOVER = "bearish_crossover"  # MACD crosses below signal
    BULLISH_DIVERGENCE = "bullish_divergence"
    BEARISH_DIVERGENCE = "bearish_divergence"
    NONE = "none"


@dataclass
class MACDResult:
    """MACD calculation result."""

    macd_line: pd.Series
    signal_line: pd.Series
    histogram: pd.Series

    @property
    def current_macd(self) -> float:
        return self.macd_line.iloc[-1]

    @property
    def current_signal(self) -> float:
        return self.signal_line.iloc[-1]

    @property
    def current_histogram(self) -> float:
        return self.histogram.iloc[-1]


@dataclass
class MACDAnalysis:
    """Complete MACD analysis result."""

    macd_value: float
    signal_value: float
    histogram_value: float
    signal_type: MACDSignalType
    trend: str  # "bullish", "bearish", "neutral"
    signal_strength: float  # -1.0 to 1.0
    histogram_trend: str  # "increasing", "decreasing", "flat"


class MACDCalculator:
    """Calculator for MACD indicator."""

    def __init__(
        self,
        fast_period: int | None = None,
        slow_period: int | None = None,
        signal_period: int | None = None,
    ) -> None:
        """Initialize MACD calculator.

        Args:
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line period (default: 9)
        """
        self.fast_period = fast_period or INDICATOR_DEFAULTS["macd_fast"]
        self.slow_period = slow_period or INDICATOR_DEFAULTS["macd_slow"]
        self.signal_period = signal_period or INDICATOR_DEFAULTS["macd_signal"]

    def calculate(self, data: pd.DataFrame) -> MACDResult:
        """Calculate MACD lines.

        Args:
            data: DataFrame with 'close' column

        Returns:
            MACDResult with MACD line, signal line, and histogram
        """
        # Calculate EMAs
        ema_fast = data["close"].ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = data["close"].ewm(span=self.slow_period, adjust=False).mean()

        # MACD line
        macd_line = ema_fast - ema_slow

        # Signal line (EMA of MACD)
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()

        # Histogram
        histogram = macd_line - signal_line

        return MACDResult(
            macd_line=macd_line,
            signal_line=signal_line,
            histogram=histogram,
        )

    def detect_crossover(self, data: pd.DataFrame) -> MACDSignalType:
        """Detect MACD crossover signals.

        Args:
            data: DataFrame with 'close' column

        Returns:
            Type of crossover signal detected
        """
        result = self.calculate(data)

        if len(result.macd_line) < 2:
            return MACDSignalType.NONE

        current_macd = result.macd_line.iloc[-1]
        current_signal = result.signal_line.iloc[-1]
        prev_macd = result.macd_line.iloc[-2]
        prev_signal = result.signal_line.iloc[-2]

        # Bullish crossover: MACD crosses above signal
        if prev_macd <= prev_signal and current_macd > current_signal:
            return MACDSignalType.BULLISH_CROSSOVER

        # Bearish crossover: MACD crosses below signal
        if prev_macd >= prev_signal and current_macd < current_signal:
            return MACDSignalType.BEARISH_CROSSOVER

        return MACDSignalType.NONE

    def get_histogram_trend(self, data: pd.DataFrame, lookback: int = 5) -> str:
        """Determine histogram trend direction using linear slope.

        Uses slope of recent histogram values instead of requiring strict
        monotonic increase/decrease, which almost never fires in real data.

        Args:
            data: DataFrame with 'close' column
            lookback: Number of periods to check

        Returns:
            "increasing", "decreasing", or "flat"
        """
        result = self.calculate(data)
        histogram = result.histogram

        if len(histogram) < lookback:
            return "flat"

        recent = histogram.iloc[-lookback:].dropna()
        if len(recent) < 3:
            return "flat"

        # Use linear regression slope for robust trend detection
        x = np.arange(len(recent), dtype=float)
        y = recent.values.astype(float)
        hist_std = np.std(y)
        if hist_std == 0:
            return "flat"

        slope = np.polyfit(x, y, 1)[0]

        # Normalize slope by histogram's recent std dev for consistent thresholds
        normalized_slope = slope / hist_std

        if normalized_slope > 0.3:
            return "increasing"
        elif normalized_slope < -0.3:
            return "decreasing"
        return "flat"

    def get_signal_strength(self, data: pd.DataFrame) -> float:
        """Calculate signal strength based on MACD.

        Args:
            data: DataFrame with 'close' column

        Returns:
            Signal strength from -1.0 to 1.0
        """
        result = self.calculate(data)

        macd = result.current_macd
        signal = result.current_signal
        histogram = result.current_histogram

        # ATR-based normalization: normalize histogram by stock's volatility, not price tier
        # This gives consistent signal strength across all price levels
        price = data["close"].iloc[-1]
        if price <= 0:
            normalized_histogram = 0.0
        else:
            # Use rolling std dev of close as volatility proxy (faster than full ATR calc)
            volatility = data["close"].iloc[-20:].std() if len(data) >= 20 else data["close"].std()
            if volatility > 0:
                normalized_histogram = histogram / volatility
            else:
                normalized_histogram = 0.0

        # Base strength on normalized histogram
        strength = max(-1.0, min(1.0, normalized_histogram * 2.0))

        # Boost strength if there's a crossover
        crossover = self.detect_crossover(data)
        if crossover == MACDSignalType.BULLISH_CROSSOVER:
            strength = min(1.0, strength + 0.3)
        elif crossover == MACDSignalType.BEARISH_CROSSOVER:
            strength = max(-1.0, strength - 0.3)

        # Adjust based on histogram trend
        hist_trend = self.get_histogram_trend(data)
        if hist_trend == "increasing" and strength > 0:
            strength = min(1.0, strength + 0.1)
        elif hist_trend == "decreasing" and strength < 0:
            strength = max(-1.0, strength - 0.1)

        return strength

    def analyze(self, data: pd.DataFrame) -> MACDAnalysis:
        """Complete MACD analysis.

        Args:
            data: DataFrame with 'close' column

        Returns:
            MACDAnalysis with all relevant metrics
        """
        result = self.calculate(data)

        # NaN guard: if any core values are NaN, return neutral analysis
        macd_val = result.current_macd
        signal_val = result.current_signal
        histogram_val = result.current_histogram
        if any(math.isnan(v) for v in (macd_val, signal_val, histogram_val)):
            return MACDAnalysis(
                macd_value=0.0,
                signal_value=0.0,
                histogram_value=0.0,
                signal_type=MACDSignalType.NONE,
                trend="neutral",
                signal_strength=0.0,
                histogram_trend="flat",
            )

        crossover = self.detect_crossover(data)
        hist_trend = self.get_histogram_trend(data)
        strength = self.get_signal_strength(data)

        # Determine overall trend
        if result.current_macd > result.current_signal and result.current_macd > 0:
            trend = "bullish"
        elif result.current_macd < result.current_signal and result.current_macd < 0:
            trend = "bearish"
        else:
            trend = "neutral"

        return MACDAnalysis(
            macd_value=result.current_macd,
            signal_value=result.current_signal,
            histogram_value=result.current_histogram,
            signal_type=crossover,
            trend=trend,
            signal_strength=strength,
            histogram_trend=hist_trend,
        )

    def add_to_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add MACD columns to dataframe.

        Args:
            data: DataFrame with 'close' column

        Returns:
            DataFrame with MACD columns added
        """
        df = data.copy()
        result = self.calculate(data)

        df["MACD"] = result.macd_line
        df["MACD_Signal"] = result.signal_line
        df["MACD_Histogram"] = result.histogram

        return df
