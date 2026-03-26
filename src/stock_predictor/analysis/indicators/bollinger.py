"""Bollinger Bands indicator calculations."""

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from stock_predictor.infrastructure.config.constants import INDICATOR_DEFAULTS


class BollingerZone(Enum):
    """Price position relative to Bollinger Bands."""

    ABOVE_UPPER = "above_upper"  # Price above upper band
    UPPER_ZONE = "upper_zone"  # Price between middle and upper
    MIDDLE_ZONE = "middle_zone"  # Price near middle band
    LOWER_ZONE = "lower_zone"  # Price between lower and middle
    BELOW_LOWER = "below_lower"  # Price below lower band


@dataclass
class BollingerResult:
    """Bollinger Bands calculation result."""

    upper: pd.Series
    middle: pd.Series
    lower: pd.Series
    percent_b: pd.Series  # Position within bands (0-1)
    bandwidth: pd.Series  # Band width as percentage

    @property
    def current_upper(self) -> float:
        return self.upper.iloc[-1]

    @property
    def current_middle(self) -> float:
        return self.middle.iloc[-1]

    @property
    def current_lower(self) -> float:
        return self.lower.iloc[-1]

    @property
    def current_percent_b(self) -> float:
        return self.percent_b.iloc[-1]

    @property
    def current_bandwidth(self) -> float:
        return self.bandwidth.iloc[-1]


@dataclass
class BollingerAnalysis:
    """Complete Bollinger Bands analysis."""

    upper_band: float
    middle_band: float
    lower_band: float
    percent_b: float
    bandwidth: float
    zone: BollingerZone
    signal_strength: float  # -1.0 to 1.0
    squeeze: bool  # True if bands are unusually tight
    expansion: bool  # True if bands are expanding


class BollingerBandsCalculator:
    """Calculator for Bollinger Bands indicator."""

    def __init__(
        self,
        period: int | None = None,
        std_dev: float | None = None,
    ) -> None:
        """Initialize Bollinger Bands calculator.

        Args:
            period: Moving average period (default: 20)
            std_dev: Standard deviation multiplier (default: 2.0)
        """
        self.period = period or INDICATOR_DEFAULTS["bb_period"]
        self.std_dev = std_dev or INDICATOR_DEFAULTS["bb_std_dev"]

    def calculate(self, data: pd.DataFrame) -> BollingerResult:
        """Calculate Bollinger Bands.

        Args:
            data: DataFrame with 'close' column

        Returns:
            BollingerResult with all band values
        """
        close = data["close"]

        # Middle band (SMA)
        middle = close.rolling(window=self.period).mean()

        # Standard deviation
        std = close.rolling(window=self.period).std()

        # Upper and lower bands
        upper = middle + (std * self.std_dev)
        lower = middle - (std * self.std_dev)

        # Percent B - where price is within the bands
        # 0 = at lower band, 1 = at upper band
        # Guard against zero band width (all prices identical)
        band_width = upper - lower
        percent_b = (close - lower) / band_width.replace(0, float('nan'))
        percent_b = percent_b.fillna(0.5)  # At middle when bands collapse

        # Bandwidth - width of bands as percentage of middle
        # Guard against middle == 0
        bandwidth = band_width / middle.replace(0, float('nan')) * 100
        bandwidth = bandwidth.fillna(0.0)

        return BollingerResult(
            upper=upper,
            middle=middle,
            lower=lower,
            percent_b=percent_b,
            bandwidth=bandwidth,
        )

    def get_zone(self, data: pd.DataFrame) -> BollingerZone:
        """Determine price zone relative to Bollinger Bands.

        Args:
            data: DataFrame with 'close' column

        Returns:
            Current Bollinger zone
        """
        result = self.calculate(data)
        current_price = data["close"].iloc[-1]

        if current_price > result.current_upper:
            return BollingerZone.ABOVE_UPPER
        elif current_price < result.current_lower:
            return BollingerZone.BELOW_LOWER
        elif current_price > result.current_middle:
            # Between middle and upper
            position = (current_price - result.current_middle) / (
                result.current_upper - result.current_middle
            )
            if position > 0.7:
                return BollingerZone.UPPER_ZONE
            return BollingerZone.MIDDLE_ZONE
        else:
            # Between lower and middle
            position = (result.current_middle - current_price) / (
                result.current_middle - result.current_lower
            )
            if position > 0.7:
                return BollingerZone.LOWER_ZONE
            return BollingerZone.MIDDLE_ZONE

    def detect_squeeze(self, data: pd.DataFrame, lookback: int = 20) -> bool:
        """Detect Bollinger Band squeeze (low volatility).

        Uses both relative (percentile vs recent history) and absolute
        thresholds to avoid detecting "normal" low-vol as squeeze when
        the entire recent period has been low volatility.

        Args:
            data: DataFrame with 'close' column
            lookback: Periods to compare against

        Returns:
            True if squeeze is detected
        """
        result = self.calculate(data)

        if len(result.bandwidth) < lookback:
            return False

        current_bandwidth = result.current_bandwidth
        recent_bw = result.bandwidth.iloc[-lookback:].dropna()

        if len(recent_bw) < lookback // 2:
            return False

        # Relative: current bandwidth below 20th percentile of recent
        percentile_20 = recent_bw.quantile(0.20)
        relative_squeeze = current_bandwidth < percentile_20

        # Absolute: bandwidth must also be below a meaningful threshold
        # Use longer history if available for absolute comparison
        if len(result.bandwidth) >= lookback * 3:
            long_avg = result.bandwidth.iloc[-lookback * 3:].mean()
        else:
            long_avg = result.bandwidth.mean()
        absolute_squeeze = current_bandwidth < long_avg * 0.6

        return relative_squeeze and absolute_squeeze

    def detect_expansion(self, data: pd.DataFrame, lookback: int = 5) -> bool:
        """Detect Bollinger Band expansion (increasing volatility).

        Args:
            data: DataFrame with 'close' column
            lookback: Periods to check for expansion

        Returns:
            True if expansion is detected
        """
        result = self.calculate(data)

        if len(result.bandwidth) < lookback:
            return False

        # Expansion if current bandwidth is above 80th percentile (not strict monotonic)
        avg_bandwidth = result.bandwidth.iloc[-lookback * 4:].mean() if len(result.bandwidth) >= lookback * 4 else result.bandwidth.mean()
        return result.bandwidth.iloc[-1] > avg_bandwidth * 1.5

    def get_signal_strength(self, data: pd.DataFrame) -> float:
        """Calculate signal strength based on Bollinger Bands.

        Args:
            data: DataFrame with 'close' column

        Returns:
            Signal strength from -1.0 (bearish) to 1.0 (bullish)
        """
        result = self.calculate(data)
        zone = self.get_zone(data)
        percent_b = result.current_percent_b

        # Price below lower band = potential oversold (bullish)
        if zone == BollingerZone.BELOW_LOWER:
            return min(1.0, 0.5 + (0 - percent_b) * 0.5)

        # Price above upper band = potential overbought (bearish)
        if zone == BollingerZone.ABOVE_UPPER:
            return max(-1.0, -0.5 - (percent_b - 1) * 0.5)

        # Price in lower zone = slightly bullish
        if zone == BollingerZone.LOWER_ZONE:
            return 0.3 * (0.5 - percent_b)

        # Price in upper zone = slightly bearish
        if zone == BollingerZone.UPPER_ZONE:
            return -0.3 * (percent_b - 0.5)

        # Middle zone = neutral
        return 0.0

    def analyze(self, data: pd.DataFrame) -> BollingerAnalysis:
        """Complete Bollinger Bands analysis.

        Args:
            data: DataFrame with 'close' column

        Returns:
            BollingerAnalysis with all metrics
        """
        result = self.calculate(data)
        zone = self.get_zone(data)
        squeeze = self.detect_squeeze(data)
        expansion = self.detect_expansion(data)
        strength = self.get_signal_strength(data)

        return BollingerAnalysis(
            upper_band=result.current_upper,
            middle_band=result.current_middle,
            lower_band=result.current_lower,
            percent_b=result.current_percent_b,
            bandwidth=result.current_bandwidth,
            zone=zone,
            signal_strength=strength,
            squeeze=squeeze,
            expansion=expansion,
        )

    def add_to_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands columns to dataframe.

        Args:
            data: DataFrame with 'close' column

        Returns:
            DataFrame with Bollinger Bands columns added
        """
        df = data.copy()
        result = self.calculate(data)

        df["BB_Upper"] = result.upper
        df["BB_Middle"] = result.middle
        df["BB_Lower"] = result.lower
        df["BB_PercentB"] = result.percent_b
        df["BB_Bandwidth"] = result.bandwidth

        return df
