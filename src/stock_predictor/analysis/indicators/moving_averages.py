"""Moving average indicator calculations."""

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from stock_predictor.infrastructure.config.constants import INDICATOR_DEFAULTS


class CrossoverType(Enum):
    """Type of moving average crossover."""

    GOLDEN_CROSS = "golden_cross"  # Short MA crosses above long MA (bullish)
    DEATH_CROSS = "death_cross"  # Short MA crosses below long MA (bearish)
    NONE = "none"


@dataclass
class CrossoverSignal:
    """Moving average crossover signal."""

    crossover_type: CrossoverType
    short_period: int
    long_period: int
    short_value: float
    long_value: float
    price: float

    @property
    def is_bullish(self) -> bool:
        return self.crossover_type == CrossoverType.GOLDEN_CROSS

    @property
    def is_bearish(self) -> bool:
        return self.crossover_type == CrossoverType.DEATH_CROSS


class MovingAverageCalculator:
    """Calculator for various moving averages."""

    def __init__(
        self,
        sma_periods: list[int] | None = None,
        ema_periods: list[int] | None = None,
    ) -> None:
        """Initialize calculator with periods.

        Args:
            sma_periods: List of periods for SMA calculation
            ema_periods: List of periods for EMA calculation
        """
        self.sma_periods = sma_periods or INDICATOR_DEFAULTS["sma_periods"]
        self.ema_periods = ema_periods or INDICATOR_DEFAULTS["ema_periods"]

    def calculate_sma(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Simple Moving Average.

        Args:
            data: DataFrame with 'close' column
            period: Number of periods for SMA

        Returns:
            Series with SMA values
        """
        return data["close"].rolling(window=period).mean()

    def calculate_ema(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Exponential Moving Average.

        Args:
            data: DataFrame with 'close' column
            period: Number of periods for EMA

        Returns:
            Series with EMA values
        """
        return data["close"].ewm(span=period, adjust=False).mean()

    def calculate_all_sma(self, data: pd.DataFrame) -> dict[str, pd.Series]:
        """Calculate SMA for all configured periods.

        Args:
            data: DataFrame with 'close' column

        Returns:
            Dictionary mapping period names to SMA series
        """
        return {
            f"SMA_{period}": self.calculate_sma(data, period) for period in self.sma_periods
        }

    def calculate_all_ema(self, data: pd.DataFrame) -> dict[str, pd.Series]:
        """Calculate EMA for all configured periods.

        Args:
            data: DataFrame with 'close' column

        Returns:
            Dictionary mapping period names to EMA series
        """
        return {
            f"EMA_{period}": self.calculate_ema(data, period) for period in self.ema_periods
        }

    def calculate_all(self, data: pd.DataFrame) -> dict[str, pd.Series]:
        """Calculate all configured moving averages.

        Args:
            data: DataFrame with 'close' column

        Returns:
            Dictionary with all SMA and EMA series
        """
        result = {}
        result.update(self.calculate_all_sma(data))
        result.update(self.calculate_all_ema(data))
        return result

    def add_to_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add all moving averages as columns to dataframe.

        Args:
            data: DataFrame with 'close' column

        Returns:
            DataFrame with added MA columns
        """
        df = data.copy()
        mas = self.calculate_all(data)
        for name, series in mas.items():
            df[name] = series
        return df

    def detect_crossover(
        self,
        data: pd.DataFrame,
        short_period: int = 9,
        long_period: int = 21,
        use_ema: bool = True,
    ) -> CrossoverSignal | None:
        """Detect most recent moving average crossover.

        Args:
            data: DataFrame with 'close' column
            short_period: Period for short-term MA
            long_period: Period for long-term MA
            use_ema: Use EMA instead of SMA

        Returns:
            CrossoverSignal if crossover detected, None otherwise
        """
        if use_ema:
            short_ma = self.calculate_ema(data, short_period)
            long_ma = self.calculate_ema(data, long_period)
        else:
            short_ma = self.calculate_sma(data, short_period)
            long_ma = self.calculate_sma(data, long_period)

        # Check last two values for crossover
        if len(short_ma) < 2 or len(long_ma) < 2:
            return None

        current_short = short_ma.iloc[-1]
        current_long = long_ma.iloc[-1]
        prev_short = short_ma.iloc[-2]
        prev_long = long_ma.iloc[-2]
        current_price = data["close"].iloc[-1]

        # Golden cross: short crosses above long
        if prev_short <= prev_long and current_short > current_long:
            return CrossoverSignal(
                crossover_type=CrossoverType.GOLDEN_CROSS,
                short_period=short_period,
                long_period=long_period,
                short_value=current_short,
                long_value=current_long,
                price=current_price,
            )

        # Death cross: short crosses below long
        if prev_short >= prev_long and current_short < current_long:
            return CrossoverSignal(
                crossover_type=CrossoverType.DEATH_CROSS,
                short_period=short_period,
                long_period=long_period,
                short_value=current_short,
                long_value=current_long,
                price=current_price,
            )

        return None

    def get_trend_signal(
        self, data: pd.DataFrame, short_period: int = 21, long_period: int = 50
    ) -> tuple[str, float]:
        """Get trend signal based on MA relationship.

        Args:
            data: DataFrame with 'close' column
            short_period: Short-term MA period
            long_period: Long-term MA period

        Returns:
            Tuple of (trend_direction, signal_strength)
            - trend_direction: "bullish", "bearish", or "neutral"
            - signal_strength: -1.0 to 1.0
        """
        short_ma = self.calculate_ema(data, short_period).iloc[-1]
        long_ma = self.calculate_ema(data, long_period).iloc[-1]
        current_price = data["close"].iloc[-1]

        # Calculate position relative to MAs
        price_vs_short = (current_price - short_ma) / short_ma
        price_vs_long = (current_price - long_ma) / long_ma
        short_vs_long = (short_ma - long_ma) / long_ma

        # Determine trend
        if current_price > short_ma > long_ma:
            # Strong uptrend
            strength = min(1.0, (price_vs_long + short_vs_long) * 10)
            return "bullish", strength
        elif current_price < short_ma < long_ma:
            # Strong downtrend
            strength = max(-1.0, (price_vs_long + short_vs_long) * 10)
            return "bearish", strength
        else:
            # Mixed or consolidating
            strength = short_vs_long * 5
            if abs(strength) < 0.1:
                return "neutral", 0.0
            return "bullish" if strength > 0 else "bearish", strength

    def is_price_above_ma(self, data: pd.DataFrame, period: int = 200) -> bool:
        """Check if current price is above a specific MA.

        Args:
            data: DataFrame with 'close' column
            period: MA period to check

        Returns:
            True if price is above MA
        """
        ma = self.calculate_sma(data, period).iloc[-1]
        return data["close"].iloc[-1] > ma

    def get_ma_values(self, data: pd.DataFrame) -> dict[str, float]:
        """Get current values for all moving averages.

        Args:
            data: DataFrame with 'close' column

        Returns:
            Dictionary with current MA values
        """
        all_mas = self.calculate_all(data)
        return {name: series.iloc[-1] for name, series in all_mas.items()}
