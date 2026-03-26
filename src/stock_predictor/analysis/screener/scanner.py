"""Stock scanner and screener implementation."""

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import pandas as pd

from stock_predictor.analysis.indicators.bollinger import BollingerBandsCalculator
from stock_predictor.analysis.indicators.macd import MACDCalculator, MACDSignalType
from stock_predictor.analysis.indicators.moving_averages import MovingAverageCalculator
from stock_predictor.analysis.indicators.rsi import RSICalculator
from stock_predictor.analysis.signals.signal_generator import SignalGenerator
from stock_predictor.core.entities.signal import ScreenerMatch, TradingStyle
from stock_predictor.infrastructure.data_providers.yahoo_provider import YahooDataProvider

logger = logging.getLogger(__name__)


@dataclass
class ScreenerFilters:
    """Filters for stock screening."""

    # Price filters
    min_price: float | None = None
    max_price: float | None = None

    # Volume filters
    min_volume: int | None = None
    min_avg_volume: int | None = None  # 20-day average

    # Technical filters
    above_sma_200: bool | None = None
    above_sma_50: bool | None = None
    rsi_min: float | None = None
    rsi_max: float | None = None
    macd_bullish: bool | None = None
    macd_bearish: bool | None = None

    # Range filters
    near_52_week_high_pct: float | None = None  # Within X% of 52-week high
    near_52_week_low_pct: float | None = None  # Within X% of 52-week low

    # Pattern filters
    golden_cross: bool | None = None
    death_cross: bool | None = None
    bollinger_squeeze: bool | None = None

    # Trend filters
    uptrend: bool | None = None
    downtrend: bool | None = None


@dataclass
class ScreenerPreset:
    """Preset screener configuration."""

    name: str
    description: str
    filters: ScreenerFilters


# Preset screener configurations
SCREENER_PRESETS = {
    "oversold_bounce": ScreenerPreset(
        name="Oversold Bounce",
        description="Stocks with RSI below 30, potential reversal candidates",
        filters=ScreenerFilters(
            rsi_max=30,
            min_volume=100000,
            above_sma_200=True,
        ),
    ),
    "breakout_candidates": ScreenerPreset(
        name="Breakout Candidates",
        description="Stocks near 52-week high with bullish momentum",
        filters=ScreenerFilters(
            near_52_week_high_pct=5,
            macd_bullish=True,
            min_volume=500000,
        ),
    ),
    "trend_followers": ScreenerPreset(
        name="Trend Followers",
        description="Strong uptrend stocks above key moving averages",
        filters=ScreenerFilters(
            above_sma_50=True,
            above_sma_200=True,
            uptrend=True,
            rsi_min=50,
            rsi_max=70,
        ),
    ),
    "value_plays": ScreenerPreset(
        name="Value Plays",
        description="Stocks near 52-week low with improving momentum",
        filters=ScreenerFilters(
            near_52_week_low_pct=10,
            rsi_min=30,
            rsi_max=50,
        ),
    ),
    "momentum_stocks": ScreenerPreset(
        name="Momentum Stocks",
        description="High momentum stocks with strong MACD",
        filters=ScreenerFilters(
            macd_bullish=True,
            rsi_min=60,
            above_sma_50=True,
            min_avg_volume=500000,
        ),
    ),
    "squeeze_plays": ScreenerPreset(
        name="Squeeze Plays",
        description="Stocks with Bollinger Band squeeze, potential breakout",
        filters=ScreenerFilters(
            bollinger_squeeze=True,
            min_volume=200000,
        ),
    ),
}


@dataclass
class ScreenerResults:
    """Results from stock screening."""

    matches: list[ScreenerMatch]
    total_scanned: int
    filters_applied: list[str]
    errors: list[str] = field(default_factory=list)


class StockScanner:
    """Scans stocks based on technical criteria."""

    def __init__(self, data_provider: YahooDataProvider | None = None) -> None:
        """Initialize scanner.

        Args:
            data_provider: Data provider for fetching stock data
        """
        self.data_provider = data_provider or YahooDataProvider()
        self.ma_calc = MovingAverageCalculator()
        self.rsi_calc = RSICalculator()
        self.macd_calc = MACDCalculator()
        self.bb_calc = BollingerBandsCalculator()
        self.signal_generator = SignalGenerator()

    def _check_filters(
        self, data: pd.DataFrame, filters: ScreenerFilters
    ) -> tuple[bool, list[str]]:
        """Check if stock data matches all filters.

        Args:
            data: DataFrame with OHLCV data
            filters: Screening criteria

        Returns:
            Tuple of (matches, list of matching criteria)
        """
        matching_criteria = []
        current_price = float(data["close"].iloc[-1])
        current_volume = int(data["volume"].iloc[-1])

        # Price filters
        if filters.min_price is not None and current_price < filters.min_price:
            return False, []
        if filters.max_price is not None and current_price > filters.max_price:
            return False, []

        # Volume filters
        if filters.min_volume is not None and current_volume < filters.min_volume:
            return False, []

        if filters.min_avg_volume is not None:
            avg_volume = data["volume"].iloc[-20:].mean()
            if avg_volume < filters.min_avg_volume:
                return False, []
            matching_criteria.append(f"Avg Vol: {avg_volume:,.0f}")

        # SMA filters
        if filters.above_sma_200 is not None:
            sma_200 = self.ma_calc.calculate_sma(data, 200).iloc[-1]
            is_above = current_price > sma_200
            if filters.above_sma_200 != is_above:
                return False, []
            if is_above:
                matching_criteria.append("Above 200 SMA")

        if filters.above_sma_50 is not None:
            sma_50 = self.ma_calc.calculate_sma(data, 50).iloc[-1]
            is_above = current_price > sma_50
            if filters.above_sma_50 != is_above:
                return False, []
            if is_above:
                matching_criteria.append("Above 50 SMA")

        # RSI filters
        rsi_value = self.rsi_calc.calculate(data).iloc[-1]
        if filters.rsi_min is not None and rsi_value < filters.rsi_min:
            return False, []
        if filters.rsi_max is not None and rsi_value > filters.rsi_max:
            return False, []
        if filters.rsi_min is not None or filters.rsi_max is not None:
            matching_criteria.append(f"RSI: {rsi_value:.1f}")

        # MACD filters
        macd_analysis = self.macd_calc.analyze(data)
        if filters.macd_bullish is not None:
            is_bullish = (
                macd_analysis.signal_type == MACDSignalType.BULLISH_CROSSOVER
                or (macd_analysis.macd_value > macd_analysis.signal_value and macd_analysis.histogram_trend == "increasing")
            )
            if filters.macd_bullish and not is_bullish:
                return False, []
            if is_bullish:
                matching_criteria.append("MACD Bullish")

        if filters.macd_bearish is not None:
            is_bearish = (
                macd_analysis.signal_type == MACDSignalType.BEARISH_CROSSOVER
                or (macd_analysis.macd_value < macd_analysis.signal_value and macd_analysis.histogram_trend == "decreasing")
            )
            if filters.macd_bearish and not is_bearish:
                return False, []
            if is_bearish:
                matching_criteria.append("MACD Bearish")

        # 52-week filters
        if len(data) >= 252:
            high_52w = float(data["high"].iloc[-252:].max())
            low_52w = float(data["low"].iloc[-252:].min())

            if filters.near_52_week_high_pct is not None:
                pct_from_high = ((high_52w - current_price) / high_52w) * 100
                if pct_from_high > filters.near_52_week_high_pct:
                    return False, []
                matching_criteria.append(f"{pct_from_high:.1f}% from 52W High")

            if filters.near_52_week_low_pct is not None:
                pct_from_low = ((current_price - low_52w) / low_52w) * 100
                if pct_from_low > filters.near_52_week_low_pct:
                    return False, []
                matching_criteria.append(f"{pct_from_low:.1f}% from 52W Low")

        # Crossover filters
        if filters.golden_cross is not None:
            crossover = self.ma_calc.detect_crossover(data, 50, 200, use_ema=False)
            if filters.golden_cross and (crossover is None or not crossover.is_bullish):
                return False, []
            if crossover and crossover.is_bullish:
                matching_criteria.append("Golden Cross")

        if filters.death_cross is not None:
            crossover = self.ma_calc.detect_crossover(data, 50, 200, use_ema=False)
            if filters.death_cross and (crossover is None or not crossover.is_bearish):
                return False, []
            if crossover and crossover.is_bearish:
                matching_criteria.append("Death Cross")

        # Bollinger squeeze
        if filters.bollinger_squeeze is not None:
            is_squeeze = self.bb_calc.detect_squeeze(data)
            if filters.bollinger_squeeze and not is_squeeze:
                return False, []
            if is_squeeze:
                matching_criteria.append("BB Squeeze")

        # Trend filters
        if filters.uptrend is not None or filters.downtrend is not None:
            trend, _ = self.ma_calc.get_trend_signal(data, 21, 50)

            if filters.uptrend and trend != "bullish":
                return False, []
            if filters.downtrend and trend != "bearish":
                return False, []
            if trend == "bullish":
                matching_criteria.append("Uptrend")
            elif trend == "bearish":
                matching_criteria.append("Downtrend")

        return True, matching_criteria

    def _calculate_score(self, data: pd.DataFrame, filters: ScreenerFilters) -> float:
        """Calculate a ranking score for the stock.

        Args:
            data: DataFrame with OHLCV data
            filters: Applied filters

        Returns:
            Score from 0 to 100
        """
        score = 50.0  # Base score

        # RSI contribution
        rsi = self.rsi_calc.calculate(data).iloc[-1]
        if filters.rsi_max and filters.rsi_max <= 30:
            # Oversold screen - lower RSI = higher score
            score += (30 - rsi) * 1.5
        elif filters.rsi_min and filters.rsi_min >= 50:
            # Momentum screen - higher RSI = higher score
            score += (rsi - 50) * 0.5

        # MACD contribution
        macd_analysis = self.macd_calc.analyze(data)
        if filters.macd_bullish:
            if macd_analysis.signal_type == MACDSignalType.BULLISH_CROSSOVER:
                score += 15
            if macd_analysis.histogram_trend == "increasing":
                score += 10

        # Volume contribution
        if len(data) >= 20:
            avg_volume = data["volume"].iloc[-20:].mean()
            current_volume = data["volume"].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            score += min(10, (volume_ratio - 1) * 5)

        return min(100, max(0, score))

    def scan(
        self,
        symbols: list[str],
        filters: ScreenerFilters,
        limit: int = 50,
        style: TradingStyle = TradingStyle.SWING,
    ) -> ScreenerResults:
        """Scan stocks and return matches.

        Args:
            symbols: List of stock symbols to scan
            filters: Screening criteria
            limit: Maximum number of results
            style: Trading style for signal generation

        Returns:
            ScreenerResults with matching stocks
        """
        matches = []
        errors = []
        filters_applied = []

        # Build list of applied filters
        if filters.min_price:
            filters_applied.append(f"Min Price: {filters.min_price}")
        if filters.max_price:
            filters_applied.append(f"Max Price: {filters.max_price}")
        if filters.rsi_min or filters.rsi_max:
            filters_applied.append(f"RSI: {filters.rsi_min or 0}-{filters.rsi_max or 100}")
        if filters.above_sma_200:
            filters_applied.append("Above 200 SMA")
        if filters.above_sma_50:
            filters_applied.append("Above 50 SMA")
        if filters.macd_bullish:
            filters_applied.append("MACD Bullish")
        if filters.bollinger_squeeze:
            filters_applied.append("BB Squeeze")

        for symbol in symbols:
            try:
                # Fetch data
                data = self.data_provider.get_historical(symbol, period="1y", interval="1d")

                if len(data) < 50:  # Need enough data for indicators
                    continue

                # Check filters
                matches_filters, criteria = self._check_filters(data, filters)

                if matches_filters:
                    # Calculate score
                    score = self._calculate_score(data, filters)

                    # Generate signal
                    signal = self.signal_generator.generate(data, symbol, style)

                    # Get quote for current price and change
                    try:
                        quote = self.data_provider.get_quote(symbol)
                        change_percent = float(quote.change_percent)
                    except Exception:
                        change_percent = 0.0

                    matches.append(
                        ScreenerMatch(
                            symbol=symbol,
                            name=symbol,  # Could fetch full name
                            current_price=Decimal(str(round(float(data["close"].iloc[-1]), 2))),
                            change_percent=Decimal(str(round(change_percent, 2))),
                            score=score,
                            matching_criteria=criteria,
                            signal=signal,
                        )
                    )

            except Exception as e:
                errors.append(f"{symbol}: {str(e)}")
                logger.warning(f"Error scanning {symbol}: {e}")
                continue

        # Sort by score
        matches.sort(key=lambda x: x.score, reverse=True)

        return ScreenerResults(
            matches=matches[:limit],
            total_scanned=len(symbols),
            filters_applied=filters_applied,
            errors=errors,
        )

    def scan_with_preset(
        self,
        symbols: list[str],
        preset_name: str,
        limit: int = 50,
        style: TradingStyle = TradingStyle.SWING,
    ) -> ScreenerResults:
        """Scan stocks using a preset configuration.

        Args:
            symbols: List of stock symbols to scan
            preset_name: Name of the preset to use
            limit: Maximum number of results
            style: Trading style for signal generation

        Returns:
            ScreenerResults with matching stocks
        """
        if preset_name not in SCREENER_PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(SCREENER_PRESETS.keys())}")

        preset = SCREENER_PRESETS[preset_name]
        return self.scan(symbols, preset.filters, limit, style)

    @staticmethod
    def get_available_presets() -> dict[str, str]:
        """Get available screener presets.

        Returns:
            Dictionary mapping preset name to description
        """
        return {name: preset.description for name, preset in SCREENER_PRESETS.items()}
