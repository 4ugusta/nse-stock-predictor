"""High confidence stock picker with strict multi-filter validation.

This module provides a more rigorous stock selection process that requires
multiple confirmations before recommending a stock.

Key improvements over basic screener:
1. ADX trend strength filter - avoids weak/choppy trends
2. Volume confirmation - requires volume to support price moves
3. Price vs 200 SMA - confirms trend direction
4. MACD momentum confirmation - requires momentum alignment
5. Risk/reward validation - only picks with good R:R ratios
6. BEAR MARKET SUPPORT - can find short opportunities in downtrends
7. ATR-based stop/target levels - volatility-adjusted
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
import logging

import pandas as pd

from stock_predictor.analysis.indicators.adx import ADXCalculator, MarketRegime
from stock_predictor.analysis.indicators.atr import ATRCalculator
from stock_predictor.analysis.indicators.volume import VolumeAnalyzer, VolumeSignal
from stock_predictor.analysis.indicators.moving_averages import MovingAverageCalculator
from stock_predictor.analysis.indicators.rsi import RSICalculator
from stock_predictor.analysis.indicators.macd import MACDCalculator, MACDSignalType
from stock_predictor.analysis.indicators.bollinger import BollingerBandsCalculator
from stock_predictor.analysis.indicators.support_resistance import SupportResistanceCalculator
from stock_predictor.core.entities.signal import SignalType
from stock_predictor.infrastructure.data_providers.yahoo_provider import YahooDataProvider

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence level for stock picks."""

    VERY_HIGH = "very_high"  # 5/5 filters pass
    HIGH = "high"  # 4/5 filters pass
    MEDIUM = "medium"  # 3/5 filters pass
    LOW = "low"  # 2/5 filters pass
    AVOID = "avoid"  # <2 filters pass


@dataclass
class FilterResult:
    """Result of a single filter check."""

    name: str
    passed: bool
    reason: str
    weight: float = 1.0


@dataclass
class HighConfidencePick:
    """A high confidence stock pick with full analysis."""

    symbol: str
    current_price: Decimal
    confidence_level: ConfidenceLevel
    confidence_score: float  # 0-100
    filters_passed: list[FilterResult]
    filters_failed: list[FilterResult]

    # Entry/Exit levels
    entry_price: Decimal
    stop_loss: Decimal
    target_1: Decimal  # Conservative target
    target_2: Decimal  # Aggressive target

    # Risk metrics
    risk_percent: float  # Distance to stop loss
    reward_percent: float  # Distance to target
    risk_reward_ratio: float

    # Key metrics
    adx_value: float
    rsi_value: float
    volume_ratio: float
    regime: MarketRegime

    # Summary
    recommendation: str
    reasons: list[str]


@dataclass
class PickerConfig:
    """Configuration for high confidence stock picker."""

    min_adx: float = 25.0
    min_rsi_long: float = 40.0
    max_rsi_long: float = 70.0
    min_rsi_short: float = 40.0  # Shorts: don't short into oversold (<40)
    max_rsi_short: float = 70.0  # Shorts: RSI should be declining from upper range
    min_volume_ratio: float = 1.0
    min_risk_reward: float = 1.5
    min_confidence: float = 0.5
    max_picks: int = 10


class HighConfidencePicker:
    """High confidence stock picker with strict validation."""

    def __init__(
        self,
        data_provider: YahooDataProvider | None = None,
        config: PickerConfig | None = None,
    ) -> None:
        """Initialize the high confidence picker.

        Args:
            data_provider: Data provider for fetching stock data
            config: Optional picker configuration. Uses defaults if not provided.
        """
        self.data_provider = data_provider or YahooDataProvider()
        self.config = config or PickerConfig()

        # Expose thresholds from config for backward compatibility
        self.MIN_ADX = self.config.min_adx
        self.MIN_RSI = self.config.min_rsi_long
        self.MAX_RSI = self.config.max_rsi_long
        self.MIN_VOLUME_RATIO = self.config.min_volume_ratio
        self.MIN_RISK_REWARD = self.config.min_risk_reward
        self.adx_calc = ADXCalculator()
        self.atr_calc = ATRCalculator()
        self.volume_calc = VolumeAnalyzer()
        self.ma_calc = MovingAverageCalculator()
        self.rsi_calc = RSICalculator()
        self.macd_calc = MACDCalculator()
        self.bb_calc = BollingerBandsCalculator()
        self.sr_calc = SupportResistanceCalculator()

    def _check_trend_filter(self, data: pd.DataFrame) -> FilterResult:
        """Check ADX trend strength filter.

        Requires:
        - ADX >= 20 (trending market)
        - +DI > -DI (bullish direction)
        - Not in downtrend regime
        """
        adx_analysis = self.adx_calc.analyze(data)

        if adx_analysis.regime in (MarketRegime.DOWNTREND, MarketRegime.STRONG_DOWNTREND):
            return FilterResult(
                name="Trend Direction",
                passed=False,
                reason=f"Downtrend detected (ADX: {adx_analysis.adx_value:.1f}, -DI > +DI)",
                weight=1.5,  # Higher weight - critical filter
            )

        if adx_analysis.adx_value < self.MIN_ADX:
            return FilterResult(
                name="Trend Strength",
                passed=False,
                reason=f"Weak trend (ADX: {adx_analysis.adx_value:.1f} < {self.MIN_ADX})",
                weight=1.2,
            )

        if adx_analysis.regime == MarketRegime.SIDEWAYS:
            return FilterResult(
                name="Trend Direction",
                passed=False,
                reason=f"Sideways/choppy market (ADX: {adx_analysis.adx_value:.1f})",
                weight=1.2,
            )

        if adx_analysis.regime == MarketRegime.VOLATILE:
            return FilterResult(
                name="Trend Direction",
                passed=False,
                reason=f"Volatile/choppy market (ADX: {adx_analysis.adx_value:.1f}, DI lines close)",
                weight=1.2,
            )

        return FilterResult(
            name="Trend",
            passed=True,
            reason=f"Strong uptrend (ADX: {adx_analysis.adx_value:.1f}, +DI: {adx_analysis.plus_di:.1f})",
            weight=1.5,
        )

    def _check_volume_filter(self, data: pd.DataFrame) -> FilterResult:
        """Check volume confirmation filter.

        Requires:
        - OBV not bearish
        - No distribution pattern
        - Reasonable volume (not dead stock)
        """
        vol_analysis = self.volume_calc.analyze(data)

        if vol_analysis.signal == VolumeSignal.DISTRIBUTION:
            return FilterResult(
                name="Volume",
                passed=False,
                reason="Distribution pattern - smart money selling",
                weight=1.3,
            )

        if vol_analysis.signal == VolumeSignal.STRONG_BEARISH:
            return FilterResult(
                name="Volume",
                passed=False,
                reason="Heavy selling pressure",
                weight=1.3,
            )

        if vol_analysis.on_balance_volume_trend == "bearish":
            return FilterResult(
                name="Volume",
                passed=False,
                reason="OBV bearish - money leaving the stock",
                weight=1.2,
            )

        if vol_analysis.volume_ratio < self.MIN_VOLUME_RATIO:
            return FilterResult(
                name="Volume",
                passed=False,
                reason=f"Low volume ({vol_analysis.volume_ratio:.1f}x avg) - no conviction",
                weight=1.0,
            )

        return FilterResult(
            name="Volume",
            passed=True,
            reason=f"Volume confirmed ({vol_analysis.volume_ratio:.1f}x avg, OBV: {vol_analysis.on_balance_volume_trend})",
            weight=1.3,
        )

    def _check_ma_filter(self, data: pd.DataFrame) -> FilterResult:
        """Check moving average filter.

        Requires:
        - Price above 200 SMA (long-term uptrend)
        - Price above 50 SMA (medium-term uptrend)
        """
        current_price = float(data["close"].iloc[-1])

        try:
            sma_200 = self.ma_calc.calculate_sma(data, 200).iloc[-1]
            sma_50 = self.ma_calc.calculate_sma(data, 50).iloc[-1]
        except Exception:
            return FilterResult(
                name="Moving Averages",
                passed=False,
                reason="Insufficient data for MA calculation",
                weight=1.0,
            )

        above_200 = current_price > sma_200
        above_50 = current_price > sma_50

        if not above_200:
            return FilterResult(
                name="Moving Averages",
                passed=False,
                reason=f"Below 200 SMA (Price: {current_price:.2f}, 200 SMA: {sma_200:.2f})",
                weight=1.4,
            )

        if not above_50:
            return FilterResult(
                name="Moving Averages",
                passed=False,
                reason=f"Below 50 SMA - short-term weakness",
                weight=1.1,
            )

        return FilterResult(
            name="Moving Averages",
            passed=True,
            reason=f"Above both 50 & 200 SMA - confirmed uptrend",
            weight=1.4,
        )

    def _check_momentum_filter(self, data: pd.DataFrame) -> FilterResult:
        """Check MACD momentum filter.

        Requires:
        - MACD bullish (line above signal) or bullish crossover
        - Histogram increasing (momentum building)
        """
        macd = self.macd_calc.analyze(data)

        is_bullish = (
            macd.signal_type == MACDSignalType.BULLISH_CROSSOVER
            or macd.macd_value > macd.signal_value
        )

        if not is_bullish:
            return FilterResult(
                name="Momentum",
                passed=False,
                reason=f"MACD bearish - momentum fading",
                weight=1.2,
            )

        if macd.histogram_trend == "decreasing":
            return FilterResult(
                name="Momentum",
                passed=False,
                reason="MACD histogram decreasing - losing momentum",
                weight=1.0,
            )

        if macd.signal_type == MACDSignalType.BULLISH_CROSSOVER:
            return FilterResult(
                name="Momentum",
                passed=True,
                reason="MACD bullish crossover - fresh momentum",
                weight=1.3,
            )

        return FilterResult(
            name="Momentum",
            passed=True,
            reason=f"MACD bullish with {macd.histogram_trend} histogram",
            weight=1.2,
        )

    def _check_rsi_filter(self, data: pd.DataFrame) -> FilterResult:
        """Check RSI filter.

        Requires:
        - RSI between 40-70 (not oversold in downtrend, not overbought)
        - Ideal zone: 50-65
        """
        rsi_result = self.rsi_calc.analyze(data)
        rsi_value = rsi_result.value

        if rsi_value > self.MAX_RSI:
            return FilterResult(
                name="RSI",
                passed=False,
                reason=f"Overbought (RSI: {rsi_value:.1f}) - pullback likely",
                weight=1.1,
            )

        if rsi_value < self.MIN_RSI:
            return FilterResult(
                name="RSI",
                passed=False,
                reason=f"RSI too low ({rsi_value:.1f}) - may be in downtrend",
                weight=1.0,
            )

        if 50 <= rsi_value <= 65:
            return FilterResult(
                name="RSI",
                passed=True,
                reason=f"RSI in ideal zone ({rsi_value:.1f}) - room to run",
                weight=1.1,
            )

        return FilterResult(
            name="RSI",
            passed=True,
            reason=f"RSI acceptable ({rsi_value:.1f})",
            weight=1.0,
        )

    def _check_whipsaw_filter(self, data: pd.DataFrame) -> FilterResult:
        """Check for choppy/whipsaw price action.

        Rejects stocks in narrow consolidation ranges where indicators
        align randomly but breakout conviction is low.
        """
        try:
            recent = data["close"].iloc[-10:]
            atr = self.atr_calc.analyze(data).current_atr
            recent_range = float(recent.max() - recent.min())

            if atr > 0 and recent_range / atr < 1.5:
                return FilterResult(
                    name="Price Action",
                    passed=False,
                    reason=f"Choppy consolidation (range: {recent_range:.1f}, ATR: {atr:.1f})",
                    weight=1.0,
                )

            return FilterResult(
                name="Price Action",
                passed=True,
                reason="Clean trending price action",
                weight=1.0,
            )
        except Exception:
            return FilterResult(
                name="Price Action",
                passed=False,
                reason="Unable to assess price action (conservative fail)",
                weight=0.5,
            )

    def _calculate_levels(
        self, data: pd.DataFrame, current_price: float, is_long: bool = True
    ) -> tuple[Decimal, Decimal, Decimal, Decimal]:
        """Calculate ATR-based entry, stop loss, and target levels.

        Uses ATR for volatility-adjusted stops instead of fixed percentages.

        Args:
            data: DataFrame with OHLCV data
            current_price: Current stock price
            is_long: True for long positions, False for shorts

        Returns:
            Tuple of (entry, stop_loss, target_1, target_2)
        """
        sr = self.sr_calc.analyze(data)
        atr_result = self.atr_calc.analyze(data)
        atr = atr_result.current_atr

        entry = current_price

        if is_long:
            # ATR-based stop: 1.5x ATR below entry
            stop_loss = entry - (atr * 1.5)

            # If S/R is tighter, use it (but only if it still gives good R:R)
            if sr.nearest_support and sr.distance_to_support_pct and sr.distance_to_support_pct < 5:
                sr_stop = sr.nearest_support * 0.99
                # Use S/R stop only if it's tighter than ATR stop AND still below entry
                if sr_stop > stop_loss and sr_stop < entry:
                    stop_loss = sr_stop

            risk = entry - stop_loss
            # Target 1: 2x risk or nearest resistance
            if sr.nearest_resistance and sr.distance_to_resistance_pct and sr.distance_to_resistance_pct < 15:
                target_1 = sr.nearest_resistance
            else:
                target_1 = entry + (risk * 2)

            # Target 2: 3x ATR above entry
            target_2 = entry + (atr * 3)
        else:
            # Short position: ATR-based stop above entry
            stop_loss = entry + (atr * 1.5)

            if sr.nearest_resistance and sr.distance_to_resistance_pct and sr.distance_to_resistance_pct < 5:
                sr_stop = sr.nearest_resistance * 1.01
                # Use S/R stop only if tighter than ATR stop AND still above entry
                if sr_stop < stop_loss and sr_stop > entry:
                    stop_loss = sr_stop

            risk = stop_loss - entry
            if sr.nearest_support and sr.distance_to_support_pct and sr.distance_to_support_pct < 15:
                target_1 = sr.nearest_support
            else:
                target_1 = entry - (risk * 2)

            target_2 = entry - (atr * 3)

        # Sanity check: cap targets at ±50% from entry (unrealistic moves = calculation error)
        max_move = entry * 0.50
        if is_long:
            target_1 = min(target_1, entry + max_move)
            target_2 = min(target_2, entry + max_move)
            stop_loss = max(stop_loss, entry - max_move)
        else:
            target_1 = max(target_1, entry - max_move, 0.01)
            target_2 = max(target_2, entry - max_move, 0.01)
            stop_loss = min(stop_loss, entry + max_move)

        return (
            Decimal(str(round(entry, 2))),
            Decimal(str(round(stop_loss, 2))),
            Decimal(str(round(target_1, 2))),
            Decimal(str(round(target_2, 2))),
        )

    def analyze_stock(self, symbol: str, signal: object | None = None) -> HighConfidencePick | None:
        """Analyze a single stock with all filters.

        Args:
            symbol: Stock symbol
            signal: Optional TradingSignal from the signal generator.
                    If provided and signal_type is NO_EDGE, the stock is skipped.

        Returns:
            HighConfidencePick if analysis successful, None otherwise
        """
        try:
            # Skip NO_EDGE signals - system says there's no tradeable edge
            if signal is not None and hasattr(signal, 'signal_type'):
                if signal.signal_type == SignalType.NO_EDGE:
                    logger.info(f"{symbol}: Skipped - NO_EDGE signal (insufficient signal clarity)")
                    return None

            data = self.data_provider.get_historical(symbol, period="1y", interval="1d")

            if len(data) < 200:
                logger.warning(f"{symbol}: Insufficient data ({len(data)} days)")
                return None

            current_price = float(data["close"].iloc[-1])

            # Run all filters (6 filters including whipsaw protection)
            trend_filter = self._check_trend_filter(data)
            filters = [
                trend_filter,
                self._check_volume_filter(data),
                self._check_ma_filter(data),
                self._check_momentum_filter(data),
                self._check_rsi_filter(data),
                self._check_whipsaw_filter(data),
            ]

            # Early exit: if the critical Trend filter fails, force AVOID.
            # No point checking other filters if there's no trend to ride.
            if not trend_filter.passed:
                return HighConfidencePick(
                    symbol=symbol,
                    current_price=Decimal(str(round(current_price, 2))),
                    confidence_level=ConfidenceLevel.AVOID,
                    confidence_score=0.0,
                    filters_passed=[f for f in filters if f.passed],
                    filters_failed=[f for f in filters if not f.passed],
                    entry_price=Decimal(str(round(current_price, 2))),
                    stop_loss=Decimal(str(round(current_price * 0.95, 2))),
                    target_1=Decimal(str(round(current_price * 1.05, 2))),
                    target_2=Decimal(str(round(current_price * 1.10, 2))),
                    risk_percent=5.0,
                    reward_percent=5.0,
                    risk_reward_ratio=1.0,
                    adx_value=0.0,
                    rsi_value=50.0,
                    volume_ratio=0.0,
                    regime=MarketRegime.SIDEWAYS,
                    recommendation=f"AVOID - {trend_filter.reason}",
                    reasons=[trend_filter.reason],
                )

            passed = [f for f in filters if f.passed]
            failed = [f for f in filters if not f.passed]

            # Calculate weighted score
            total_weight = sum(f.weight for f in filters)
            passed_weight = sum(f.weight for f in passed)
            confidence_score = (passed_weight / total_weight) * 100

            # Determine confidence level (adjusted for 6 filters)
            num_passed = len(passed)
            if num_passed >= 6:
                confidence_level = ConfidenceLevel.VERY_HIGH
            elif num_passed >= 5:
                confidence_level = ConfidenceLevel.HIGH
            elif num_passed >= 4:
                confidence_level = ConfidenceLevel.MEDIUM
            elif num_passed >= 3:
                confidence_level = ConfidenceLevel.LOW
            else:
                confidence_level = ConfidenceLevel.AVOID

            # Calculate entry/exit levels
            entry, stop_loss, target_1, target_2 = self._calculate_levels(data, current_price)

            # Calculate risk metrics
            risk_percent = float((entry - stop_loss) / entry * 100)
            reward_percent = float((target_1 - entry) / entry * 100)
            risk_reward = reward_percent / risk_percent if risk_percent > 0 and reward_percent > 0 else 0

            # Enforce minimum Risk:Reward ratio
            if risk_reward < self.MIN_RISK_REWARD:
                confidence_level = ConfidenceLevel.AVOID

            # Reuse indicator values from filters (avoid redundant calculations)
            # ADX was computed in _check_trend_filter, RSI in _check_rsi_filter,
            # Volume in _check_volume_filter — re-fetch once for summary metrics
            adx = self.adx_calc.analyze(data)
            rsi = self.rsi_calc.analyze(data)
            vol = self.volume_calc.analyze(data)

            # Generate recommendation
            if confidence_level == ConfidenceLevel.VERY_HIGH:
                recommendation = "STRONG BUY"
            elif confidence_level == ConfidenceLevel.HIGH:
                recommendation = "BUY"
            elif confidence_level == ConfidenceLevel.MEDIUM:
                recommendation = "WEAK BUY - Proceed with caution"
            elif risk_reward < self.MIN_RISK_REWARD:
                recommendation = (
                    f"AVOID - Risk:Reward too low ({risk_reward:.1f}:1, "
                    f"need {self.MIN_RISK_REWARD}:1)"
                )
            else:
                recommendation = "AVOID - Too many red flags"

            # Collect reasons
            reasons = [f.reason for f in passed]

            return HighConfidencePick(
                symbol=symbol,
                current_price=Decimal(str(round(current_price, 2))),
                confidence_level=confidence_level,
                confidence_score=confidence_score,
                filters_passed=passed,
                filters_failed=failed,
                entry_price=entry,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                risk_percent=risk_percent,
                reward_percent=reward_percent,
                risk_reward_ratio=risk_reward,
                adx_value=adx.adx_value,
                rsi_value=rsi.value,
                volume_ratio=vol.volume_ratio,
                regime=adx.regime,
                recommendation=recommendation,
                reasons=reasons,
            )

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None

    def scan_stocks(
        self,
        symbols: list[str],
        min_confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
    ) -> list[HighConfidencePick]:
        """Scan multiple stocks and return high confidence picks.

        Args:
            symbols: List of stock symbols
            min_confidence: Minimum confidence level to include

        Returns:
            List of HighConfidencePick sorted by confidence score
        """
        picks = []
        confidence_order = [
            ConfidenceLevel.VERY_HIGH,
            ConfidenceLevel.HIGH,
            ConfidenceLevel.MEDIUM,
            ConfidenceLevel.LOW,
            ConfidenceLevel.AVOID,  # Last position - worst
        ]
        min_index = confidence_order.index(min_confidence)

        for symbol in symbols:
            pick = self.analyze_stock(symbol)
            if pick and pick.confidence_level != ConfidenceLevel.AVOID:
                pick_index = confidence_order.index(pick.confidence_level)
                if pick_index <= min_index:
                    picks.append(pick)

        # Sort by confidence score
        picks.sort(key=lambda p: p.confidence_score, reverse=True)

        return picks

    def get_best_pick(self, symbols: list[str]) -> HighConfidencePick | None:
        """Get the single best stock pick from a list.

        Args:
            symbols: List of stock symbols

        Returns:
            Best HighConfidencePick or None if no good picks
        """
        picks = self.scan_stocks(symbols, min_confidence=ConfidenceLevel.HIGH)
        return picks[0] if picks else None

    def analyze_short(self, symbol: str, signal: object | None = None) -> HighConfidencePick | None:
        """Analyze a stock for SHORT opportunities (bear market picks).

        Inverts the filter logic: looks for confirmed downtrends with
        bearish momentum. Useful in bear markets when the long-only
        scanner produces zero picks.

        Args:
            symbol: Stock symbol
            signal: Optional TradingSignal from the signal generator.
                    If provided and signal_type is NO_EDGE, the stock is skipped.

        Returns:
            HighConfidencePick for short side, or None
        """
        try:
            # Skip NO_EDGE signals - system says there's no tradeable edge
            if signal is not None and hasattr(signal, 'signal_type'):
                if signal.signal_type == SignalType.NO_EDGE:
                    logger.info(f"{symbol}: Skipped short - NO_EDGE signal (insufficient signal clarity)")
                    return None

            data = self.data_provider.get_historical(symbol, period="1y", interval="1d")

            if len(data) < 200:
                return None

            current_price = float(data["close"].iloc[-1])

            # Short-side filters (inverted from long)
            filters = []

            # 1. Trend: must be downtrend
            adx_analysis = self.adx_calc.analyze(data)
            if adx_analysis.regime in (MarketRegime.DOWNTREND, MarketRegime.STRONG_DOWNTREND):
                filters.append(FilterResult(
                    name="Downtrend", passed=True,
                    reason=f"Confirmed downtrend (ADX: {adx_analysis.adx_value:.1f})",
                    weight=1.5,
                ))
            else:
                filters.append(FilterResult(
                    name="Downtrend", passed=False,
                    reason=f"Not in downtrend (regime: {adx_analysis.regime.value})",
                    weight=1.5,
                ))

            # 2. Volume: distribution or bearish
            vol_analysis = self.volume_calc.analyze(data)
            if vol_analysis.signal in (VolumeSignal.DISTRIBUTION, VolumeSignal.STRONG_BEARISH, VolumeSignal.BEARISH):
                filters.append(FilterResult(
                    name="Volume", passed=True,
                    reason=f"Bearish volume ({vol_analysis.signal.value})",
                    weight=1.3,
                ))
            else:
                filters.append(FilterResult(
                    name="Volume", passed=False,
                    reason=f"Volume not bearish ({vol_analysis.signal.value})",
                    weight=1.0,
                ))

            # 3. MA: price below 200 SMA
            try:
                sma_200 = self.ma_calc.calculate_sma(data, 200).iloc[-1]
                sma_50 = self.ma_calc.calculate_sma(data, 50).iloc[-1]
                below_200 = current_price < sma_200
                below_50 = current_price < sma_50

                if below_200 and below_50:
                    filters.append(FilterResult(
                        name="Moving Averages", passed=True,
                        reason="Below 50 & 200 SMA - confirmed downtrend",
                        weight=1.4,
                    ))
                elif below_200:
                    filters.append(FilterResult(
                        name="Moving Averages", passed=True,
                        reason="Below 200 SMA",
                        weight=1.2,
                    ))
                else:
                    filters.append(FilterResult(
                        name="Moving Averages", passed=False,
                        reason="Above 200 SMA - not a short candidate",
                        weight=1.4,
                    ))
            except Exception:
                filters.append(FilterResult(
                    name="Moving Averages", passed=False,
                    reason="Insufficient data for MA", weight=1.0,
                ))

            # 4. Momentum: MACD bearish
            macd = self.macd_calc.analyze(data)
            is_bearish = (
                macd.signal_type == MACDSignalType.BEARISH_CROSSOVER
                or macd.macd_value < macd.signal_value
            )
            if is_bearish:
                filters.append(FilterResult(
                    name="Momentum", passed=True,
                    reason="MACD bearish - momentum fading",
                    weight=1.2,
                ))
            else:
                filters.append(FilterResult(
                    name="Momentum", passed=False,
                    reason="MACD not bearish",
                    weight=1.2,
                ))

            # 5. RSI: not oversold (don't short into oversold) + must be declining
            rsi = self.rsi_calc.analyze(data)
            rsi_value = rsi.value
            rsi_in_range = self.config.min_rsi_short < rsi_value < self.config.max_rsi_short

            # Check if RSI is declining (comparing current to recent average)
            try:
                rsi_series = self.rsi_calc.calculate(data)
                recent_rsi = rsi_series.iloc[-3:]
                rsi_declining = float(recent_rsi.iloc[-1]) < float(recent_rsi.iloc[0])
            except Exception:
                rsi_declining = rsi_value > 50  # Fallback: above midline means room to fall

            short_rsi_ok = rsi_in_range and rsi_declining

            if short_rsi_ok:
                filters.append(FilterResult(
                    name="RSI", passed=True,
                    reason=f"RSI {rsi_value:.1f} declining - room to fall",
                    weight=1.0,
                ))
            elif rsi_value <= self.config.min_rsi_short:
                filters.append(FilterResult(
                    name="RSI", passed=False,
                    reason=f"RSI oversold ({rsi_value:.1f}) - bounce likely",
                    weight=1.1,
                ))
            elif not rsi_declining and rsi_in_range:
                filters.append(FilterResult(
                    name="RSI", passed=False,
                    reason=f"RSI {rsi_value:.1f} in range but not declining",
                    weight=0.9,
                ))
            else:
                filters.append(FilterResult(
                    name="RSI", passed=False,
                    reason=f"RSI too high ({rsi_value:.1f}) for short",
                    weight=0.8,
                ))

            # 6. Whipsaw protection
            filters.append(self._check_whipsaw_filter(data))

            passed = [f for f in filters if f.passed]
            failed = [f for f in filters if not f.passed]

            total_weight = sum(f.weight for f in filters)
            passed_weight = sum(f.weight for f in passed)
            confidence_score = (passed_weight / total_weight) * 100

            # Adjusted for 6 filters
            num_passed = len(passed)
            if num_passed >= 6:
                confidence_level = ConfidenceLevel.VERY_HIGH
            elif num_passed >= 5:
                confidence_level = ConfidenceLevel.HIGH
            elif num_passed >= 4:
                confidence_level = ConfidenceLevel.MEDIUM
            elif num_passed >= 3:
                confidence_level = ConfidenceLevel.LOW
            else:
                confidence_level = ConfidenceLevel.AVOID

            entry, stop_loss, target_1, target_2 = self._calculate_levels(
                data, current_price, is_long=False
            )

            risk_percent = float((stop_loss - entry) / entry * 100)
            reward_percent = float((entry - target_1) / entry * 100)
            risk_reward = reward_percent / risk_percent if risk_percent > 0 and reward_percent > 0 else 0

            # Enforce minimum Risk:Reward ratio
            if risk_reward < self.MIN_RISK_REWARD:
                confidence_level = ConfidenceLevel.AVOID

            if confidence_level == ConfidenceLevel.VERY_HIGH:
                recommendation = "STRONG SHORT"
            elif confidence_level == ConfidenceLevel.HIGH:
                recommendation = "SHORT"
            elif confidence_level == ConfidenceLevel.MEDIUM:
                recommendation = "WEAK SHORT - Proceed with caution"
            else:
                recommendation = "AVOID SHORT"

            return HighConfidencePick(
                symbol=symbol,
                current_price=Decimal(str(round(current_price, 2))),
                confidence_level=confidence_level,
                confidence_score=confidence_score,
                filters_passed=passed,
                filters_failed=failed,
                entry_price=entry,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                risk_percent=risk_percent,
                reward_percent=reward_percent,
                risk_reward_ratio=risk_reward,
                adx_value=adx_analysis.adx_value,
                rsi_value=rsi.value,
                volume_ratio=vol_analysis.volume_ratio,
                regime=adx_analysis.regime,
                recommendation=recommendation,
                reasons=[f.reason for f in passed],
            )

        except Exception as e:
            logger.error(f"Error analyzing short for {symbol}: {e}")
            return None

    def scan_both_sides(
        self,
        symbols: list[str],
        min_confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
    ) -> dict[str, list[HighConfidencePick]]:
        """Scan for both long AND short opportunities.

        In bear markets, the long scanner may return nothing. This method
        provides short candidates as well.

        Deduplication: if a stock appears in both lists (transitional regime),
        keep only the side with higher confidence.

        Args:
            symbols: List of symbols to scan
            min_confidence: Minimum confidence level

        Returns:
            Dictionary with "long" and "short" pick lists
        """
        long_picks = self.scan_stocks(symbols, min_confidence)
        long_symbols = {p.symbol for p in long_picks}

        short_picks = []
        confidence_order = [
            ConfidenceLevel.VERY_HIGH, ConfidenceLevel.HIGH,
            ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW, ConfidenceLevel.AVOID,
        ]
        min_index = confidence_order.index(min_confidence)

        for symbol in symbols:
            pick = self.analyze_short(symbol)
            if pick and pick.confidence_level != ConfidenceLevel.AVOID:
                pick_index = confidence_order.index(pick.confidence_level)
                if pick_index <= min_index:
                    short_picks.append(pick)

        short_picks.sort(key=lambda p: p.confidence_score, reverse=True)

        # Deduplicate: remove stocks appearing in both lists, keeping higher confidence
        short_symbol_map = {p.symbol: p for p in short_picks}
        long_symbol_map = {p.symbol: p for p in long_picks}
        overlapping = set(short_symbol_map.keys()) & set(long_symbol_map.keys())

        for sym in overlapping:
            long_score = long_symbol_map[sym].confidence_score
            short_score = short_symbol_map[sym].confidence_score
            if long_score >= short_score:
                short_picks = [p for p in short_picks if p.symbol != sym]
            else:
                long_picks = [p for p in long_picks if p.symbol != sym]

        return {
            "long": long_picks,
            "short": short_picks,
        }
