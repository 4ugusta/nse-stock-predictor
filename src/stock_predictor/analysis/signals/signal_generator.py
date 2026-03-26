"""Trading signal generator combining multiple indicators, sentiment, and fundamentals.

Major improvements over the original:
1. ATR-based stop loss and targets (volatility-adjusted)
2. Candlestick pattern integration (leading indicators)
3. Market regime adaptation (trend-following vs mean-reversion)
4. Sentiment and fundamental score integration
5. Conflicting signal detection and warnings
6. Multi-timeframe confirmation (optional)
7. Volume confirmation as bonus/penalty
"""

from dataclasses import dataclass
from decimal import Decimal

import pandas as pd

from stock_predictor.analysis.indicators.adx import ADXCalculator, MarketRegime
from stock_predictor.analysis.indicators.atr import ATRCalculator
from stock_predictor.analysis.indicators.bollinger import BollingerBandsCalculator
from stock_predictor.analysis.indicators.candlestick_patterns import CandlestickPatternDetector
from stock_predictor.analysis.indicators.macd import MACDCalculator, MACDSignalType
from stock_predictor.analysis.indicators.moving_averages import MovingAverageCalculator
from stock_predictor.analysis.indicators.rsi import RSICalculator
from stock_predictor.analysis.indicators.support_resistance import SupportResistanceCalculator
from stock_predictor.analysis.indicators.volume import VolumeAnalyzer
from stock_predictor.analysis.market_intelligence import (
    fetch_india_vix,
    analyze_gap,
    get_market_breadth,
    get_fii_dii_activity,
)
from stock_predictor.core.entities.signal import (
    IndicatorSignal,
    SignalStrength,
    SignalType,
    Timeframe,
    TradingSignal,
    TradingStyle,
)
from stock_predictor.core.validation import DataQuality, detect_circuit_breaker, validate_ohlcv_data
from stock_predictor.infrastructure.config.constants import (
    MIN_DATA_WARMUP,
    MIN_LIQUIDITY_TURNOVER_INR,
    MIN_LIQUIDITY_VOLUME,
    REGIME_SIGNAL_THRESHOLDS,
    SIGNAL_THRESHOLDS,
    TRANSACTION_COST_BUFFER_PCT,
)

import logging

logger = logging.getLogger(__name__)


@dataclass
class IndicatorWeights:
    """Weights for different indicators in signal calculation."""

    moving_average: float = 0.20
    rsi: float = 0.20
    macd: float = 0.25
    bollinger: float = 0.15
    support_resistance: float = 0.20


@dataclass
class ExternalScores:
    """Optional external scores from sentiment and fundamental analysis.

    Pass these to the signal generator to integrate non-technical data
    into the final signal. Each score ranges from -1.0 to 1.0.
    """

    sentiment_score: float | None = None  # From SentimentAnalyzer
    fundamental_score: float | None = None  # From FundamentalFetcher (normalized to -1..1)
    sentiment_weight: float = 0.10  # How much sentiment affects the signal
    fundamental_weight: float = 0.10  # How much fundamentals affect the signal


@dataclass
class SignalDiagnostics:
    """Detailed diagnostics for debugging and transparency."""

    raw_technical_score: float
    regime_adjusted_score: float
    external_adjusted_score: float
    final_score: float
    market_regime: MarketRegime
    conflicting_signals: list[str]
    candlestick_patterns: list[str]
    atr_percent: float  # Stock's volatility as % of price
    regime_strategy: str  # "trend_following" or "mean_reversion"


# Weights for different trading styles
STYLE_WEIGHTS = {
    TradingStyle.INTRADAY: IndicatorWeights(
        moving_average=0.15,
        rsi=0.25,
        macd=0.25,
        bollinger=0.20,
        support_resistance=0.15,
    ),
    TradingStyle.SWING: IndicatorWeights(
        moving_average=0.20,
        rsi=0.20,
        macd=0.25,
        bollinger=0.15,
        support_resistance=0.20,
    ),
    TradingStyle.POSITIONAL: IndicatorWeights(
        moving_average=0.30,
        rsi=0.15,
        macd=0.20,
        bollinger=0.10,
        support_resistance=0.25,
    ),
}

# Regime-adapted weights: in sideways markets, shift to mean-reversion indicators
SIDEWAYS_WEIGHTS = IndicatorWeights(
    moving_average=0.10,  # MAs are useless in sideways
    rsi=0.30,  # RSI mean-reversion works well
    macd=0.10,  # MACD generates whipsaws
    bollinger=0.30,  # BB bounce strategy works
    support_resistance=0.20,
)

# Volatile regime: high ADX but close DI lines — choppy, trust S/R and BB
VOLATILE_WEIGHTS = IndicatorWeights(
    moving_average=0.10,
    rsi=0.25,
    macd=0.10,
    bollinger=0.25,
    support_resistance=0.30,  # S/R most reliable in volatile chop
)


class SignalGenerator:
    """Generates trading signals by combining multiple indicators."""

    def __init__(self) -> None:
        """Initialize signal generator with all indicator calculators."""
        self.ma_calc = MovingAverageCalculator()
        self.rsi_calc = RSICalculator()
        self.macd_calc = MACDCalculator()
        self.bb_calc = BollingerBandsCalculator()
        self.sr_calc = SupportResistanceCalculator()
        self.adx_calc = ADXCalculator()
        self.atr_calc = ATRCalculator()
        self.volume_calc = VolumeAnalyzer()
        self.pattern_detector = CandlestickPatternDetector()

    def _score_to_signal_type(
        self, score: float, regime: MarketRegime | None = None
    ) -> SignalType:
        """Convert numerical score to signal type.

        Uses tighter thresholds in sideways/volatile markets to reduce
        false signals when indicators are unreliable.
        """
        if regime == MarketRegime.SIDEWAYS:
            thresholds = REGIME_SIGNAL_THRESHOLDS["sideways"]
        elif regime == MarketRegime.VOLATILE:
            thresholds = REGIME_SIGNAL_THRESHOLDS["volatile"]
        elif regime in (MarketRegime.STRONG_UPTREND, MarketRegime.STRONG_DOWNTREND):
            thresholds = REGIME_SIGNAL_THRESHOLDS["trending"]
        else:
            thresholds = SIGNAL_THRESHOLDS

        if score >= thresholds["strong_buy"]:
            return SignalType.STRONG_BUY
        elif score >= thresholds["buy"]:
            return SignalType.BUY
        elif score <= thresholds["strong_sell"]:
            return SignalType.STRONG_SELL
        elif score <= thresholds["sell"]:
            return SignalType.SELL
        return SignalType.HOLD

    def _score_to_strength(self, score: float) -> SignalStrength:
        """Convert numerical score to signal strength."""
        abs_score = abs(score)
        if abs_score >= 0.7:
            return SignalStrength.STRONG
        elif abs_score >= 0.4:
            return SignalStrength.MODERATE
        return SignalStrength.WEAK

    def _style_to_timeframe(self, style: TradingStyle) -> Timeframe:
        """Convert trading style to timeframe."""
        return {
            TradingStyle.INTRADAY: Timeframe.INTRADAY,
            TradingStyle.SWING: Timeframe.SWING,
            TradingStyle.POSITIONAL: Timeframe.POSITIONAL,
        }[style]

    def _detect_conflicting_signals(
        self, signals: list[IndicatorSignal]
    ) -> tuple[list[str], float, int]:
        """Detect when indicators are giving conflicting signals.

        Uses score magnitude (not just count) to weight conflict severity.
        Volume/pattern signals are separated from core directional indicators.

        Returns:
            - conflicts: list of conflict description strings
            - agreement_ratio: 0.0-1.0 weighted ratio of majority vs total
            - conflict_count: number of minority-side (conflicting) indicators

        The conflict_count enables graduated score penalties:
        - 1-2 conflicting indicators: multiply score by 0.9
        - 3-4 conflicting indicators: multiply score by 0.7
        - 5+ conflicting indicators: multiply score by 0.5
        """
        conflicts = []

        # Separate core directional indicators from confirmation indicators
        CONFIRMATION_INDICATORS = {"volume", "candlestick_patterns"}
        core_signals = [s for s in signals if s.indicator not in CONFIRMATION_INDICATORS]
        confirm_signals = [s for s in signals if s.indicator in CONFIRMATION_INDICATORS]

        # Use score magnitude for weighted conflict detection
        bullish = [s for s in core_signals if s.score > 0.10]
        bearish = [s for s in core_signals if s.score < -0.10]

        if bullish and bearish:
            bull_names = [f"{s.indicator}({s.score:+.2f})" for s in bullish]
            bear_names = [f"{s.indicator}({s.score:+.2f})" for s in bearish]
            conflicts.append(
                f"CONFLICTING: {', '.join(bull_names)} bullish vs "
                f"{', '.join(bear_names)} bearish"
            )

        # Weighted agreement: weight by abs(score) so strong disagreements count more
        bull_weight = sum(abs(s.score) for s in bullish)
        bear_weight = sum(abs(s.score) for s in bearish)
        total_weight = bull_weight + bear_weight

        if total_weight > 0:
            majority_weight = max(bull_weight, bear_weight)
            agreement_ratio = majority_weight / total_weight
            minority = min(len(bullish), len(bearish))
        else:
            minority = 0
            agreement_ratio = 1.0

        # Add confirmation indicator conflicts as warnings (lower severity)
        for cs in confirm_signals:
            if bullish and cs.score < -0.3:
                conflicts.append(
                    f"WARNING: {cs.indicator} bearish ({cs.score:+.2f}) vs bullish technicals"
                )
            elif bearish and cs.score > 0.3:
                conflicts.append(
                    f"WARNING: {cs.indicator} bullish ({cs.score:+.2f}) vs bearish technicals"
                )

        return conflicts, agreement_ratio, minority

    def _get_ma_signal(self, data: pd.DataFrame, style: TradingStyle) -> IndicatorSignal:
        """Generate signal from moving averages."""
        if style == TradingStyle.INTRADAY:
            short, long = 9, 21
        elif style == TradingStyle.SWING:
            short, long = 21, 50
        else:
            short, long = 50, 200

        trend, score = self.ma_calc.get_trend_signal(data, short, long)
        crossover = self.ma_calc.detect_crossover(data, short, long)

        reason_parts = [f"Price {trend} trend (EMA {short}/{long})"]

        if crossover:
            if crossover.is_bullish:
                score = min(1.0, score + 0.3)
                reason_parts.append("Golden cross detected")
            else:
                score = max(-1.0, score - 0.3)
                reason_parts.append("Death cross detected")

        if style == TradingStyle.POSITIONAL:
            above_200 = self.ma_calc.is_price_above_ma(data, 200)
            if above_200:
                score = min(1.0, score + 0.1)
                reason_parts.append("Above 200 SMA")
            else:
                score = max(-1.0, score - 0.1)
                reason_parts.append("Below 200 SMA")

        return IndicatorSignal(
            indicator="moving_average",
            value=score,
            signal=self._score_to_signal_type(score),
            score=score,
            reason="; ".join(reason_parts),
        )

    def _get_rsi_signal(self, data: pd.DataFrame) -> IndicatorSignal:
        """Generate signal from RSI."""
        result = self.rsi_calc.analyze(data)
        divergence = self.rsi_calc.detect_divergence(data)

        score = result.signal_strength
        reason_parts = [f"RSI: {result.value:.1f} ({result.zone.value})"]

        if divergence.divergence_type == "bullish":
            score = min(1.0, score + 0.3 * divergence.strength)
            reason_parts.append("Bullish divergence")
        elif divergence.divergence_type == "bearish":
            score = max(-1.0, score - 0.3 * divergence.strength)
            reason_parts.append("Bearish divergence")

        return IndicatorSignal(
            indicator="rsi",
            value=result.value,
            signal=self._score_to_signal_type(score),
            score=score,
            reason="; ".join(reason_parts),
        )

    def _get_macd_signal(self, data: pd.DataFrame) -> IndicatorSignal:
        """Generate signal from MACD."""
        analysis = self.macd_calc.analyze(data)

        score = analysis.signal_strength
        reason_parts = [f"MACD {analysis.trend}"]

        if analysis.signal_type == MACDSignalType.BULLISH_CROSSOVER:
            reason_parts.append("Bullish crossover")
        elif analysis.signal_type == MACDSignalType.BEARISH_CROSSOVER:
            reason_parts.append("Bearish crossover")

        if analysis.histogram_trend == "increasing":
            reason_parts.append("Momentum increasing")
        elif analysis.histogram_trend == "decreasing":
            reason_parts.append("Momentum decreasing")

        return IndicatorSignal(
            indicator="macd",
            value=analysis.macd_value,
            signal=self._score_to_signal_type(score),
            score=score,
            reason="; ".join(reason_parts),
        )

    def _get_bollinger_signal(self, data: pd.DataFrame) -> IndicatorSignal:
        """Generate signal from Bollinger Bands."""
        analysis = self.bb_calc.analyze(data)

        score = analysis.signal_strength
        reason_parts = [f"BB zone: {analysis.zone.value}"]

        if analysis.squeeze:
            reason_parts.append("Squeeze detected (breakout imminent)")
        if analysis.expansion:
            reason_parts.append("Band expansion (high volatility)")

        return IndicatorSignal(
            indicator="bollinger",
            value=analysis.percent_b,
            signal=self._score_to_signal_type(score),
            score=score,
            reason="; ".join(reason_parts),
        )

    def _get_sr_signal(self, data: pd.DataFrame) -> IndicatorSignal:
        """Generate signal from Support/Resistance."""
        score = self.sr_calc.get_signal_strength(data)
        analysis = self.sr_calc.analyze(data)

        reason_parts = []

        if analysis.distance_to_support_pct is not None:
            if analysis.distance_to_support_pct < 2:
                reason_parts.append(f"Near support ({analysis.distance_to_support_pct:.1f}% away)")
            else:
                reason_parts.append(f"Support at {analysis.nearest_support:.2f}")

        if analysis.distance_to_resistance_pct is not None:
            if analysis.distance_to_resistance_pct < 2:
                reason_parts.append(
                    f"Near resistance ({analysis.distance_to_resistance_pct:.1f}% away)"
                )
            else:
                reason_parts.append(f"Resistance at {analysis.nearest_resistance:.2f}")

        if not reason_parts:
            reason_parts.append("No clear S/R levels nearby")

        return IndicatorSignal(
            indicator="support_resistance",
            value=score,
            signal=self._score_to_signal_type(score),
            score=score,
            reason="; ".join(reason_parts),
        )

    def _get_volume_signal(self, data: pd.DataFrame) -> IndicatorSignal:
        """Generate signal from volume analysis."""
        analysis = self.volume_calc.analyze(data)

        score_map = {
            "strong_bullish": 0.6,
            "bullish": 0.3,
            "accumulation": 0.4,
            "neutral": 0.0,
            "bearish": -0.3,
            "strong_bearish": -0.6,
            "distribution": -0.5,
        }
        score = score_map.get(analysis.signal.value, 0.0)

        reason = (
            f"Volume: {analysis.volume_ratio:.1f}x avg, "
            f"OBV: {analysis.on_balance_volume_trend}, "
            f"Signal: {analysis.signal.value}"
        )

        return IndicatorSignal(
            indicator="volume",
            value=analysis.volume_ratio,
            signal=self._score_to_signal_type(score),
            score=score,
            reason=reason,
        )

    def _get_pattern_signal(self, data: pd.DataFrame) -> IndicatorSignal:
        """Generate signal from candlestick patterns."""
        analysis = self.pattern_detector.analyze(data)

        return IndicatorSignal(
            indicator="candlestick_patterns",
            value=analysis.net_score,
            signal=self._score_to_signal_type(analysis.net_score),
            score=analysis.net_score,
            reason=analysis.summary,
        )

    def _adapt_weights_to_regime(
        self,
        base_weights: IndicatorWeights,
        regime: MarketRegime,
    ) -> tuple[IndicatorWeights, str]:
        """Adapt indicator weights based on market regime.

        In trending markets, use trend-following weights.
        In sideways markets, shift to mean-reversion weights.
        In volatile regime, use cautious weights favouring S/R and BB.
        """
        if regime == MarketRegime.SIDEWAYS:
            return SIDEWAYS_WEIGHTS, "mean_reversion"

        if regime == MarketRegime.VOLATILE:
            return VOLATILE_WEIGHTS, "volatile_caution"

        if regime in (MarketRegime.STRONG_UPTREND, MarketRegime.STRONG_DOWNTREND):
            return IndicatorWeights(
                moving_average=base_weights.moving_average * 1.3,
                rsi=base_weights.rsi * 0.7,
                macd=base_weights.macd * 1.2,
                bollinger=base_weights.bollinger * 0.6,
                support_resistance=base_weights.support_resistance,
            ), "trend_following"

        return base_weights, "trend_following"

    def _calculate_atr_targets(
        self,
        data: pd.DataFrame,
        signal_type: SignalType,
        style: TradingStyle,
    ) -> tuple[Decimal | None, Decimal | None]:
        """Calculate ATR-based stop loss and target instead of fixed percentages.

        Targets are adjusted by TRANSACTION_COST_BUFFER_PCT so the displayed
        R:R accounts for real round-trip costs (STT, brokerage, exchange fees).
        """
        atr_result = self.atr_calc.analyze(data)
        entry_price = float(data["close"].iloc[-1])
        atr = atr_result.current_atr
        cost_buffer = entry_price * (TRANSACTION_COST_BUFFER_PCT / 100)

        stop_mult = {
            TradingStyle.INTRADAY: 1.0,
            TradingStyle.SWING: 1.5,
            TradingStyle.POSITIONAL: 2.0,
        }[style]

        target_mult = {
            TradingStyle.INTRADAY: 2.0,
            TradingStyle.SWING: 3.0,
            TradingStyle.POSITIONAL: 4.0,
        }[style]

        sr_analysis = self.sr_calc.analyze(data)

        if signal_type in (SignalType.BUY, SignalType.STRONG_BUY):
            stop_loss = entry_price - (atr * stop_mult)
            atr_target = entry_price + (atr * target_mult) + cost_buffer

            if sr_analysis.nearest_resistance:
                sr_target = sr_analysis.nearest_resistance
                risk = entry_price - stop_loss
                sr_reward = sr_target - entry_price - cost_buffer
                if risk > 0 and sr_reward / risk >= 1.5:
                    target = sr_target
                else:
                    target = atr_target
            else:
                target = atr_target

            return Decimal(str(round(target, 2))), Decimal(str(round(stop_loss, 2)))

        elif signal_type in (SignalType.SELL, SignalType.STRONG_SELL):
            stop_loss = entry_price + (atr * stop_mult)
            atr_target = entry_price - (atr * target_mult) - cost_buffer
            # Floor: target can never go below 0.01 (penny floor)
            atr_target = max(atr_target, 0.01)

            if sr_analysis.nearest_support:
                sr_target = max(sr_analysis.nearest_support, 0.01)
                risk = stop_loss - entry_price
                sr_reward = entry_price - sr_target - cost_buffer
                if risk > 0 and sr_reward > 0 and sr_reward / risk >= 1.5:
                    target = sr_target
                else:
                    target = atr_target
            else:
                target = atr_target

            return Decimal(str(round(target, 2))), Decimal(str(round(stop_loss, 2)))

        return None, None

    def _detect_whipsaw(self, data: pd.DataFrame, style: TradingStyle = TradingStyle.SWING) -> bool:
        """Detect whipsaw conditions where price reverses rapidly.

        Improved: uses ATR-relative magnitude to ignore tiny reversals that
        are just noise. Only counts reversals that move at least 0.3 * ATR.

        Returns True if whipsaw detected (signal should be degraded).
        """
        lookback_map = {
            TradingStyle.INTRADAY: 20,
            TradingStyle.SWING: 10,
            TradingStyle.POSITIONAL: 30,
        }
        lookback = lookback_map.get(style, 10)

        if len(data) < lookback + 1:
            return False

        # Get ATR for magnitude threshold
        try:
            atr_result = self.atr_calc.analyze(data)
            atr = atr_result.current_atr
        except Exception:
            atr = 0

        recent = data["close"].iloc[-lookback:]
        changes = recent.diff().dropna()

        if len(changes) < 3:
            return False

        # Only count reversals that are meaningful (>= 0.3 * ATR)
        min_move = atr * 0.3 if atr > 0 else 0
        meaningful_changes = changes[changes.abs() >= min_move]

        if len(meaningful_changes) < 3:
            return False

        # Count sign changes in meaningful moves
        signs = (meaningful_changes > 0).astype(int)
        reversals = signs.diff().abs().sum()

        reversal_ratio = reversals / max(len(meaningful_changes) - 1, 1)

        return reversal_ratio > 0.6

    def _integrate_external_scores(
        self,
        technical_score: float,
        external: ExternalScores | None,
    ) -> float:
        """Integrate sentiment and fundamental scores into the technical score.

        External scores can boost or dampen the signal but cannot flip it.
        The adjustment is capped at ±0.2 to prevent overriding technicals.
        """
        if external is None:
            return technical_score

        adjustment = 0.0

        if external.sentiment_score is not None:
            clamped_sentiment = max(-1.0, min(1.0, external.sentiment_score))
            adjustment += clamped_sentiment * external.sentiment_weight

        if external.fundamental_score is not None:
            clamped_fundamental = max(-1.0, min(1.0, external.fundamental_score))
            adjustment += clamped_fundamental * external.fundamental_weight

        # Cap at ±0.15: external scores should nudge, not override technicals.
        # Higher caps let noisy sentiment data flip the signal direction.
        adjustment = max(-0.15, min(0.15, adjustment))

        return max(-1.0, min(1.0, technical_score + adjustment))

    def generate(
        self,
        data: pd.DataFrame,
        symbol: str,
        style: TradingStyle = TradingStyle.SWING,
        external_scores: ExternalScores | None = None,
    ) -> TradingSignal:
        """Generate a complete trading signal.

        Args:
            data: DataFrame with OHLCV data
            symbol: Stock symbol
            style: Trading style
            external_scores: Optional sentiment/fundamental scores

        Returns:
            TradingSignal with entry/exit levels
        """
        # Data warmup guard — indicators need minimum history to be reliable
        if len(data) < MIN_DATA_WARMUP:
            entry_price = float(data["close"].iloc[-1])
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                strength=SignalStrength.WEAK,
                timeframe=self._style_to_timeframe(style),
                entry_price=Decimal(str(round(entry_price, 2))),
                target_price=None,
                stop_loss=None,
                confidence=0.0,
                reasons=[f"Insufficient data ({len(data)} bars, need {MIN_DATA_WARMUP})"],
                indicator_signals=[],
            )

        # Liquidity filter — reject illiquid stocks that can't be traded efficiently
        if "volume" in data.columns:
            avg_volume = data["volume"].iloc[-20:].mean()
            if avg_volume < MIN_LIQUIDITY_VOLUME:
                entry_price = float(data["close"].iloc[-1])
                return TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.HOLD,
                    strength=SignalStrength.WEAK,
                    timeframe=self._style_to_timeframe(style),
                    entry_price=Decimal(str(round(entry_price, 2))),
                    target_price=None,
                    stop_loss=None,
                    confidence=0.0,
                    reasons=[
                        f"Insufficient liquidity (avg vol: {avg_volume:,.0f}, "
                        f"need {MIN_LIQUIDITY_VOLUME:,})"
                    ],
                    indicator_signals=[],
                )

            # Notional turnover check (price * volume)
            notional = self.volume_calc.calculate_notional_turnover(data)
            if notional < MIN_LIQUIDITY_TURNOVER_INR:
                entry_price = float(data["close"].iloc[-1])
                return TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.HOLD,
                    strength=SignalStrength.WEAK,
                    timeframe=self._style_to_timeframe(style),
                    entry_price=Decimal(str(round(entry_price, 2))),
                    target_price=None,
                    stop_loss=None,
                    confidence=0.0,
                    reasons=[
                        f"Insufficient notional turnover (Rs {notional:,.0f}/day, "
                        f"need Rs {MIN_LIQUIDITY_TURNOVER_INR:,})"
                    ],
                    indicator_signals=[],
                )

        # Circuit breaker check — skip signals on halted stocks
        if len(data) >= 2:
            cb_series = detect_circuit_breaker(data)
            if cb_series.iloc[-1]:
                entry_price = float(data["close"].iloc[-1])
                return TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.HOLD,
                    strength=SignalStrength.WEAK,
                    timeframe=self._style_to_timeframe(style),
                    entry_price=Decimal(str(round(entry_price, 2))),
                    target_price=None,
                    stop_loss=None,
                    confidence=0.0,
                    reasons=["Circuit breaker detected — stock likely halted"],
                    indicator_signals=[],
                )

        # Data quality validation
        validation = validate_ohlcv_data(data)
        if not validation.is_usable:
            entry_price = float(data["close"].iloc[-1])
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.NO_EDGE,
                strength=SignalStrength.WEAK,
                timeframe=self._style_to_timeframe(style),
                entry_price=Decimal(str(round(entry_price, 2))),
                target_price=None,
                stop_loss=None,
                confidence=0.0,
                reasons=[
                    f"Data quality insufficient: {'; '.join(validation.issues)}"
                ],
                indicator_signals=[],
            )
        if validation.quality == DataQuality.DEGRADED:
            data_quality_penalty = 0.15
        elif validation.issues:  # GOOD quality but has minor issues
            data_quality_penalty = 0.05
        else:
            data_quality_penalty = 0.0

        base_weights = STYLE_WEIGHTS[style]

        # Detect market regime and adapt weights
        try:
            adx_analysis = self.adx_calc.analyze(data)
            regime = adx_analysis.regime
        except Exception as e:
            logger.warning(f"ADX calculation failed for {symbol}: {e}. Using SIDEWAYS fallback.")
            regime = MarketRegime.SIDEWAYS

        weights, regime_strategy = self._adapt_weights_to_regime(base_weights, regime)

        # Normalize weights to sum to 1.0
        total_weight = (
            weights.moving_average + weights.rsi + weights.macd
            + weights.bollinger + weights.support_resistance
        )

        # Generate individual signals
        ma_signal = self._get_ma_signal(data, style)
        rsi_signal = self._get_rsi_signal(data)
        macd_signal = self._get_macd_signal(data)
        bb_signal = self._get_bollinger_signal(data)
        sr_signal = self._get_sr_signal(data)

        indicator_signals = [ma_signal, rsi_signal, macd_signal, bb_signal, sr_signal]

        # Calculate weighted score (normalized)
        raw_score = (
            ma_signal.score * weights.moving_average
            + rsi_signal.score * weights.rsi
            + macd_signal.score * weights.macd
            + bb_signal.score * weights.bollinger
            + sr_signal.score * weights.support_resistance
        ) / total_weight

        regime_score = raw_score

        # Volume confirmation: bonus/penalty (not a core weighted component)
        volume_signal = None
        failed_indicators: list[str] = []
        try:
            volume_signal = self._get_volume_signal(data)
            indicator_signals.append(volume_signal)
            volume_adjustment = volume_signal.score * 0.15
            regime_score = max(-1.0, min(1.0, regime_score + volume_adjustment))
        except Exception as e:
            failed_indicators.append(f"volume: {e}")
            logger.warning(f"Volume analysis failed for {symbol}: {e}")

        # Volume hard-block: strong contradicting volume kills the signal entirely.
        # Rationale: a buy signal with heavy distribution volume is a trap.
        if volume_signal and volume_signal.score <= -0.5 and regime_score > 0:
            regime_score = 0.0  # Zero out bullish signal on strong bearish volume
        elif volume_signal and volume_signal.score >= 0.5 and regime_score < 0:
            regime_score = 0.0  # Zero out bearish signal on strong bullish volume

        # Candlestick patterns: leading indicator bonus (only if aligned with signal)
        pattern_signal = None
        try:
            pattern_signal = self._get_pattern_signal(data)
            indicator_signals.append(pattern_signal)
            if abs(pattern_signal.score) > 0.2:
                # Only apply bonus if pattern agrees with signal direction
                pattern_agrees = (pattern_signal.score > 0 and regime_score > 0) or \
                                 (pattern_signal.score < 0 and regime_score < 0)
                if pattern_agrees:
                    pattern_adjustment = pattern_signal.score * 0.15
                    regime_score = max(-1.0, min(1.0, regime_score + pattern_adjustment))
                else:
                    # Pattern strongly disagrees: kill the signal if pattern is strong
                    if abs(pattern_signal.score) >= 0.5:
                        regime_score = 0.0  # Strong pattern contradiction -> no edge
                    else:
                        penalty = 0.25 * min(abs(pattern_signal.score), 1.0)
                        regime_score *= (1.0 - penalty)
        except Exception as e:
            failed_indicators.append(f"candlestick: {e}")
            logger.warning(f"Candlestick analysis failed for {symbol}: {e}")

        # Integrate external scores BEFORE volume hard-block so they can't bypass it
        final_score = self._integrate_external_scores(regime_score, external_scores)

        # Detect conflicting signals with graduated penalty
        conflicts, agreement_ratio, conflict_count = self._detect_conflicting_signals(
            indicator_signals
        )

        # Graduated score penalty based on number of conflicting indicators
        if conflict_count >= 5:
            final_score *= 0.5
        elif conflict_count >= 3:
            final_score *= 0.7
        elif conflict_count >= 1:
            final_score *= 0.9

        # Confidence = indicator agreement, NOT absolute score
        # Build indicator_scores dict from all generated signals
        indicator_scores = {sig.indicator: sig.score for sig in indicator_signals}

        # Count how many indicators agree with the final direction
        # Use a minimum threshold so near-zero scores don't inflate agreement
        AGREEMENT_THRESHOLD = 0.15  # Higher threshold to exclude noise from agreement count
        if final_score > 0:
            agreeing = sum(1 for s in indicator_scores.values() if s > AGREEMENT_THRESHOLD)
        elif final_score < 0:
            agreeing = sum(1 for s in indicator_scores.values() if s < -AGREEMENT_THRESHOLD)
        else:
            agreeing = 0

        total_indicators = max(len(indicator_scores), 1)
        indicator_agreement_ratio = agreeing / total_indicators

        # Confidence combines agreement strength (not just count) with score magnitude.
        # Weight each agreeing indicator by how strongly it agrees, not just binary count.
        if final_score > 0:
            agreeing_scores = [s for s in indicator_scores.values() if s > AGREEMENT_THRESHOLD]
            agreement_strength = (
                sum(min(s, 1.0) for s in agreeing_scores) / len(agreeing_scores)
                if agreeing_scores else 0.0
            )
        elif final_score < 0:
            agreeing_scores = [s for s in indicator_scores.values() if s < -AGREEMENT_THRESHOLD]
            agreement_strength = (
                sum(min(abs(s), 1.0) for s in agreeing_scores) / len(agreeing_scores)
                if agreeing_scores else 0.0
            )
        else:
            agreement_strength = 0.0

        # Confidence = how many agree * how strongly they agree * score magnitude
        score_magnitude = min(abs(final_score), 1.0)
        confidence = (
            indicator_agreement_ratio * 0.35
            + agreement_strength * 0.35
            + score_magnitude * 0.30
        )
        confidence = min(confidence, 0.70)  # Cap at 70%: technical analysis alone cannot justify higher certainty

        # Graduated confidence penalty based on conflict agreement ratio
        if conflicts:
            if agreement_ratio >= 0.8:  # 4 vs 1 — mild disagreement
                confidence *= 0.9
            elif agreement_ratio >= 0.6:  # 3 vs 2 — moderate disagreement
                confidence *= 0.7
            else:  # Even split or worse — severe disagreement
                confidence *= 0.5

        # Whipsaw filter: scale penalty by reversal severity
        if self._detect_whipsaw(data, style):
            # Get reversal ratio for graduated penalty
            lookback = {TradingStyle.INTRADAY: 20, TradingStyle.SWING: 10, TradingStyle.POSITIONAL: 30}.get(style, 10)
            recent = data["close"].iloc[-lookback:] if len(data) >= lookback else data["close"]
            changes = recent.diff().dropna()
            if len(changes) >= 3:
                signs = (changes > 0).astype(int)
                rev_ratio = float(signs.diff().abs().sum()) / max(len(changes) - 1, 1)
                # Scale: 0.6 reversal ratio = 0.6x, 0.8 = 0.4x, 1.0 = 0.2x
                whipsaw_multiplier = max(0.2, 1.0 - rev_ratio)
                confidence *= whipsaw_multiplier
            else:
                confidence *= 0.6

        # Apply data quality penalty (from validation above)
        confidence -= data_quality_penalty

        # Penalty for failed indicators: each missing indicator degrades confidence
        if failed_indicators:
            per_indicator_penalty = 0.05
            confidence -= len(failed_indicators) * per_indicator_penalty

        # India VIX confidence adjustment — high fear = less reliable signals
        vix_reading = None
        try:
            vix_reading = fetch_india_vix()
            if vix_reading.regime != "unavailable":
                confidence *= vix_reading.confidence_adjustment
        except Exception as e:
            logger.debug(f"VIX fetch skipped for {symbol}: {e}")

        # Market breadth confidence adjustment — broad participation confirms signals
        breadth = None
        try:
            breadth = get_market_breadth()
            if breadth.breadth_signal != "unavailable":
                confidence *= breadth.confidence_adjustment
        except Exception as e:
            logger.debug(f"Market breadth fetch skipped for {symbol}: {e}")

        # FII/DII flow confidence adjustment — institutional money flow
        fii_dii = None
        try:
            fii_dii = get_fii_dii_activity()
            if fii_dii.flow_signal != "unavailable":
                confidence *= fii_dii.confidence_adjustment
        except Exception as e:
            logger.debug(f"FII/DII fetch skipped for {symbol}: {e}")

        confidence = max(0.0, min(1.0, confidence))

        # Determine signal type and strength (regime-adaptive thresholds)
        signal_type = self._score_to_signal_type(final_score, regime)
        strength = self._score_to_strength(final_score)
        timeframe = self._style_to_timeframe(style)

        # NO_EDGE detection: expanded criteria to catch marginal signals
        is_no_edge = (
            confidence < 0.25
            or (indicator_agreement_ratio < 0.4 and abs(final_score) < 0.15)
            or (confidence < 0.35 and abs(final_score) < 0.10)
            or (agreement_strength < 0.1 and abs(final_score) < 0.20)
        )
        if is_no_edge:
            signal_type = SignalType.NO_EDGE
            # keep actual confidence for transparency

        # ATR-based targets instead of fixed percentages
        entry_price = float(data["close"].iloc[-1])
        target, stop_loss = self._calculate_atr_targets(data, signal_type, style)

        # Gap analysis — warn about overnight gaps that may affect entry
        gap = analyze_gap(data)

        # Collect reasons from significant signals (lowered from 0.3 to 0.2 for better transparency)
        reasons = []
        for sig in indicator_signals:
            if abs(sig.score) >= 0.2:
                reasons.append(sig.reason)

        if conflicts:
            reasons.extend(conflicts)

        reasons.append(f"Market regime: {regime.value} ({regime_strategy})")

        # VIX context
        if vix_reading and vix_reading.regime not in ("unavailable", "low", "normal"):
            reasons.append(vix_reading.description)

        # Market breadth context
        if breadth and breadth.breadth_signal not in ("unavailable", "neutral"):
            reasons.append(breadth.description)

        # FII/DII flow context
        if fii_dii and fii_dii.flow_signal != "unavailable":
            reasons.append(fii_dii.description)

        # Gap analysis context
        if gap.has_gap:
            direction = "up" if gap.gap_percent > 0 else "down"
            reasons.append(
                f"Gap {direction} {abs(gap.gap_percent):.1f}% ({gap.gap_type}) — "
                f"fill prob {gap.fill_probability:.0%}, entry: {gap.entry_adjustment.replace('_', ' ')}"
            )

        # Add failed indicator warnings to reasons
        if failed_indicators:
            reasons.append(
                f"WARNING: {len(failed_indicators)} indicator(s) failed: "
                + ", ".join(failed_indicators)
            )

        # Build diagnostics
        try:
            atr_pct = self.atr_calc.analyze(data).atr_percent
        except Exception:
            atr_pct = 0

        pattern_names = []
        if pattern_signal:
            try:
                pa = self.pattern_detector.analyze(data)
                pattern_names = [p.name for p in pa.patterns_found]
            except Exception:
                pass

        diagnostics = SignalDiagnostics(
            raw_technical_score=raw_score,
            regime_adjusted_score=regime_score,
            external_adjusted_score=final_score,
            final_score=final_score,
            market_regime=regime,
            conflicting_signals=conflicts,
            candlestick_patterns=pattern_names,
            atr_percent=atr_pct,
            regime_strategy=regime_strategy,
        )

        signal = TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            timeframe=timeframe,
            entry_price=Decimal(str(round(entry_price, 2))),
            target_price=target,
            stop_loss=stop_loss,
            confidence=confidence,
            reasons=reasons,
            indicator_signals=indicator_signals,
        )

        # Attach diagnostics for callers that need it
        signal.diagnostics = diagnostics  # type: ignore[attr-defined]

        return signal

    def generate_all_styles(
        self, data: pd.DataFrame, symbol: str
    ) -> dict[TradingStyle, TradingSignal]:
        """Generate signals for all trading styles."""
        return {
            TradingStyle.INTRADAY: self.generate(data, symbol, TradingStyle.INTRADAY),
            TradingStyle.SWING: self.generate(data, symbol, TradingStyle.SWING),
            TradingStyle.POSITIONAL: self.generate(data, symbol, TradingStyle.POSITIONAL),
        }

    def generate_multi_timeframe(
        self,
        data_daily: pd.DataFrame,
        data_weekly: pd.DataFrame,
        symbol: str,
        style: TradingStyle = TradingStyle.SWING,
    ) -> TradingSignal:
        """Generate signal with multi-timeframe confirmation.

        The weekly signal acts as a trend filter — only take daily
        signals in the direction of the weekly trend.

        Args:
            data_daily: Daily OHLCV data
            data_weekly: Weekly OHLCV data
            symbol: Stock symbol
            style: Trading style

        Returns:
            TradingSignal with multi-timeframe confirmation
        """
        daily_signal = self.generate(data_daily, symbol, style)
        weekly_signal = self.generate(data_weekly, symbol, TradingStyle.POSITIONAL)

        weekly_bullish = weekly_signal.signal_type in (SignalType.BUY, SignalType.STRONG_BUY)
        weekly_bearish = weekly_signal.signal_type in (SignalType.SELL, SignalType.STRONG_SELL)
        daily_bullish = daily_signal.signal_type in (SignalType.BUY, SignalType.STRONG_BUY)
        daily_bearish = daily_signal.signal_type in (SignalType.SELL, SignalType.STRONG_SELL)

        if (daily_bullish and weekly_bullish) or (daily_bearish and weekly_bearish):
            # Scale boost by weekly signal strength (not fixed 1.2x)
            weekly_boost = 1.0 + (weekly_signal.confidence * 0.3)  # 1.0 to 1.3x
            daily_signal.confidence = min(1.0, daily_signal.confidence * weekly_boost)
            daily_signal.reasons.append(
                f"Multi-TF confirmed: weekly is {weekly_signal.signal_type.value}"
            )
        elif (daily_bullish and weekly_bearish) or (daily_bearish and weekly_bullish):
            # Weekly opposes daily: kill the signal entirely
            daily_signal.signal_type = SignalType.NO_EDGE
            daily_signal.confidence *= 0.3
            daily_signal.reasons.append(
                f"BLOCKED: Weekly opposes daily "
                f"(weekly: {weekly_signal.signal_type.value}) — no edge"
            )
        elif daily_bullish or daily_bearish:
            # Weekly is HOLD/sideways — significant penalty, trend lacks confirmation
            daily_signal.confidence *= 0.65
            daily_signal.reasons.append(
                f"Weekly trend neutral ({weekly_signal.signal_type.value}) — reduced conviction"
            )

        return daily_signal

    def get_indicator_summary(self, data: pd.DataFrame) -> dict:
        """Get a summary of all indicator values."""
        rsi_result = self.rsi_calc.analyze(data)
        macd_analysis = self.macd_calc.analyze(data)
        bb_analysis = self.bb_calc.analyze(data)
        sr_analysis = self.sr_calc.analyze(data)
        ma_values = self.ma_calc.get_ma_values(data)

        summary = {
            "price": float(data["close"].iloc[-1]),
            "moving_averages": ma_values,
            "rsi": {
                "value": rsi_result.value,
                "zone": rsi_result.zone.value,
            },
            "macd": {
                "macd": macd_analysis.macd_value,
                "signal": macd_analysis.signal_value,
                "histogram": macd_analysis.histogram_value,
                "trend": macd_analysis.trend,
            },
            "bollinger": {
                "upper": bb_analysis.upper_band,
                "middle": bb_analysis.middle_band,
                "lower": bb_analysis.lower_band,
                "percent_b": bb_analysis.percent_b,
            },
            "support_resistance": {
                "nearest_support": sr_analysis.nearest_support,
                "nearest_resistance": sr_analysis.nearest_resistance,
                "pivot": sr_analysis.pivot_points.pivot,
            },
        }

        try:
            atr_result = self.atr_calc.analyze(data)
            summary["atr"] = {
                "value": round(atr_result.current_atr, 2),
                "percent": round(atr_result.atr_percent, 2),
                "volatility_regime": atr_result.volatility_regime.value,
            }
        except Exception:
            pass

        try:
            adx_analysis = self.adx_calc.analyze(data)
            summary["adx"] = {
                "value": round(adx_analysis.adx_value, 1),
                "trend_direction": adx_analysis.trend_direction,
                "regime": adx_analysis.regime.value,
            }
        except Exception:
            pass

        try:
            pattern_analysis = self.pattern_detector.analyze(data)
            summary["candlestick_patterns"] = {
                "patterns": [p.name for p in pattern_analysis.patterns_found],
                "net_score": round(pattern_analysis.net_score, 2),
                "dominant": pattern_analysis.dominant_signal,
            }
        except Exception:
            pass

        return summary
