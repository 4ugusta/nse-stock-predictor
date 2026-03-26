"""Technical indicator calculations."""

from stock_predictor.analysis.indicators.moving_averages import MovingAverageCalculator
from stock_predictor.analysis.indicators.rsi import RSICalculator
from stock_predictor.analysis.indicators.macd import MACDCalculator
from stock_predictor.analysis.indicators.bollinger import BollingerBandsCalculator
from stock_predictor.analysis.indicators.support_resistance import SupportResistanceCalculator
from stock_predictor.analysis.indicators.adx import ADXCalculator, TrendStrength, MarketRegime
from stock_predictor.analysis.indicators.volume import VolumeAnalyzer, VolumeSignal
from stock_predictor.analysis.indicators.atr import ATRCalculator, VolatilityRegime
from stock_predictor.analysis.indicators.candlestick_patterns import (
    CandlestickPatternDetector,
    PatternType,
)

__all__ = [
    "MovingAverageCalculator",
    "RSICalculator",
    "MACDCalculator",
    "BollingerBandsCalculator",
    "SupportResistanceCalculator",
    "ADXCalculator",
    "TrendStrength",
    "MarketRegime",
    "VolumeAnalyzer",
    "VolumeSignal",
    "ATRCalculator",
    "VolatilityRegime",
    "CandlestickPatternDetector",
    "PatternType",
]
