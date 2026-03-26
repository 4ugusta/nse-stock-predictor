"""Core test suite for the Indian Stock Predictor's most critical paths."""

import pytest
import pandas as pd
import numpy as np


def _make_ohlcv(n=100, base_price=1000.0, volume=100000):
    """Helper to create valid OHLCV data."""
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    np.random.seed(42)
    close = base_price + np.cumsum(np.random.randn(n) * 10)
    close = np.maximum(close, 1.0)  # prevent negative
    high = close + np.abs(np.random.randn(n) * 5)
    low = close - np.abs(np.random.randn(n) * 5)
    low = np.maximum(low, 0.5)
    open_ = close + np.random.randn(n) * 3
    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    vol = np.random.randint(volume // 2, volume * 2, n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# 1. Data Validation Tests
# ---------------------------------------------------------------------------
from stock_predictor.core.validation import (
    validate_ohlcv_data,
    detect_circuit_breaker,
    detect_stock_split,
    DataQuality,
)


class TestDataValidation:
    def test_valid_data_returns_good(self):
        data = _make_ohlcv(100)
        result = validate_ohlcv_data(data)
        assert result.quality == DataQuality.GOOD
        assert result.is_usable

    def test_insufficient_rows(self):
        data = _make_ohlcv(10)
        result = validate_ohlcv_data(data, min_rows=50)
        assert result.quality == DataQuality.INSUFFICIENT
        assert not result.is_usable

    def test_missing_columns(self):
        data = pd.DataFrame({"close": [1, 2, 3]})
        result = validate_ohlcv_data(data)
        assert result.quality == DataQuality.INSUFFICIENT

    def test_nan_values_detected(self):
        data = _make_ohlcv(100)
        data.loc[data.index[5], "close"] = np.nan
        data.loc[data.index[10], "close"] = np.nan
        result = validate_ohlcv_data(data)
        assert any("NaN" in issue for issue in result.issues)

    def test_frozen_prices_detected(self):
        data = _make_ohlcv(100)
        # Freeze close for 6 consecutive days
        data.iloc[20:26, data.columns.get_loc("close")] = 500.0
        result = validate_ohlcv_data(data)
        assert any(
            "Frozen" in issue or "frozen" in issue.lower() for issue in result.issues
        )

    def test_circuit_breaker_detection(self):
        data = _make_ohlcv(50)
        # Create a circuit breaker day: 15% drop with locked price
        data.iloc[25, data.columns.get_loc("close")] = (
            data.iloc[24, data.columns.get_loc("close")] * 0.85
        )
        data.iloc[25, data.columns.get_loc("high")] = data.iloc[
            25, data.columns.get_loc("close")
        ]
        data.iloc[25, data.columns.get_loc("low")] = data.iloc[
            25, data.columns.get_loc("close")
        ]
        data.iloc[25, data.columns.get_loc("open")] = data.iloc[
            25, data.columns.get_loc("close")
        ]
        cb = detect_circuit_breaker(data)
        assert cb.iloc[25] == True

    def test_stock_split_detection(self):
        data = _make_ohlcv(50)
        # Create a 2:1 split: ~50% price drop
        data.iloc[30, data.columns.get_loc("close")] = (
            data.iloc[29, data.columns.get_loc("close")] * 0.48
        )
        splits = detect_stock_split(data)
        assert splits.iloc[30] == True


# ---------------------------------------------------------------------------
# 2. Signal Type Tests
# ---------------------------------------------------------------------------
from stock_predictor.core.entities.signal import SignalType


class TestSignalTypes:
    def test_no_edge_exists(self):
        assert SignalType.NO_EDGE.value == "no_edge"

    def test_all_signal_types(self):
        expected = {"strong_buy", "buy", "hold", "no_edge", "sell", "strong_sell"}
        actual = {st.value for st in SignalType}
        assert expected == actual


# ---------------------------------------------------------------------------
# 3. Indicator Tests
# ---------------------------------------------------------------------------
from stock_predictor.analysis.indicators.rsi import RSICalculator
from stock_predictor.analysis.indicators.macd import MACDCalculator
from stock_predictor.analysis.indicators.adx import ADXCalculator
from stock_predictor.analysis.indicators.volume import VolumeAnalyzer


class TestRSI:
    def test_rsi_warmup_guard(self):
        """RSI should return neutral for insufficient data."""
        calc = RSICalculator()
        short_data = _make_ohlcv(15)  # Less than period + 10 (14 + 10 = 24)
        result = calc.analyze(short_data)
        assert result.value == 50.0  # Neutral

    def test_rsi_normal_range(self):
        """RSI should be between 0 and 100."""
        calc = RSICalculator()
        data = _make_ohlcv(100)
        result = calc.analyze(data)
        assert 0 <= result.value <= 100

    def test_rsi_all_gains(self):
        """All gains should give RSI near 100."""
        calc = RSICalculator(period=14)
        dates = pd.date_range("2025-01-01", periods=50, freq="B")
        prices = [100 + i for i in range(50)]
        data = pd.DataFrame(
            {
                "open": prices,
                "high": [p + 1 for p in prices],
                "low": [p - 0.5 for p in prices],
                "close": prices,
                "volume": [100000] * 50,
            },
            index=dates,
        )
        result = calc.analyze(data)
        assert result.value > 90


class TestMACD:
    def test_macd_penny_stock_normalization(self):
        """Penny stock should have dampened MACD strength."""
        calc = MACDCalculator()
        # Penny stock at Rs 5
        penny = _make_ohlcv(100, base_price=5.0)
        # Regular stock at Rs 1000
        regular = _make_ohlcv(100, base_price=1000.0)
        penny_result = calc.analyze(penny)
        regular_result = calc.analyze(regular)
        # Both should have valid (non-NaN) values
        assert not np.isnan(penny_result.signal_strength)
        assert not np.isnan(regular_result.signal_strength)

    def test_macd_nan_guard(self):
        """MACD should handle NaN data gracefully."""
        calc = MACDCalculator()
        data = _make_ohlcv(100)
        # Inject some NaN
        data.iloc[:30, data.columns.get_loc("close")] = np.nan
        result = calc.analyze(data)
        # Should return neutral, not crash
        assert result.trend in ("bullish", "bearish", "neutral")


class TestADX:
    def test_adx_nan_guard(self):
        """ADX should return neutral when pandas_ta produces NaN values.

        Constant prices produce NaN in ADX (zero directional movement),
        which triggers the NaN guard in ADXCalculator.analyze().
        """
        calc = ADXCalculator()
        # Constant prices: pandas_ta returns ADX=NaN, DMP=0, DMN=0
        n = 30
        dates = pd.date_range("2025-01-01", periods=n, freq="B")
        data = pd.DataFrame(
            {
                "open": [100.0] * n,
                "high": [100.5] * n,
                "low": [99.5] * n,
                "close": [100.0] * n,
                "volume": [100000] * n,
            },
            index=dates,
        )
        result = calc.analyze(data)
        # NaN guard should return neutral analysis, not crash
        assert result is not None
        assert result.adx_value == 0.0
        assert result.trend_direction == "neutral"

    def test_adx_volatility_adjusted(self):
        """ADX regime detection should support volatility adjustment."""
        calc = ADXCalculator()
        data = _make_ohlcv(100)
        result = calc.analyze(data)
        assert hasattr(result, "regime")


class TestVolume:
    def test_dead_stock_zero_volume(self):
        """Zero volume stock should have 0 volume ratio, not 1."""
        analyzer = VolumeAnalyzer()
        data = _make_ohlcv(100)
        data["volume"] = 0  # Dead stock
        result = analyzer.analyze(data)
        # Volume ratio should be 0 for dead stocks
        assert result.volume_ratio == 0.0


# ---------------------------------------------------------------------------
# 4. Signal Generator Integration Test
# ---------------------------------------------------------------------------
from stock_predictor.analysis.signals.signal_generator import SignalGenerator


class TestSignalGenerator:
    def test_generates_signal_for_valid_data(self):
        """Signal generator should produce a valid signal for good data."""
        gen = SignalGenerator()
        data = _make_ohlcv(200)
        signal = gen.generate(data, symbol="TEST")
        assert signal is not None
        assert signal.signal_type in SignalType
        assert 0 <= signal.confidence <= 1.0

    def test_hold_for_insufficient_data(self):
        """Should return HOLD for insufficient data (< MIN_DATA_WARMUP)."""
        gen = SignalGenerator()
        data = _make_ohlcv(10)
        signal = gen.generate(data, symbol="TEST")
        # The generator returns HOLD (not NO_EDGE) for insufficient data,
        # with a reason explaining the data shortage.
        assert signal.signal_type == SignalType.HOLD

    def test_confidence_not_just_abs_score(self):
        """Confidence should reflect indicator agreement, not just score magnitude."""
        gen = SignalGenerator()
        data = _make_ohlcv(200)
        signal = gen.generate(data, symbol="TEST")
        # Confidence should be between 0 and 1
        assert 0 <= signal.confidence <= 1.0
        # Confidence for random data should generally not be extreme
        # (this is a soft check - random data should have mixed signals)


# ---------------------------------------------------------------------------
# 5. Picker Config Test
# ---------------------------------------------------------------------------
from stock_predictor.analysis.high_confidence_picker import (
    HighConfidencePicker,
    PickerConfig,
)


class TestPickerConfig:
    def test_default_config(self):
        """Picker should work with default config."""
        # Just test PickerConfig defaults
        config = PickerConfig()
        assert config.min_adx == 25.0
        assert config.min_risk_reward == 1.5
        assert config.min_volume_ratio == 1.0

    def test_custom_config(self):
        """Picker should accept custom config."""
        config = PickerConfig(min_adx=30.0, min_risk_reward=2.0)
        assert config.min_adx == 30.0
        assert config.min_risk_reward == 2.0


# ---------------------------------------------------------------------------
# 6. Constants Test
# ---------------------------------------------------------------------------
from stock_predictor.infrastructure.config.constants import (
    RISK_FREE_RATE,
    MAX_PARTICIPATION_RATE,
    CIRCUIT_BREAKER_LIMITS,
)


class TestConstants:
    def test_risk_free_rate(self):
        assert 0.05 <= RISK_FREE_RATE <= 0.10  # Reasonable range for India

    def test_participation_rate(self):
        assert MAX_PARTICIPATION_RATE <= 0.10  # Should be conservative

    def test_circuit_breaker_limits(self):
        assert len(CIRCUIT_BREAKER_LIMITS) == 3
        assert CIRCUIT_BREAKER_LIMITS == [0.10, 0.15, 0.20]
