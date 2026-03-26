"""Tests for the high confidence stock picker."""

import pytest
import pandas as pd
import numpy as np
from decimal import Decimal
from unittest.mock import patch, MagicMock

from stock_predictor.analysis.high_confidence_picker import (
    HighConfidencePicker,
    PickerConfig,
    ConfidenceLevel,
    FilterResult,
)
from stock_predictor.analysis.indicators.adx import MarketRegime


def _make_ohlcv(n=250, base_price=1000.0, volume=100000, seed=42):
    """Helper to create valid OHLCV data."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    np.random.seed(seed)
    close = base_price + np.cumsum(np.random.randn(n) * 10)
    close = np.maximum(close, 1.0)
    high = close + np.abs(np.random.randn(n) * 5)
    low = close - np.abs(np.random.randn(n) * 5)
    low = np.maximum(low, 0.5)
    open_ = close + np.random.randn(n) * 3
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
            "timestamp": dates,
        },
        index=dates,
    )


def _make_strong_uptrend(n=250, base_price=500.0):
    """Create a clear uptrend for picker to find."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    np.random.seed(55)
    close = base_price + np.arange(n) * 3.0 + np.random.randn(n) * 2
    close = np.maximum(close, 50.0)
    high = close + np.abs(np.random.randn(n)) + 2
    low = close - np.abs(np.random.randn(n)) - 1
    low = np.maximum(low, 1.0)
    open_ = close - 1.5 + np.random.randn(n) * 0.5
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    vol = np.random.randint(120000, 250000, n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
            "timestamp": dates,
        },
        index=dates,
    )


class TestPickerConfig:
    """Tests for picker configuration."""

    def test_default_config_values(self):
        config = PickerConfig()
        assert config.min_adx == 25.0
        assert config.min_rsi_long == 40.0
        assert config.max_rsi_long == 70.0
        assert config.min_rsi_short == 40.0
        assert config.max_rsi_short == 70.0
        assert config.min_volume_ratio == 1.0
        assert config.min_risk_reward == 1.5

    def test_custom_config(self):
        config = PickerConfig(min_adx=30.0, min_risk_reward=2.5)
        assert config.min_adx == 30.0
        assert config.min_risk_reward == 2.5

    def test_short_rsi_range_wider_than_before(self):
        """Short RSI range should be 40-70 (not the old 30-60)."""
        config = PickerConfig()
        assert config.min_rsi_short == 40.0
        assert config.max_rsi_short == 70.0


class TestPickerFilters:
    """Tests for individual filter checks."""

    def test_trend_filter_downtrend_fails(self):
        """Downtrend data should fail the trend filter."""
        picker = HighConfidencePicker(config=PickerConfig())
        dates = pd.date_range("2024-01-01", periods=250, freq="B")
        np.random.seed(33)
        close = 1500 - np.arange(250) * 3.0 + np.random.randn(250) * 2
        close = np.maximum(close, 50.0)
        data = pd.DataFrame({
            "open": close + 1, "high": close + 3,
            "low": close - 2, "close": close,
            "volume": np.random.randint(80000, 200000, 250),
        }, index=dates)
        result = picker._check_trend_filter(data)
        # Should fail for downtrend
        assert not result.passed or "downtrend" in result.reason.lower()

    def test_volume_filter_low_volume_fails(self):
        """Very low volume should fail the volume filter."""
        picker = HighConfidencePicker(config=PickerConfig(min_volume_ratio=1.0))
        data = _make_ohlcv(250)
        # Make last bar have very low volume compared to average
        data.iloc[-1, data.columns.get_loc("volume")] = 100
        result = picker._check_volume_filter(data)
        assert not result.passed

    def test_rsi_filter_overbought_fails(self):
        """RSI > 70 should fail the RSI filter for longs."""
        picker = HighConfidencePicker(config=PickerConfig())
        # Create data where all prices go up (RSI near 100)
        dates = pd.date_range("2024-01-01", periods=250, freq="B")
        close = np.array([100 + i * 0.5 for i in range(250)])
        data = pd.DataFrame({
            "open": close - 0.2, "high": close + 0.3,
            "low": close - 0.3, "close": close,
            "volume": [100000] * 250,
        }, index=dates)
        result = picker._check_rsi_filter(data)
        assert not result.passed


class TestPickerAnalysis:
    """Tests for the full stock analysis pipeline."""

    def _make_picker_with_mock_data(self, data):
        """Create a picker with mocked data provider."""
        picker = HighConfidencePicker()
        picker.data_provider = MagicMock()
        picker.data_provider.get_historical.return_value = data
        return picker

    def test_insufficient_data_returns_none(self):
        """Should return None for stocks with < 200 bars of data."""
        picker = self._make_picker_with_mock_data(_make_ohlcv(100))
        result = picker.analyze_stock("SMALLDATA")
        assert result is None

    def test_trend_failure_forces_avoid(self):
        """If trend filter fails, result should be AVOID (early exit)."""
        dates = pd.date_range("2024-01-01", periods=250, freq="B")
        np.random.seed(33)
        # Strong downtrend
        close = 1500 - np.arange(250) * 4.0 + np.random.randn(250) * 1
        close = np.maximum(close, 50.0)
        data = pd.DataFrame({
            "open": close + 2, "high": close + 4,
            "low": close - 2, "close": close,
            "volume": np.random.randint(80000, 200000, 250),
        }, index=dates)
        picker = self._make_picker_with_mock_data(data)
        result = picker.analyze_stock("DOWNTREND")
        if result is not None:
            assert result.confidence_level == ConfidenceLevel.AVOID

    def test_targets_capped_at_50_percent(self):
        """Targets should never exceed ±50% from entry."""
        picker = self._make_picker_with_mock_data(_make_strong_uptrend(250))
        result = picker.analyze_stock("TESTSTOCK")
        if result is not None:
            entry = float(result.entry_price)
            if result.target_1 is not None:
                assert float(result.target_1) <= entry * 1.50 + 0.01
            if result.target_2 is not None:
                assert float(result.target_2) <= entry * 1.50 + 0.01
            if result.stop_loss is not None:
                assert float(result.stop_loss) >= entry * 0.50 - 0.01


class TestPickerLevels:
    """Tests for ATR-based level calculations."""

    def test_long_levels_ordering(self):
        """For longs: stop < entry < target."""
        picker = HighConfidencePicker()
        data = _make_strong_uptrend(250)
        price = float(data["close"].iloc[-1])
        entry, stop, t1, t2 = picker._calculate_levels(data, price, is_long=True)
        assert stop < entry
        assert t1 > entry or t2 > entry

    def test_short_levels_ordering(self):
        """For shorts: entry < stop, target < entry."""
        picker = HighConfidencePicker()
        data = _make_ohlcv(250)
        price = float(data["close"].iloc[-1])
        entry, stop, t1, t2 = picker._calculate_levels(data, price, is_long=False)
        assert stop > entry
        assert t1 < entry or t2 < entry

    def test_levels_max_distance(self):
        """All levels should be within ±50% of entry."""
        picker = HighConfidencePicker()
        data = _make_ohlcv(250)
        price = float(data["close"].iloc[-1])
        for is_long in [True, False]:
            entry, stop, t1, t2 = picker._calculate_levels(data, price, is_long=is_long)
            entry_f = float(entry)
            for level in [stop, t1, t2]:
                level_f = float(level)
                distance = abs(level_f - entry_f) / entry_f
                assert distance <= 0.51, f"Level {level_f} is {distance:.0%} from entry {entry_f}"
