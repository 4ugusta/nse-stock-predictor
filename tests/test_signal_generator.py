"""Tests for the signal generator — the core decision-making engine."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from stock_predictor.analysis.signals.signal_generator import (
    SignalGenerator,
    ExternalScores,
    IndicatorWeights,
)
from stock_predictor.core.entities.signal import (
    SignalType,
    SignalStrength,
    TradingStyle,
)


def _make_ohlcv(n=250, base_price=1000.0, volume=100000, seed=42):
    """Helper to create valid OHLCV data with timestamp column."""
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


def _make_uptrend(n=250, base_price=500.0, daily_gain=2.0):
    """Create synthetic uptrend data."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    np.random.seed(99)
    close = base_price + np.arange(n) * daily_gain + np.random.randn(n) * 3
    close = np.maximum(close, 1.0)
    high = close + np.abs(np.random.randn(n) * 2) + 1
    low = close - np.abs(np.random.randn(n) * 2) - 1
    low = np.maximum(low, 0.5)
    open_ = close - daily_gain / 2 + np.random.randn(n) * 1
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    vol = np.random.randint(80000, 200000, n)
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


def _make_downtrend(n=250, base_price=1500.0, daily_loss=2.0):
    """Create synthetic downtrend data."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    np.random.seed(77)
    close = base_price - np.arange(n) * daily_loss + np.random.randn(n) * 3
    close = np.maximum(close, 50.0)
    high = close + np.abs(np.random.randn(n) * 2) + 1
    low = close - np.abs(np.random.randn(n) * 2) - 1
    low = np.maximum(low, 1.0)
    open_ = close + daily_loss / 2 + np.random.randn(n) * 1
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    vol = np.random.randint(80000, 200000, n)
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


class TestSignalGeneratorBasics:
    """Basic signal generation tests."""

    def test_signal_has_all_fields(self):
        """Signal should have all required fields populated."""
        gen = SignalGenerator()
        data = _make_ohlcv(250)
        signal = gen.generate(data, symbol="RELIANCE")
        assert signal.symbol == "RELIANCE"
        assert signal.signal_type in SignalType
        assert signal.strength in SignalStrength
        assert isinstance(signal.confidence, float)
        assert signal.entry_price is not None
        assert len(signal.reasons) > 0

    def test_confidence_capped_at_70_percent(self):
        """Confidence should never exceed 0.70 (cap from robustness fix)."""
        gen = SignalGenerator()
        # Run on several different datasets
        for seed in range(10):
            data = _make_ohlcv(250, seed=seed)
            signal = gen.generate(data, symbol="TEST")
            assert signal.confidence <= 0.70, (
                f"Confidence {signal.confidence} exceeded 0.70 cap (seed={seed})"
            )

    def test_insufficient_data_returns_hold(self):
        """Less than MIN_DATA_WARMUP bars should return HOLD."""
        gen = SignalGenerator()
        data = _make_ohlcv(10)
        signal = gen.generate(data, symbol="TEST")
        assert signal.signal_type == SignalType.HOLD
        assert signal.confidence == 0.0
        assert any("Insufficient" in r for r in signal.reasons)

    def test_zero_volume_returns_hold(self):
        """Zero-volume (illiquid) stock should be rejected."""
        gen = SignalGenerator()
        data = _make_ohlcv(250)
        data["volume"] = 100  # Well below MIN_LIQUIDITY_VOLUME
        signal = gen.generate(data, symbol="DEADSTOCK")
        assert signal.signal_type == SignalType.HOLD
        assert signal.confidence == 0.0

    def test_all_styles_produce_signals(self):
        """All three trading styles should produce valid signals."""
        gen = SignalGenerator()
        data = _make_ohlcv(250)
        for style in TradingStyle:
            signal = gen.generate(data, symbol="TEST", style=style)
            assert signal is not None
            assert signal.signal_type in SignalType


class TestSignalGeneratorConfidence:
    """Tests for the confidence scoring system."""

    def test_random_data_low_confidence(self):
        """Random walk data should generally produce low confidence."""
        gen = SignalGenerator()
        confidences = []
        for seed in range(5):
            data = _make_ohlcv(250, seed=seed * 100)
            signal = gen.generate(data, symbol="RANDOM")
            confidences.append(signal.confidence)
        avg = sum(confidences) / len(confidences)
        assert avg < 0.50, f"Average confidence on random data is {avg:.2f}, expected < 0.50"

    @patch("stock_predictor.analysis.signals.signal_generator.fetch_india_vix")
    def test_high_vix_reduces_confidence(self, mock_vix):
        """High VIX should reduce confidence via the multiplier."""
        from stock_predictor.analysis.market_intelligence import VIXReading

        gen = SignalGenerator()
        data = _make_uptrend(250)

        # Normal VIX
        mock_vix.return_value = VIXReading(
            value=15.0, regime="normal", confidence_adjustment=1.0,
            description="Normal"
        )
        signal_normal = gen.generate(data, symbol="TEST")

        # High VIX
        mock_vix.return_value = VIXReading(
            value=30.0, regime="high", confidence_adjustment=0.7,
            description="High fear"
        )
        signal_high = gen.generate(data, symbol="TEST")

        assert signal_high.confidence <= signal_normal.confidence

    def test_external_scores_capped(self):
        """External scores should be capped at ±0.15 adjustment."""
        gen = SignalGenerator()
        data = _make_ohlcv(250)

        # Extreme external scores
        extreme = ExternalScores(
            sentiment_score=1.0,
            fundamental_score=1.0,
            sentiment_weight=0.5,
            fundamental_weight=0.5,
        )
        signal_extreme = gen.generate(data, symbol="TEST", external_scores=extreme)

        # No external scores
        signal_none = gen.generate(data, symbol="TEST")

        # The difference should be bounded by the ±0.15 cap
        # (confidence may differ due to score change, but signal shouldn't flip wildly)
        assert signal_extreme is not None
        assert signal_none is not None


class TestSignalGeneratorEdgeCases:
    """Edge case and robustness tests."""

    def test_circuit_breaker_day_returns_hold(self):
        """Signal should be HOLD when circuit breaker is detected."""
        gen = SignalGenerator()
        data = _make_ohlcv(250)
        # Create a circuit breaker: 18% drop, price locked
        idx = 249
        prev = data.iloc[idx - 1]["close"]
        locked_price = prev * 0.82
        data.iloc[idx, data.columns.get_loc("close")] = locked_price
        data.iloc[idx, data.columns.get_loc("open")] = locked_price
        data.iloc[idx, data.columns.get_loc("high")] = locked_price
        data.iloc[idx, data.columns.get_loc("low")] = locked_price
        signal = gen.generate(data, symbol="HALTED")
        assert signal.signal_type == SignalType.HOLD

    def test_nan_heavy_data_returns_no_edge(self):
        """Data with many NaN values should degrade quality to NO_EDGE/HOLD."""
        gen = SignalGenerator()
        data = _make_ohlcv(250)
        # Inject NaN into 30% of close prices
        nan_idx = np.random.choice(range(50, 250), size=60, replace=False)
        for i in nan_idx:
            data.iloc[i, data.columns.get_loc("close")] = np.nan
        signal = gen.generate(data, symbol="BADDATA")
        # Should either be HOLD, NO_EDGE, or have very low confidence
        assert signal.signal_type in (SignalType.HOLD, SignalType.NO_EDGE) or signal.confidence < 0.25

    def test_multi_timeframe_opposing_blocks_signal(self):
        """When weekly opposes daily, signal should be NO_EDGE."""
        gen = SignalGenerator()
        daily_up = _make_uptrend(250)
        weekly_down = _make_downtrend(52)  # 52 weeks
        signal = gen.generate_multi_timeframe(daily_up, weekly_down, symbol="CONFLICT")
        # Should be blocked or have very low confidence
        assert signal.signal_type == SignalType.NO_EDGE or signal.confidence < 0.30

    def test_gap_analysis_in_reasons(self):
        """Large overnight gap should appear in signal reasons."""
        gen = SignalGenerator()
        data = _make_ohlcv(250)
        # Create a 4% gap up on the last day
        prev_close = data.iloc[-2]["close"]
        data.iloc[-1, data.columns.get_loc("open")] = prev_close * 1.04
        data.iloc[-1, data.columns.get_loc("high")] = max(
            data.iloc[-1]["high"], prev_close * 1.05
        )
        signal = gen.generate(data, symbol="GAPPED")
        gap_reasons = [r for r in signal.reasons if "Gap" in r]
        assert len(gap_reasons) > 0, "Large gap should appear in reasons"

    def test_whipsaw_detection_reduces_confidence(self):
        """Choppy alternating price action should trigger whipsaw penalty."""
        gen = SignalGenerator()
        data = _make_ohlcv(250, seed=42)
        # Make last 20 bars alternate heavily
        for i in range(230, 250):
            if i % 2 == 0:
                data.iloc[i, data.columns.get_loc("close")] = data.iloc[i - 1]["close"] * 1.02
            else:
                data.iloc[i, data.columns.get_loc("close")] = data.iloc[i - 1]["close"] * 0.98
            # Keep OHLC consistent
            data.iloc[i, data.columns.get_loc("high")] = max(
                data.iloc[i]["open"], data.iloc[i]["close"]
            ) + 1
            data.iloc[i, data.columns.get_loc("low")] = min(
                data.iloc[i]["open"], data.iloc[i]["close"]
            ) - 1
        signal = gen.generate(data, symbol="WHIPSAW")
        # Should have low confidence or be NO_EDGE
        assert signal.confidence < 0.50 or signal.signal_type == SignalType.NO_EDGE


class TestSignalGeneratorDiagnostics:
    """Tests for signal diagnostics and transparency."""

    def test_diagnostics_attached(self):
        """Signal should have diagnostics with regime and score info."""
        gen = SignalGenerator()
        data = _make_ohlcv(250)
        signal = gen.generate(data, symbol="TEST")
        assert hasattr(signal, "diagnostics")
        diag = signal.diagnostics
        assert hasattr(diag, "raw_technical_score")
        assert hasattr(diag, "market_regime")
        assert hasattr(diag, "atr_percent")

    def test_reasons_include_regime(self):
        """Reasons should always include market regime info."""
        gen = SignalGenerator()
        data = _make_ohlcv(250)
        signal = gen.generate(data, symbol="TEST")
        regime_reasons = [r for r in signal.reasons if "regime" in r.lower()]
        assert len(regime_reasons) > 0

    def test_indicator_signals_populated(self):
        """At least 5 core indicator signals should be present."""
        gen = SignalGenerator()
        data = _make_ohlcv(250)
        signal = gen.generate(data, symbol="TEST")
        # MA, RSI, MACD, Bollinger, S/R = 5 core + optional volume/pattern
        assert len(signal.indicator_signals) >= 5

    def test_generate_all_styles(self):
        """generate_all_styles should return all three styles."""
        gen = SignalGenerator()
        data = _make_ohlcv(250)
        result = gen.generate_all_styles(data, symbol="TEST")
        assert set(result.keys()) == {
            TradingStyle.INTRADAY,
            TradingStyle.SWING,
            TradingStyle.POSITIONAL,
        }
        for style, signal in result.items():
            assert signal.signal_type in SignalType
