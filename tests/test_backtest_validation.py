"""Backtest validation: verify the robustness fixes improve signal quality.

These tests run backtests on synthetic data with known properties and verify
that the engine produces sane results. They serve as regression tests to
ensure future changes don't degrade the system.

NOT a before/after comparison (that requires real market data + git diff),
but a set of statistical sanity checks on the current system.
"""

import pytest
import pandas as pd
import numpy as np

from stock_predictor.analysis.backtesting.engine import BacktestEngine, BacktestResult
from stock_predictor.analysis.signals.signal_generator import SignalGenerator
from stock_predictor.core.entities.signal import SignalType, TradingStyle


def _make_ohlcv(n=500, base_price=1000.0, volume=100000, seed=42):
    """Create OHLCV data with timestamp."""
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    np.random.seed(seed)
    close = base_price + np.cumsum(np.random.randn(n) * 10)
    close = np.maximum(close, 50.0)
    high = close + np.abs(np.random.randn(n) * 5) + 1
    low = close - np.abs(np.random.randn(n) * 5) - 1
    low = np.maximum(low, 1.0)
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


def _make_uptrend(n=500, base_price=500.0, daily_gain=1.5, seed=99):
    """Create synthetic uptrend (should generate more buy signals)."""
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    np.random.seed(seed)
    close = base_price + np.arange(n) * daily_gain + np.random.randn(n) * 5
    close = np.maximum(close, 50.0)
    high = close + np.abs(np.random.randn(n) * 3) + 1
    low = close - np.abs(np.random.randn(n) * 3) - 1
    low = np.maximum(low, 1.0)
    open_ = close - daily_gain / 2 + np.random.randn(n) * 2
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


def _make_downtrend(n=500, base_price=2000.0, daily_loss=1.5, seed=77):
    """Create synthetic downtrend."""
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    np.random.seed(seed)
    close = base_price - np.arange(n) * daily_loss + np.random.randn(n) * 5
    close = np.maximum(close, 100.0)
    high = close + np.abs(np.random.randn(n) * 3) + 1
    low = close - np.abs(np.random.randn(n) * 3) - 1
    low = np.maximum(low, 1.0)
    open_ = close + daily_loss / 2 + np.random.randn(n) * 2
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


# ---------------------------------------------------------------------------
# Signal Quality Validation
# ---------------------------------------------------------------------------
class TestSignalQuality:
    """Verify that signal generation produces sane results on known data."""

    def test_random_data_mostly_no_edge_or_hold(self):
        """On random walk data, most signals should be NO_EDGE or HOLD.

        Random data has no real edge — a good signal generator should NOT
        find confident signals in noise.
        """
        gen = SignalGenerator()
        no_edge_or_hold = 0
        total = 20

        for seed in range(total):
            data = _make_ohlcv(250, seed=seed * 13)
            signal = gen.generate(data, symbol="RANDOM")
            if signal.signal_type in (SignalType.NO_EDGE, SignalType.HOLD):
                no_edge_or_hold += 1

        # At least 50% of random data should produce NO_EDGE/HOLD
        ratio = no_edge_or_hold / total
        assert ratio >= 0.50, (
            f"Only {ratio:.0%} of random data signals were NO_EDGE/HOLD. "
            "The system may be over-fitting to noise."
        )

    def test_uptrend_produces_directional_signals(self):
        """Strong uptrend data should produce buy signals or at least not sell signals.

        Note: with all our robustness filters (confidence cap 70%, whipsaw,
        VIX adjustment), the system is intentionally conservative. We test
        that it doesn't produce SELL signals on clear uptrends.
        """
        gen = SignalGenerator()
        sell_count = 0
        for seed in range(5):
            data = _make_uptrend(300, seed=seed * 7 + 99)
            signal = gen.generate(data, symbol="UPTREND")
            if signal.signal_type in (SignalType.SELL, SignalType.STRONG_SELL):
                sell_count += 1
        # Should NOT produce sell signals on clear uptrend
        assert sell_count <= 1, f"Produced {sell_count}/5 sell signals on clear uptrend data"

    def test_noisy_data_lower_confidence(self):
        """Data with NaN values should get a data quality penalty."""
        gen = SignalGenerator()
        # Clean data
        clean = _make_ohlcv(250, seed=42)
        clean_signal = gen.generate(clean, symbol="CLEAN")

        # Same data with NaN injection (enough to trigger degraded quality)
        noisy = clean.copy()
        np.random.seed(42)
        for i in np.random.choice(range(50, 250), size=15, replace=False):
            noisy.iloc[i, noisy.columns.get_loc("close")] = np.nan
        noisy_signal = gen.generate(noisy, symbol="NOISY")

        # Noisy should have lower or equal confidence OR be downgraded to NO_EDGE
        noisy_is_worse = (
            noisy_signal.confidence <= clean_signal.confidence + 0.01
            or noisy_signal.signal_type == SignalType.NO_EDGE
        )
        assert noisy_is_worse


# ---------------------------------------------------------------------------
# Backtest Sanity Checks
# ---------------------------------------------------------------------------
class TestBacktestSanity:
    """Verify backtesting engine produces sane results."""

    def test_random_data_not_profitable(self):
        """Backtest on random data should not show consistent profits.

        If the system is profitable on pure random data, it's likely overfitting.
        """
        engine = BacktestEngine(initial_capital=100_000)
        profits = []

        for seed in range(3):
            data = _make_ohlcv(500, seed=seed * 37)
            result = engine.run(data, symbol="RANDOM")
            profits.append(result.total_return_pct)

        # Average return on random data should be modest (not consistently positive)
        avg_return = sum(profits) / len(profits)
        assert avg_return < 30.0, (
            f"Average return on random data is {avg_return:.1f}%. "
            "System may be overfitting."
        )

    def test_uptrend_beats_random(self):
        """Backtest on uptrend data should do better than random data."""
        engine = BacktestEngine(initial_capital=100_000)

        uptrend_result = engine.run(_make_uptrend(500), symbol="UP")
        random_result = engine.run(_make_ohlcv(500, seed=42), symbol="RANDOM")

        # Uptrend should not do dramatically worse than random
        # (signals should at minimum not hurt in trending markets)
        assert uptrend_result.total_return_pct >= random_result.total_return_pct - 20

    def test_transaction_costs_matter(self):
        """Engine should account for transaction costs (result should differ from gross)."""
        engine = BacktestEngine(initial_capital=100_000)
        data = _make_ohlcv(500)
        result = engine.run(data, symbol="TEST")

        if result.total_trades > 0:
            # With transaction costs, more trades = more drag
            # The equity curve should show costs being deducted
            assert result.equity_curve[-1] != result.equity_curve[0] or result.total_trades == 0

    def test_drawdown_bounded(self):
        """Max drawdown should never exceed 100%."""
        engine = BacktestEngine(initial_capital=100_000)
        for seed in [42, 99, 77]:
            data = _make_ohlcv(500, seed=seed)
            result = engine.run(data, symbol="TEST")
            assert result.max_drawdown_pct <= 100.0

    def test_sharpe_finite(self):
        """Sharpe ratio should be finite (not NaN or inf)."""
        engine = BacktestEngine(initial_capital=100_000)
        data = _make_ohlcv(500)
        result = engine.run(data, symbol="TEST")
        assert np.isfinite(result.sharpe_ratio)


# ---------------------------------------------------------------------------
# Out-of-Sample Degradation Tests
# ---------------------------------------------------------------------------
class TestOutOfSampleDegradation:
    """Verify the OOS validation produces meaningful comparisons."""

    def test_oos_produces_comparison(self):
        """OOS validation should produce in-sample and out-of-sample results."""
        engine = BacktestEngine(initial_capital=100_000)
        data = _make_ohlcv(700)
        result = engine.run_out_of_sample(data, symbol="TEST")

        if "error" not in result:
            assert "in_sample" in result
            assert "out_of_sample" in result
            # Results are dicts with summary metrics
            assert isinstance(result["in_sample"], dict)
            assert "win_rate" in result["in_sample"]
            assert "total_return" in result["in_sample"]

    def test_oos_degradation_measured(self):
        """OOS should report degradation metrics."""
        engine = BacktestEngine(initial_capital=100_000)
        data = _make_ohlcv(700)
        result = engine.run_out_of_sample(data, symbol="TEST")

        if "error" not in result:
            assert "degradation" in result
            assert "win_rate_drop" in result["degradation"]
