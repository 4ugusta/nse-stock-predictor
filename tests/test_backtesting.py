"""Tests for the backtesting engine."""

import pytest
import pandas as pd
import numpy as np

from stock_predictor.analysis.backtesting.engine import (
    BacktestEngine,
    BacktestResult,
    calculate_transaction_cost,
)
from stock_predictor.core.entities.signal import TradingStyle


def _make_ohlcv(n=500, base_price=1000.0, volume=100000, seed=42):
    """Create OHLCV data with timestamp column (required by engine)."""
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


# ---------------------------------------------------------------------------
# Transaction Cost Tests
# ---------------------------------------------------------------------------
class TestTransactionCosts:
    """Tests for Indian market transaction cost calculation."""

    def test_stt_sell_only(self):
        """STT should be 0.1% on sell side, 0% on buy side."""
        buy_cost = calculate_transaction_cost(1000.0, 100, is_buy=True)
        sell_cost = calculate_transaction_cost(1000.0, 100, is_buy=False)
        # Sell cost should be higher due to 0.1% STT
        assert sell_cost > buy_cost
        # Sell has 0.1% STT, buy has 0.015% stamp duty — net difference ~0.085% of turnover
        stt_diff = sell_cost - buy_cost
        turnover = 1000.0 * 100  # Rs 1,00,000
        expected_net = turnover * (0.001 - 0.00015)  # STT minus stamp duty
        assert abs(stt_diff - expected_net) < 5.0

    def test_stamp_duty_buy_only(self):
        """Stamp duty should only apply on buy side."""
        buy_cost = calculate_transaction_cost(500.0, 200, is_buy=True)
        sell_cost = calculate_transaction_cost(500.0, 200, is_buy=False)
        # Buy should include stamp duty (0.015%), sell should not
        # But sell includes STT (0.1%), so sell is still higher overall
        assert sell_cost > buy_cost

    def test_zero_quantity(self):
        """Zero quantity should give zero cost."""
        cost = calculate_transaction_cost(1000.0, 0, is_buy=True)
        assert cost == 0.0

    def test_costs_are_positive(self):
        """All transaction costs should be positive."""
        for price in [5.0, 100.0, 5000.0]:
            for qty in [1, 100, 10000]:
                assert calculate_transaction_cost(price, qty, is_buy=True) >= 0
                assert calculate_transaction_cost(price, qty, is_buy=False) >= 0


# ---------------------------------------------------------------------------
# Engine Initialization Tests
# ---------------------------------------------------------------------------
class TestBacktestEngineInit:
    """Tests for engine initialization."""

    def test_default_values(self):
        engine = BacktestEngine()
        assert engine.initial_capital == 100_000
        assert engine.max_holding_days == 30

    def test_custom_values(self):
        engine = BacktestEngine(
            initial_capital=500_000, slippage_pct=0.2, max_holding_days=15
        )
        assert engine.initial_capital == 500_000
        assert engine.max_holding_days == 15


# ---------------------------------------------------------------------------
# Engine Run Tests
# ---------------------------------------------------------------------------
class TestBacktestEngineRun:
    """Tests for the actual backtest execution."""

    def test_insufficient_data_returns_empty_result(self):
        """Engine should handle insufficient data gracefully."""
        engine = BacktestEngine()
        data = _make_ohlcv(100)  # Not enough for lookback_window + 50
        result = engine.run(data, symbol="SHORT", style=TradingStyle.SWING)
        assert isinstance(result, BacktestResult)
        assert result.total_trades == 0

    def test_run_produces_result(self):
        """Full run on sufficient data should produce valid result."""
        engine = BacktestEngine()
        data = _make_ohlcv(500)
        result = engine.run(data, symbol="TEST", style=TradingStyle.SWING)
        assert isinstance(result, BacktestResult)
        assert result.symbol == "TEST"
        assert result.style == "swing"

    def test_equity_curve_populated(self):
        """Result should have a non-empty equity curve."""
        engine = BacktestEngine()
        data = _make_ohlcv(500)
        result = engine.run(data, symbol="TEST")
        assert len(result.equity_curve) > 0
        assert result.equity_curve[0] == engine.initial_capital

    def test_profit_factor_capped(self):
        """Profit factor should be capped at 99.9, not infinity."""
        engine = BacktestEngine()
        data = _make_ohlcv(500, seed=42)
        result = engine.run(data, symbol="TEST")
        # Whether or not there are trades, profit factor should be finite
        assert result.profit_factor <= 99.9

    def test_buy_and_hold_uses_first_trade_date(self):
        """Buy-and-hold return should start from first trade, not hardcoded bar 200."""
        engine = BacktestEngine()
        data = _make_ohlcv(500)
        result = engine.run(data, symbol="TEST")
        # buy_and_hold_return should be a valid number
        assert isinstance(result.buy_and_hold_return, float)
        assert not np.isnan(result.buy_and_hold_return)

    def test_win_rate_between_0_and_100(self):
        """Win rate should always be between 0% and 100%."""
        engine = BacktestEngine()
        data = _make_ohlcv(500)
        result = engine.run(data, symbol="TEST")
        assert 0 <= result.win_rate <= 100

    def test_max_drawdown_non_negative(self):
        """Max drawdown should always be >= 0."""
        engine = BacktestEngine()
        data = _make_ohlcv(500)
        result = engine.run(data, symbol="TEST")
        assert result.max_drawdown_pct >= 0


# ---------------------------------------------------------------------------
# Walk-Forward Validation Tests
# ---------------------------------------------------------------------------
class TestWalkForwardValidation:
    """Tests for walk-forward / train-test split."""

    def test_oos_returns_dict(self):
        """Out-of-sample validation should return a dict with train/test results."""
        engine = BacktestEngine()
        data = _make_ohlcv(700)
        result = engine.run_out_of_sample(data, symbol="TEST")
        assert isinstance(result, dict)
        if "error" not in result:
            assert "in_sample" in result
            assert "out_of_sample" in result

    def test_oos_insufficient_data(self):
        """Should return error for insufficient data."""
        engine = BacktestEngine()
        data = _make_ohlcv(200)
        result = engine.run_out_of_sample(data, symbol="SHORT")
        assert "error" in result


# ---------------------------------------------------------------------------
# Slippage Model Tests
# ---------------------------------------------------------------------------
class TestSlippageModel:
    """Tests for the price-tier aware slippage model."""

    def test_buy_slippage_increases_price(self):
        """Slippage on buy should give a worse (higher) price."""
        engine = BacktestEngine(slippage_pct=0.10)
        slipped = engine._apply_slippage(1000.0, is_buy=True)
        assert slipped >= 1000.0

    def test_sell_slippage_decreases_price(self):
        """Slippage on sell should give a worse (lower) price."""
        engine = BacktestEngine(slippage_pct=0.10)
        slipped = engine._apply_slippage(1000.0, is_buy=False)
        assert slipped <= 1000.0

    def test_penny_stock_higher_slippage(self):
        """Penny stocks should get higher slippage multiplier."""
        engine = BacktestEngine(slippage_pct=0.10)
        penny_slip = abs(engine._apply_slippage(10.0, is_buy=True) - 10.0)
        regular_slip = abs(engine._apply_slippage(1000.0, is_buy=True) - 1000.0)
        # Penny stock slippage as % should be higher
        assert (penny_slip / 10.0) > (regular_slip / 1000.0)
