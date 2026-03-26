"""Tests for the portfolio risk manager."""

import pytest
import pandas as pd
import numpy as np

from stock_predictor.analysis.risk_manager import (
    RiskManager,
    PortfolioPosition,
    PositionSize,
)


def _make_ohlcv(n=250, base_price=1000.0, volume=100000, seed=42):
    """Create OHLCV data."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
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
# Basic Risk Manager Tests
# ---------------------------------------------------------------------------
class TestRiskManagerInit:
    def test_default_init(self):
        rm = RiskManager(total_capital=1_000_000)
        assert rm.total_capital == 1_000_000
        assert rm.risk_per_trade <= rm.MAX_RISK_PER_TRADE
        assert len(rm.positions) == 0

    def test_risk_per_trade_capped(self):
        """Risk per trade should be capped at MAX_RISK_PER_TRADE."""
        rm = RiskManager(total_capital=1_000_000, risk_per_trade=5.0)
        assert rm.risk_per_trade <= rm.MAX_RISK_PER_TRADE


# ---------------------------------------------------------------------------
# Position Sizing Tests
# ---------------------------------------------------------------------------
class TestPositionSizing:
    def test_position_size_within_limits(self):
        """Position size should not exceed MAX_POSITION_SIZE."""
        rm = RiskManager(total_capital=1_000_000)
        data = _make_ohlcv(250)
        size = rm.calculate_position_size("RELIANCE", data, is_long=True)
        assert isinstance(size, PositionSize)
        assert size.position_value <= rm.total_capital * (rm.MAX_POSITION_SIZE / 100) + 1

    def test_position_size_positive(self):
        """Position size should be positive for valid input."""
        rm = RiskManager(total_capital=1_000_000)
        data = _make_ohlcv(250)
        size = rm.calculate_position_size("TCS", data)
        assert size.shares >= 0
        assert size.position_value >= 0

    def test_small_capital_small_position(self):
        """Small capital should lead to proportionally small positions."""
        rm_small = RiskManager(total_capital=50_000)
        rm_large = RiskManager(total_capital=5_000_000)
        data = _make_ohlcv(250, base_price=500.0)
        size_small = rm_small.calculate_position_size("TEST", data)
        size_large = rm_large.calculate_position_size("TEST", data)
        assert size_small.position_value <= size_large.position_value


# ---------------------------------------------------------------------------
# Sector Exposure Tests
# ---------------------------------------------------------------------------
class TestSectorExposure:
    def test_empty_portfolio_no_exposure(self):
        """Empty portfolio should have no sector exposure."""
        rm = RiskManager(total_capital=1_000_000)
        exposure = rm._get_sector_exposure()
        assert exposure == {}

    def test_sector_exposure_sums_to_100(self):
        """All sector exposures should sum to ~100% of invested value."""
        positions = [
            PortfolioPosition(
                symbol="RELIANCE", shares=10, entry_price=2500.0,
                current_price=2600.0, sector="Energy"
            ),
            PortfolioPosition(
                symbol="TCS", shares=20, entry_price=3500.0,
                current_price=3600.0, sector="IT"
            ),
        ]
        rm = RiskManager(total_capital=1_000_000, existing_positions=positions)
        exposure = rm._get_sector_exposure()
        total = sum(exposure.values())
        assert abs(total - 100.0) < 1.0  # Should sum to ~100%

    def test_sector_exposure_uses_invested_value(self):
        """Sector exposure should use invested value, not total_capital."""
        positions = [
            PortfolioPosition(
                symbol="RELIANCE", shares=10, entry_price=2500.0,
                current_price=2600.0, sector="Energy"
            ),
        ]
        rm = RiskManager(total_capital=10_000_000, existing_positions=positions)
        exposure = rm._get_sector_exposure()
        # With only 26K invested in 10M portfolio, exposure should be 100% of invested
        assert exposure.get("Energy", 0) == pytest.approx(100.0, abs=1.0)


# ---------------------------------------------------------------------------
# Portfolio Heat Tests
# ---------------------------------------------------------------------------
class TestPortfolioHeat:
    def test_empty_portfolio_zero_heat(self):
        """Empty portfolio should have zero heat."""
        rm = RiskManager(total_capital=1_000_000)
        heat = rm._get_portfolio_heat()
        assert heat == 0.0

    def test_heat_increases_with_positions(self):
        """Adding positions should increase portfolio heat."""
        pos1 = [PortfolioPosition(
            symbol="RELIANCE", shares=10, entry_price=2500.0,
            current_price=2600.0, sector="Energy"
        )]
        pos2 = pos1 + [PortfolioPosition(
            symbol="TCS", shares=20, entry_price=3500.0,
            current_price=3600.0, sector="IT"
        )]
        rm1 = RiskManager(total_capital=1_000_000, existing_positions=pos1)
        rm2 = RiskManager(total_capital=1_000_000, existing_positions=pos2)
        assert rm2._get_portfolio_heat() >= rm1._get_portfolio_heat()

    def test_heat_correlation_adjusted(self):
        """Same-sector positions should have higher heat than diversified."""
        # Two positions in same sector
        same_sector = [
            PortfolioPosition(
                symbol="RELIANCE", shares=10, entry_price=2500.0,
                current_price=2600.0, sector="Energy"
            ),
            PortfolioPosition(
                symbol="ONGC", shares=100, entry_price=250.0,
                current_price=260.0, sector="Energy"
            ),
        ]
        # Two positions in different sectors
        diff_sector = [
            PortfolioPosition(
                symbol="RELIANCE", shares=10, entry_price=2500.0,
                current_price=2600.0, sector="Energy"
            ),
            PortfolioPosition(
                symbol="TCS", shares=100, entry_price=250.0,
                current_price=260.0, sector="IT"
            ),
        ]
        rm_same = RiskManager(total_capital=1_000_000, existing_positions=same_sector)
        rm_diff = RiskManager(total_capital=1_000_000, existing_positions=diff_sector)
        # Same-sector should have higher heat due to correlation
        assert rm_same._get_portfolio_heat() >= rm_diff._get_portfolio_heat()


# ---------------------------------------------------------------------------
# Circuit Breaker Tests
# ---------------------------------------------------------------------------
class TestCircuitBreaker:
    def test_no_drawdown_no_trigger(self):
        """No drawdown should not trigger circuit breaker."""
        rm = RiskManager(total_capital=1_000_000)
        triggered, msg = rm.check_drawdown_circuit_breaker(
            peak_capital=1_000_000, realized_pnl=0
        )
        assert not triggered

    def test_large_drawdown_triggers(self):
        """20%+ drawdown should trigger circuit breaker."""
        rm = RiskManager(total_capital=800_000)  # Capital already reduced
        triggered, msg = rm.check_drawdown_circuit_breaker(
            peak_capital=1_000_000, realized_pnl=-200_000
        )
        # 800K capital + (-200K realized) = 600K current value
        # Drawdown = (1M - 600K) / 1M = 40%
        assert triggered
        assert "CIRCUIT BREAKER" in msg

    def test_drawdown_cannot_be_negative(self):
        """Circuit breaker should use max(0, drawdown)."""
        rm = RiskManager(total_capital=1_200_000)
        triggered, msg = rm.check_drawdown_circuit_breaker(
            peak_capital=1_000_000, realized_pnl=200_000
        )
        # Current value > peak: drawdown should be 0, not negative
        assert not triggered


# ---------------------------------------------------------------------------
# Risk Assessment Tests
# ---------------------------------------------------------------------------
class TestRiskAssessment:
    def test_risk_assessment_valid(self):
        """assess_trade should return valid assessment."""
        rm = RiskManager(total_capital=1_000_000)
        data = _make_ohlcv(250)
        assessment = rm.assess_trade(
            symbol="RELIANCE", data=data, is_long=True
        )
        assert assessment is not None
        assert hasattr(assessment, "can_trade")
        assert hasattr(assessment, "position_size")
        assert hasattr(assessment, "warnings")

    def test_too_many_positions_blocks(self):
        """Exceeding MAX_POSITIONS should block new trades."""
        positions = [
            PortfolioPosition(
                symbol=f"STOCK{i}", shares=10, entry_price=100.0,
                current_price=100.0, sector=f"Sector{i}"
            )
            for i in range(16)  # Exceeds MAX_POSITIONS (15)
        ]
        rm = RiskManager(total_capital=1_000_000, existing_positions=positions)
        data = _make_ohlcv(250)
        assessment = rm.assess_trade("NEWSTOCK", data)
        assert not assessment.can_trade


# ---------------------------------------------------------------------------
# Correlation Alignment Tests
# ---------------------------------------------------------------------------
class TestCorrelation:
    def test_correlation_returns_float(self):
        """Correlation calculation should return a float."""
        rm = RiskManager(total_capital=1_000_000)
        data_a = _make_ohlcv(250, seed=1)
        data_b = _make_ohlcv(250, seed=2)
        corr = rm.calculate_rolling_correlation(data_a, data_b)
        assert isinstance(corr, float)
        assert -1.0 <= corr <= 1.0

    def test_self_correlation_is_one(self):
        """Correlation of a stock with itself should be ~1.0."""
        rm = RiskManager(total_capital=1_000_000)
        data = _make_ohlcv(250)
        corr = rm.calculate_rolling_correlation(data, data)
        assert corr > 0.95

    def test_short_data_returns_zero(self):
        """Very short data should return 0 correlation (not crash)."""
        rm = RiskManager(total_capital=1_000_000)
        data_a = _make_ohlcv(10)
        data_b = _make_ohlcv(10)
        corr = rm.calculate_rolling_correlation(data_a, data_b)
        assert corr == 0.0
