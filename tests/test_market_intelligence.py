"""Tests for market intelligence module (VIX, gap analysis, breadth, FII/DII)."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from stock_predictor.analysis.market_intelligence import (
    fetch_india_vix,
    analyze_gap,
    get_market_breadth,
    get_fii_dii_activity,
    get_fii_dii_note,
    VIXReading,
    GapAnalysis,
    MarketBreadth,
    FIIDIIActivity,
    _VIX_CACHE_TTL,
    _BREADTH_CACHE_TTL,
    _FII_DII_CACHE_TTL,
)


def _make_ohlcv(n=50, base_price=1000.0, volume=100000, seed=42):
    """Create OHLCV data for gap analysis tests."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    np.random.seed(seed)
    close = base_price + np.cumsum(np.random.randn(n) * 10)
    close = np.maximum(close, 50.0)
    high = close + np.abs(np.random.randn(n) * 3) + 1
    low = close - np.abs(np.random.randn(n) * 3) - 1
    low = np.maximum(low, 1.0)
    open_ = close + np.random.randn(n) * 2
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    vol = np.random.randint(volume // 2, volume * 2, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=dates,
    )


def _make_mock_indices_df(advances=300, declines=200, unchanged=5, nifty_change=-0.5):
    """Create a mock DataFrame matching nselib market_watch_all_indices output."""
    return pd.DataFrame({
        "key": ["IDX", "IDX", "IDX"],
        "index": ["NIFTY 50", "NIFTY 500", "NIFTY TOTAL MKT"],
        "indexSymbol": ["NIFTY 50", "NIFTY 500", "NIFTY TOTAL MKT"],
        "last": [25000.0, 23000.0, 21000.0],
        "variation": [-50.0, -40.0, -30.0],
        "percentChange": [nifty_change, -0.3, -0.2],
        "advances": [25, advances, advances + 50],
        "declines": [25, declines, declines + 50],
        "unchanged": [0, unchanged, unchanged + 2],
    })


def _make_mock_participant_df(
    fii_fut_long=15000, fii_fut_short=12000,
    fii_call_long=2000000, fii_put_long=2500000,
    fii_call_short=2100000, fii_put_short=2400000,
    fii_total_long=5000000, fii_total_short=5200000,
    dii_total_long=100000, dii_total_short=80000,
):
    """Create a mock DataFrame matching nselib participant_wise_trading_volume output."""
    return pd.DataFrame({
        "Client Type": ["Client", "DII", "FII", "Pro", "TOTAL"],
        "Future Index Long": [30000, 5, fii_fut_long, 25000, 70005],
        "Future Index Short": [35000, 2000, fii_fut_short, 21000, 70000],
        "Future Stock Long": [300000, 90000, 300000, 500000, 1190000],
        "Future Stock Short       ": [310000, 50000, 310000, 520000, 1190000],
        "Option Index Call Long": [10000000, 0, fii_call_long, 12000000, 24000000],
        "Option Index Put Long": [11000000, 0, fii_put_long, 13000000, 26500000],
        "Option Index Call Short": [10000000, 50, fii_call_short, 11900000, 24000050],
        "Option Index Put Short": [11000000, 0, fii_put_short, 13100000, 26500000],
        "Option Stock Call Long": [2000000, 5000, 300000, 3000000, 5305000],
        "Option Stock Put Long": [800000, 0, 150000, 2000000, 2950000],
        "Option Stock Call Short": [2000000, 20000, 310000, 3000000, 5330000],
        "Option Stock Put Short": [800000, 500, 160000, 2000000, 2960500],
        "Total Long Contracts      ": [23130000, dii_total_long, fii_total_long, 31525000, 59755000],
        "Total Short Contracts": [23145000, dii_total_short, fii_total_short, 31570000, 59795000],
    })


# ---------------------------------------------------------------------------
# VIX Tests
# ---------------------------------------------------------------------------
class TestVIXFetch:
    @patch("stock_predictor.analysis.market_intelligence.yf")
    def test_low_vix_no_confidence_penalty(self, mock_yf):
        """Low VIX should not reduce confidence."""
        import stock_predictor.analysis.market_intelligence as mi
        mi._vix_cache = None  # Clear cache

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame(
            {"Close": [10.0, 11.0, 10.5]},
            index=pd.date_range("2024-01-01", periods=3),
        )
        mock_yf.Ticker.return_value = mock_ticker

        result = fetch_india_vix()
        assert result.regime == "low"
        assert result.confidence_adjustment == 1.0

    @patch("stock_predictor.analysis.market_intelligence.yf")
    def test_high_vix_reduces_confidence(self, mock_yf):
        """High VIX should reduce confidence multiplier."""
        import stock_predictor.analysis.market_intelligence as mi
        mi._vix_cache = None  # Clear cache

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame(
            {"Close": [28.0, 30.0, 32.0]},
            index=pd.date_range("2024-01-01", periods=3),
        )
        mock_yf.Ticker.return_value = mock_ticker

        result = fetch_india_vix()
        assert result.regime == "high"
        assert result.confidence_adjustment == 0.7

    @patch("stock_predictor.analysis.market_intelligence.yf")
    def test_extreme_vix(self, mock_yf):
        """Extreme VIX (>35) should halve confidence."""
        import stock_predictor.analysis.market_intelligence as mi
        mi._vix_cache = None

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame(
            {"Close": [38.0, 40.0, 42.0]},
            index=pd.date_range("2024-01-01", periods=3),
        )
        mock_yf.Ticker.return_value = mock_ticker

        result = fetch_india_vix()
        assert result.regime == "extreme"
        assert result.confidence_adjustment == 0.5

    @patch("stock_predictor.analysis.market_intelligence.yf")
    def test_empty_data_returns_unavailable(self, mock_yf):
        """Empty VIX data should return unavailable (no penalty)."""
        import stock_predictor.analysis.market_intelligence as mi
        mi._vix_cache = None

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_yf.Ticker.return_value = mock_ticker

        result = fetch_india_vix()
        assert result.regime == "unavailable"
        assert result.confidence_adjustment == 1.0

    @patch("stock_predictor.analysis.market_intelligence.yf")
    def test_fetch_failure_returns_unavailable(self, mock_yf):
        """API failure should return unavailable, not crash."""
        import stock_predictor.analysis.market_intelligence as mi
        mi._vix_cache = None

        mock_yf.Ticker.side_effect = Exception("Network error")
        result = fetch_india_vix()
        assert result.regime == "unavailable"
        assert result.confidence_adjustment == 1.0

    @patch("stock_predictor.analysis.market_intelligence.yf")
    def test_vix_caching(self, mock_yf):
        """Second call should use cache, not fetch again."""
        import stock_predictor.analysis.market_intelligence as mi
        mi._vix_cache = None

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame(
            {"Close": [15.0]},
            index=pd.date_range("2024-01-01", periods=1),
        )
        mock_yf.Ticker.return_value = mock_ticker

        result1 = fetch_india_vix()
        result2 = fetch_india_vix()

        # yf.Ticker should be called only once due to caching
        assert mock_yf.Ticker.call_count == 1
        assert result1.value == result2.value


# ---------------------------------------------------------------------------
# Gap Analysis Tests
# ---------------------------------------------------------------------------
class TestGapAnalysis:
    def test_no_gap_for_small_move(self):
        """Less than 0.5% difference should be 'no gap'."""
        data = _make_ohlcv(50)
        # Set open nearly equal to prev close
        data.iloc[-1, data.columns.get_loc("open")] = data.iloc[-2]["close"] * 1.001
        result = analyze_gap(data)
        assert not result.has_gap
        assert result.gap_type == "none"
        assert result.entry_adjustment == "enter_at_open"

    def test_small_gap_up(self):
        """0.5-1.5% gap should be classified as 'small'."""
        data = _make_ohlcv(50)
        prev_close = data.iloc[-2]["close"]
        data.iloc[-1, data.columns.get_loc("open")] = prev_close * 1.01  # 1% gap
        result = analyze_gap(data)
        assert result.has_gap
        assert result.gap_type == "small"
        assert result.gap_percent > 0

    def test_medium_gap_wait_for_fill(self):
        """1.5-3% gap should recommend 'wait_for_fill'."""
        data = _make_ohlcv(50)
        prev_close = data.iloc[-2]["close"]
        data.iloc[-1, data.columns.get_loc("open")] = prev_close * 1.02  # 2% gap
        result = analyze_gap(data)
        assert result.gap_type == "medium"
        assert result.entry_adjustment == "wait_for_fill"

    def test_large_gap_skip(self):
        """3%+ gap should recommend 'skip'."""
        data = _make_ohlcv(50)
        prev_close = data.iloc[-2]["close"]
        data.iloc[-1, data.columns.get_loc("open")] = prev_close * 1.04  # 4% gap
        result = analyze_gap(data)
        assert result.gap_type == "large"
        assert result.entry_adjustment == "skip"

    def test_gap_down(self):
        """Gap down should have negative gap_percent."""
        data = _make_ohlcv(50)
        prev_close = data.iloc[-2]["close"]
        data.iloc[-1, data.columns.get_loc("open")] = prev_close * 0.97  # -3% gap
        result = analyze_gap(data)
        assert result.has_gap
        assert result.gap_percent < 0

    def test_single_row_no_gap(self):
        """Single row of data should return no gap."""
        data = _make_ohlcv(1)
        result = analyze_gap(data)
        assert not result.has_gap

    def test_zero_prev_close(self):
        """Zero previous close should return no gap (not division error)."""
        data = _make_ohlcv(5)
        data.iloc[-2, data.columns.get_loc("close")] = 0
        result = analyze_gap(data)
        assert not result.has_gap


# ---------------------------------------------------------------------------
# Market Breadth Tests (nselib)
# ---------------------------------------------------------------------------
class TestMarketBreadth:
    def _clear_cache(self):
        import stock_predictor.analysis.market_intelligence as mi
        mi._breadth_cache = None

    @patch("nselib.capital_market.market_watch_all_indices")
    def test_bullish_breadth(self, mock_indices):
        """High A/D ratio (>=2.0) should return bullish signal."""
        self._clear_cache()
        # NIFTY TOTAL MKT gets advances+50, declines+50 in mock helper
        mock_indices.return_value = _make_mock_indices_df(advances=400, declines=150)
        result = get_market_breadth()
        assert result.breadth_signal == "bullish"
        assert result.confidence_adjustment > 1.0
        assert result.advances == 450  # NIFTY TOTAL MKT = advances + 50
        assert result.declines == 200  # NIFTY TOTAL MKT = declines + 50
        assert result.advance_decline_ratio > 2.0

    @patch("nselib.capital_market.market_watch_all_indices")
    def test_bearish_breadth(self, mock_indices):
        """Low A/D ratio (<0.5) should return bearish signal."""
        self._clear_cache()
        mock_indices.return_value = _make_mock_indices_df(advances=80, declines=400)
        result = get_market_breadth()
        assert result.breadth_signal == "bearish"
        assert result.confidence_adjustment < 1.0

    @patch("nselib.capital_market.market_watch_all_indices")
    def test_neutral_breadth(self, mock_indices):
        """A/D ratio near 1.0 should return neutral signal."""
        self._clear_cache()
        mock_indices.return_value = _make_mock_indices_df(advances=250, declines=260)
        result = get_market_breadth()
        assert result.breadth_signal == "neutral"
        assert result.confidence_adjustment == 1.0

    @patch("nselib.capital_market.market_watch_all_indices")
    def test_mildly_bullish_breadth(self, mock_indices):
        """A/D ratio 1.2-2.0 should return mildly_bullish."""
        self._clear_cache()
        mock_indices.return_value = _make_mock_indices_df(advances=300, declines=200)
        result = get_market_breadth()
        assert result.breadth_signal == "mildly_bullish"
        assert result.confidence_adjustment == 1.0

    @patch("nselib.capital_market.market_watch_all_indices")
    def test_mildly_bearish_breadth(self, mock_indices):
        """A/D ratio 0.5-0.8 should return mildly_bearish."""
        self._clear_cache()
        mock_indices.return_value = _make_mock_indices_df(advances=150, declines=250)
        result = get_market_breadth()
        assert result.breadth_signal == "mildly_bearish"
        assert result.confidence_adjustment < 1.0

    @patch("nselib.capital_market.market_watch_all_indices")
    def test_empty_data_returns_unavailable(self, mock_indices):
        """Empty NSE data should return unavailable."""
        self._clear_cache()
        mock_indices.return_value = pd.DataFrame()
        result = get_market_breadth()
        assert result.breadth_signal == "unavailable"
        assert result.confidence_adjustment == 1.0

    @patch("nselib.capital_market.market_watch_all_indices")
    def test_api_failure_returns_unavailable(self, mock_indices):
        """NSE API failure should return unavailable, not crash."""
        self._clear_cache()
        mock_indices.side_effect = Exception("Connection refused")
        result = get_market_breadth()
        assert result.breadth_signal == "unavailable"
        assert result.confidence_adjustment == 1.0

    @patch("nselib.capital_market.market_watch_all_indices")
    def test_breadth_caching(self, mock_indices):
        """Second call should use cache, not fetch again."""
        self._clear_cache()
        mock_indices.return_value = _make_mock_indices_df(advances=300, declines=200)

        result1 = get_market_breadth()
        result2 = get_market_breadth()

        assert mock_indices.call_count == 1
        assert result1.advance_decline_ratio == result2.advance_decline_ratio

    @patch("nselib.capital_market.market_watch_all_indices")
    def test_nifty_change_captured(self, mock_indices):
        """NIFTY 50 percentage change should be captured."""
        self._clear_cache()
        mock_indices.return_value = _make_mock_indices_df(nifty_change=-1.5)
        result = get_market_breadth()
        assert result.nifty_change_pct == pytest.approx(-1.5)

    @patch("nselib.capital_market.market_watch_all_indices")
    def test_zero_declines_no_division_error(self, mock_indices):
        """Zero declines should not cause division by zero."""
        self._clear_cache()
        # NIFTY TOTAL MKT gets declines+50=50, so A/D = 450/50 = 9.0
        mock_indices.return_value = _make_mock_indices_df(advances=400, declines=0)
        result = get_market_breadth()
        assert result.advance_decline_ratio == 9.0  # 450 / max(50, 1)
        assert result.breadth_signal == "bullish"


# ---------------------------------------------------------------------------
# FII/DII Activity Tests (nselib)
# ---------------------------------------------------------------------------
class TestFIIDIIActivity:
    def _clear_cache(self):
        import stock_predictor.analysis.market_intelligence as mi
        mi._fii_dii_cache = None

    @patch("nselib.derivatives.participant_wise_trading_volume")
    def test_fii_bullish(self, mock_pwv):
        """FII net long >2000 index futures should return fii_bullish."""
        self._clear_cache()
        mock_pwv.return_value = _make_mock_participant_df(
            fii_fut_long=15000, fii_fut_short=10000  # net +5000
        )
        result = get_fii_dii_activity()
        assert result.flow_signal == "fii_bullish"
        assert result.confidence_adjustment > 1.0
        assert result.fii_index_futures_net == 5000.0

    @patch("nselib.derivatives.participant_wise_trading_volume")
    def test_fii_bearish(self, mock_pwv):
        """FII net short >2000 index futures should return fii_bearish."""
        self._clear_cache()
        mock_pwv.return_value = _make_mock_participant_df(
            fii_fut_long=10000, fii_fut_short=15000  # net -5000
        )
        result = get_fii_dii_activity()
        assert result.flow_signal == "fii_bearish"
        assert result.confidence_adjustment < 1.0
        assert result.fii_index_futures_net == -5000.0

    @patch("nselib.derivatives.participant_wise_trading_volume")
    def test_fii_mixed(self, mock_pwv):
        """FII near-neutral index futures should return mixed."""
        self._clear_cache()
        mock_pwv.return_value = _make_mock_participant_df(
            fii_fut_long=12000, fii_fut_short=12500  # net -500
        )
        result = get_fii_dii_activity()
        assert result.flow_signal == "mixed"
        assert result.confidence_adjustment == 1.0

    @patch("nselib.derivatives.participant_wise_trading_volume")
    def test_dii_net_captured(self, mock_pwv):
        """DII net position should be captured."""
        self._clear_cache()
        mock_pwv.return_value = _make_mock_participant_df(
            dii_total_long=200000, dii_total_short=150000
        )
        result = get_fii_dii_activity()
        assert result.dii_net_contracts == 50000.0

    @patch("nselib.derivatives.participant_wise_trading_volume")
    def test_api_failure_returns_unavailable(self, mock_pwv):
        """NSE API failure should return unavailable, not crash."""
        self._clear_cache()
        mock_pwv.side_effect = Exception("Network error")
        result = get_fii_dii_activity()
        assert result.flow_signal == "unavailable"
        assert result.confidence_adjustment == 1.0

    @patch("nselib.derivatives.participant_wise_trading_volume")
    def test_empty_data_returns_unavailable(self, mock_pwv):
        """Empty data should return unavailable."""
        self._clear_cache()
        mock_pwv.return_value = pd.DataFrame()
        result = get_fii_dii_activity()
        assert result.flow_signal == "unavailable"

    @patch("nselib.derivatives.participant_wise_trading_volume")
    def test_fii_dii_caching(self, mock_pwv):
        """Second call should use cache, not fetch again."""
        self._clear_cache()
        mock_pwv.return_value = _make_mock_participant_df()

        result1 = get_fii_dii_activity()
        result2 = get_fii_dii_activity()

        assert mock_pwv.call_count == 1
        assert result1.fii_net_contracts == result2.fii_net_contracts

    @patch("nselib.derivatives.participant_wise_trading_volume")
    def test_trade_date_populated(self, mock_pwv):
        """Trade date should be populated from the successful fetch."""
        self._clear_cache()
        mock_pwv.return_value = _make_mock_participant_df()
        result = get_fii_dii_activity()
        assert result.trade_date != ""


# ---------------------------------------------------------------------------
# FII/DII Note Integration Test
# ---------------------------------------------------------------------------
class TestFIIDIINote:
    @patch("nselib.derivatives.participant_wise_trading_volume")
    def test_note_returns_description(self, mock_pwv):
        """get_fii_dii_note should return the activity description."""
        import stock_predictor.analysis.market_intelligence as mi
        mi._fii_dii_cache = None
        mock_pwv.return_value = _make_mock_participant_df(
            fii_fut_long=15000, fii_fut_short=10000
        )
        note = get_fii_dii_note()
        assert isinstance(note, str)
        assert "FII" in note

    @patch("nselib.derivatives.participant_wise_trading_volume")
    def test_note_unavailable_fallback(self, mock_pwv):
        """Unavailable data should return informational fallback."""
        import stock_predictor.analysis.market_intelligence as mi
        mi._fii_dii_cache = None
        mock_pwv.side_effect = Exception("No data")
        note = get_fii_dii_note()
        assert isinstance(note, str)
        assert "unavailable" in note.lower() or "NSE" in note
