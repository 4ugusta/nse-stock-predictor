"""Market intelligence utilities for India-specific market context.

Provides:
- India VIX fetching (fear/greed gauge for signal confidence adjustment)
- Gap analysis (overnight gap detection for entry adjustment)
- Market breadth via nselib (NIFTY 500 advance/decline ratio from NSE)
- FII/DII derivatives flow via nselib (participant-wise trading data from NSE)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Module-level caches: avoids hammering NSE/Yahoo when scanning many stocks.
# VIX: 5 minutes, Breadth: 5 minutes, FII/DII: 30 minutes (published once/day).
_vix_cache: VIXReading | None = None
_vix_cache_ts: float = 0.0
_VIX_CACHE_TTL = 300  # seconds

_breadth_cache: MarketBreadth | None = None
_breadth_cache_ts: float = 0.0
_BREADTH_CACHE_TTL = 300  # seconds

_fii_dii_cache: FIIDIIActivity | None = None
_fii_dii_cache_ts: float = 0.0
_FII_DII_CACHE_TTL = 1800  # 30 minutes (data only changes once/day)


@dataclass
class VIXReading:
    """India VIX reading with interpretation."""

    value: float
    regime: str  # "low", "normal", "elevated", "high", "extreme"
    confidence_adjustment: float  # Multiplier for signal confidence (0.5-1.0)
    description: str


@dataclass
class GapAnalysis:
    """Overnight gap analysis for a stock."""

    has_gap: bool
    gap_percent: float  # Positive = gap up, negative = gap down
    gap_type: str  # "none", "small", "medium", "large"
    fill_probability: float  # Estimated probability of gap fill (0-1)
    entry_adjustment: str  # "wait_for_fill", "enter_at_open", "skip"


@dataclass
class MarketBreadth:
    """Market breadth indicators from NSE via nselib."""

    advance_decline_ratio: float | None
    advances: int | None
    declines: int | None
    unchanged: int | None
    nifty_change_pct: float | None
    breadth_signal: str  # "bullish", "bearish", "neutral", "unavailable"
    confidence_adjustment: float  # 0.85-1.1 multiplier for signal confidence
    description: str


@dataclass
class FIIDIIActivity:
    """FII/DII derivatives trading activity from NSE."""

    fii_net_contracts: float  # Positive = net long, negative = net short
    fii_index_futures_net: float  # FII index futures net position
    fii_index_options_net: float  # FII index options net position
    dii_net_contracts: float
    trade_date: str
    flow_signal: str  # "fii_bullish", "fii_bearish", "mixed", "unavailable"
    confidence_adjustment: float  # 0.9-1.1 multiplier
    description: str


def fetch_india_vix() -> VIXReading:
    """Fetch current India VIX from Yahoo Finance.

    India VIX measures expected near-term volatility of the Nifty 50.
    Higher VIX = more fear = signals are less reliable.

    Results are cached for 5 minutes to avoid excessive API calls when
    scanning multiple stocks.

    Returns:
        VIXReading with value and confidence adjustment
    """
    global _vix_cache, _vix_cache_ts
    if _vix_cache is not None and (time.monotonic() - _vix_cache_ts) < _VIX_CACHE_TTL:
        return _vix_cache

    try:
        vix_ticker = yf.Ticker("^INDIAVIX")
        history = vix_ticker.history(period="5d")

        if history.empty:
            return VIXReading(
                value=0.0,
                regime="unavailable",
                confidence_adjustment=1.0,
                description="India VIX data unavailable",
            )

        vix_value = float(history["Close"].iloc[-1])

        if vix_value < 12:
            regime = "low"
            adj = 1.0  # Low fear, signals reliable
            desc = f"India VIX {vix_value:.1f}: Low fear, market complacent"
        elif vix_value < 18:
            regime = "normal"
            adj = 1.0
            desc = f"India VIX {vix_value:.1f}: Normal volatility"
        elif vix_value < 25:
            regime = "elevated"
            adj = 0.85  # Mild confidence reduction
            desc = f"India VIX {vix_value:.1f}: Elevated fear, reduce position sizes"
        elif vix_value < 35:
            regime = "high"
            adj = 0.7  # Significant confidence reduction
            desc = f"India VIX {vix_value:.1f}: High fear, expect large moves"
        else:
            regime = "extreme"
            adj = 0.5  # Major confidence reduction
            desc = f"India VIX {vix_value:.1f}: Extreme fear, avoid new positions"

        result = VIXReading(
            value=vix_value,
            regime=regime,
            confidence_adjustment=adj,
            description=desc,
        )
        _vix_cache = result
        _vix_cache_ts = time.monotonic()
        return result

    except Exception as e:
        logger.warning(f"Failed to fetch India VIX: {e}")
        result = VIXReading(
            value=0.0,
            regime="unavailable",
            confidence_adjustment=1.0,
            description=f"India VIX fetch failed: {e}",
        )
        # Cache failures too to avoid repeated failed fetches
        _vix_cache = result
        _vix_cache_ts = time.monotonic()
        return result


def analyze_gap(data: pd.DataFrame) -> GapAnalysis:
    """Analyze overnight gap between previous close and current open.

    Large gaps often fill during the session, so entering at the open
    after a gap may give a worse price than waiting.

    Args:
        data: DataFrame with OHLCV data (at least 2 rows)

    Returns:
        GapAnalysis with gap details and entry recommendation
    """
    if len(data) < 2:
        return GapAnalysis(
            has_gap=False,
            gap_percent=0.0,
            gap_type="none",
            fill_probability=0.0,
            entry_adjustment="enter_at_open",
        )

    prev_close = float(data["close"].iloc[-2])
    curr_open = float(data["open"].iloc[-1])

    if prev_close <= 0:
        return GapAnalysis(
            has_gap=False,
            gap_percent=0.0,
            gap_type="none",
            fill_probability=0.0,
            entry_adjustment="enter_at_open",
        )

    gap_pct = ((curr_open - prev_close) / prev_close) * 100

    abs_gap = abs(gap_pct)
    if abs_gap < 0.5:
        gap_type = "none"
        fill_prob = 0.0
        entry = "enter_at_open"
    elif abs_gap < 1.5:
        gap_type = "small"
        fill_prob = 0.7  # Small gaps fill ~70% of the time
        entry = "enter_at_open"
    elif abs_gap < 3.0:
        gap_type = "medium"
        fill_prob = 0.55
        entry = "wait_for_fill"  # Better to wait for partial fill
    else:
        gap_type = "large"
        fill_prob = 0.35  # Large gaps often don't fill same day
        entry = "skip"  # Too risky to chase

    return GapAnalysis(
        has_gap=abs_gap >= 0.5,
        gap_percent=round(gap_pct, 2),
        gap_type=gap_type,
        fill_probability=fill_prob,
        entry_adjustment=entry,
    )


def get_market_breadth() -> MarketBreadth:
    """Fetch market breadth from NSE via nselib.

    Uses NIFTY TOTAL MKT index (broadest coverage: ~750 stocks) for
    advance/decline data. Falls back to NIFTY 500 if unavailable.

    Results are cached for 5 minutes to avoid excessive NSE API calls.

    Returns:
        MarketBreadth with advance/decline ratio and signal
    """
    global _breadth_cache, _breadth_cache_ts
    if _breadth_cache is not None and (time.monotonic() - _breadth_cache_ts) < _BREADTH_CACHE_TTL:
        return _breadth_cache

    try:
        from nselib import capital_market

        df = capital_market.market_watch_all_indices()

        if df is None or df.empty:
            return _breadth_unavailable("NSE returned empty index data")

        # Prefer NIFTY TOTAL MKT (broadest), fall back to NIFTY 500
        row = None
        for symbol in ["NIFTY TOTAL MKT", "NIFTY 500", "NIFTY 50"]:
            matches = df[df["indexSymbol"] == symbol]
            if len(matches) > 0:
                row = matches.iloc[0]
                break

        if row is None:
            return _breadth_unavailable("Could not find NIFTY index in data")

        advances = int(row["advances"])
        declines = int(row["declines"])
        unchanged = int(row["unchanged"])

        # Get NIFTY 50 change for context
        nifty_change = None
        nifty_row = df[df["indexSymbol"] == "NIFTY 50"]
        if len(nifty_row) > 0:
            nifty_change = float(nifty_row.iloc[0]["percentChange"])

        # A/D ratio: advances / max(declines, 1) to avoid division by zero
        ad_ratio = advances / max(declines, 1)

        # Classify breadth signal
        if ad_ratio >= 2.0:
            signal = "bullish"
            adj = 1.05  # Broad participation confirms signals
            desc = f"Strong breadth: {advances} advancing vs {declines} declining (A/D {ad_ratio:.2f})"
        elif ad_ratio >= 1.2:
            signal = "mildly_bullish"
            adj = 1.0
            desc = f"Positive breadth: {advances} advancing vs {declines} declining (A/D {ad_ratio:.2f})"
        elif ad_ratio >= 0.8:
            signal = "neutral"
            adj = 1.0
            desc = f"Neutral breadth: {advances} advancing vs {declines} declining (A/D {ad_ratio:.2f})"
        elif ad_ratio >= 0.5:
            signal = "mildly_bearish"
            adj = 0.95
            desc = f"Weak breadth: {advances} advancing vs {declines} declining (A/D {ad_ratio:.2f})"
        else:
            signal = "bearish"
            adj = 0.90  # Broad selling pressure degrades signal confidence
            desc = f"Poor breadth: {advances} advancing vs {declines} declining (A/D {ad_ratio:.2f})"

        result = MarketBreadth(
            advance_decline_ratio=round(ad_ratio, 3),
            advances=advances,
            declines=declines,
            unchanged=unchanged,
            nifty_change_pct=nifty_change,
            breadth_signal=signal,
            confidence_adjustment=adj,
            description=desc,
        )
        _breadth_cache = result
        _breadth_cache_ts = time.monotonic()
        return result

    except ImportError:
        logger.warning("nselib not installed — market breadth unavailable")
        return _breadth_unavailable("nselib not installed")
    except Exception as e:
        logger.warning(f"Failed to fetch market breadth: {e}")
        result = _breadth_unavailable(f"Fetch failed: {e}")
        # Cache failures to avoid repeated failed fetches
        _breadth_cache = result
        _breadth_cache_ts = time.monotonic()
        return result


def _breadth_unavailable(reason: str) -> MarketBreadth:
    """Return an unavailable MarketBreadth with neutral confidence."""
    return MarketBreadth(
        advance_decline_ratio=None,
        advances=None,
        declines=None,
        unchanged=None,
        nifty_change_pct=None,
        breadth_signal="unavailable",
        confidence_adjustment=1.0,
        description=reason,
    )


def get_fii_dii_activity() -> FIIDIIActivity:
    """Fetch FII/DII derivatives trading activity from NSE via nselib.

    Uses participant_wise_trading_volume to get FII/DII net positions in
    index futures and options. This data is published by NSE after market hours.

    Results are cached for 30 minutes (data only changes once per day).

    Returns:
        FIIDIIActivity with net flow signal and confidence adjustment
    """
    global _fii_dii_cache, _fii_dii_cache_ts
    if _fii_dii_cache is not None and (time.monotonic() - _fii_dii_cache_ts) < _FII_DII_CACHE_TTL:
        return _fii_dii_cache

    try:
        from nselib import derivatives

        # Try recent trading days (skip weekends, NSE holidays)
        now = datetime.now()
        df = None
        trade_date_str = ""
        for days_back in range(5):
            check_date = now - timedelta(days=days_back)
            if check_date.weekday() >= 5:  # Skip weekends
                continue
            trade_date_str = check_date.strftime("%d-%m-%Y")
            try:
                df = derivatives.participant_wise_trading_volume(trade_date_str)
                if df is not None and not df.empty:
                    break
                df = None
            except Exception:
                continue

        if df is None or df.empty:
            return _fii_dii_unavailable("No recent FII/DII data available from NSE")

        # Extract FII and DII rows
        fii_row = df[df["Client Type"] == "FII"]
        dii_row = df[df["Client Type"] == "DII"]

        if fii_row.empty:
            return _fii_dii_unavailable("FII row not found in participant data")

        fii = fii_row.iloc[0]
        fii_idx_fut_long = float(fii["Future Index Long"])
        fii_idx_fut_short = float(fii["Future Index Short"])
        fii_idx_fut_net = fii_idx_fut_long - fii_idx_fut_short

        fii_idx_call_long = float(fii["Option Index Call Long"])
        fii_idx_put_long = float(fii["Option Index Put Long"])
        fii_idx_call_short = float(fii["Option Index Call Short"])
        fii_idx_put_short = float(fii["Option Index Put Short"])
        # Net bullish options = (calls bought + puts sold) - (puts bought + calls sold)
        fii_idx_opt_net = (fii_idx_call_long + fii_idx_put_short) - (fii_idx_put_long + fii_idx_call_short)

        fii_total_long = float(fii["Total Long Contracts      "].strip()) if isinstance(fii["Total Long Contracts      "], str) else float(fii["Total Long Contracts      "])
        fii_total_short = float(fii["Total Short Contracts"].strip()) if isinstance(fii["Total Short Contracts"], str) else float(fii["Total Short Contracts"])
        fii_net = fii_total_long - fii_total_short

        # DII data
        dii_net = 0.0
        if not dii_row.empty:
            dii = dii_row.iloc[0]
            dii_total_long = float(dii["Total Long Contracts      "].strip()) if isinstance(dii["Total Long Contracts      "], str) else float(dii["Total Long Contracts      "])
            dii_total_short = float(dii["Total Short Contracts"].strip()) if isinstance(dii["Total Short Contracts"], str) else float(dii["Total Short Contracts"])
            dii_net = dii_total_long - dii_total_short

        # Classify FII flow signal (FII is the dominant institutional force)
        # Use index futures as the primary signal — it reflects directional bets
        if fii_idx_fut_net > 2000:
            signal = "fii_bullish"
            adj = 1.05
            desc = (
                f"FII bullish: net long {fii_idx_fut_net:+,.0f} index futures, "
                f"net {fii_net:+,.0f} total contracts ({trade_date_str})"
            )
        elif fii_idx_fut_net < -2000:
            signal = "fii_bearish"
            adj = 0.92
            desc = (
                f"FII bearish: net short {fii_idx_fut_net:+,.0f} index futures, "
                f"net {fii_net:+,.0f} total contracts ({trade_date_str})"
            )
        else:
            signal = "mixed"
            adj = 1.0
            desc = (
                f"FII neutral: net {fii_idx_fut_net:+,.0f} index futures, "
                f"net {fii_net:+,.0f} total contracts ({trade_date_str})"
            )

        result = FIIDIIActivity(
            fii_net_contracts=fii_net,
            fii_index_futures_net=fii_idx_fut_net,
            fii_index_options_net=fii_idx_opt_net,
            dii_net_contracts=dii_net,
            trade_date=trade_date_str,
            flow_signal=signal,
            confidence_adjustment=adj,
            description=desc,
        )
        _fii_dii_cache = result
        _fii_dii_cache_ts = time.monotonic()
        return result

    except ImportError:
        logger.warning("nselib not installed — FII/DII data unavailable")
        return _fii_dii_unavailable("nselib not installed")
    except Exception as e:
        logger.warning(f"Failed to fetch FII/DII data: {e}")
        result = _fii_dii_unavailable(f"Fetch failed: {e}")
        _fii_dii_cache = result
        _fii_dii_cache_ts = time.monotonic()
        return result


def _fii_dii_unavailable(reason: str) -> FIIDIIActivity:
    """Return an unavailable FIIDIIActivity with neutral confidence."""
    return FIIDIIActivity(
        fii_net_contracts=0.0,
        fii_index_futures_net=0.0,
        fii_index_options_net=0.0,
        dii_net_contracts=0.0,
        trade_date="",
        flow_signal="unavailable",
        confidence_adjustment=1.0,
        description=reason,
    )


def get_fii_dii_note() -> str:
    """Return a human-readable summary of FII/DII activity.

    Fetches real data from NSE via nselib and formats it as a note.
    Falls back to an informational message if data is unavailable.

    Returns:
        Formatted string with FII/DII activity summary
    """
    activity = get_fii_dii_activity()
    if activity.flow_signal == "unavailable":
        return (
            "FII/DII flow data unavailable. "
            "Check https://www.nseindia.com/reports/fii-dii for latest data. "
            f"Reason: {activity.description}"
        )
    return activity.description
