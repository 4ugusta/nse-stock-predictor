"""Fundamental analysis data fetcher."""

import logging
from dataclasses import dataclass
from decimal import Decimal

import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class FundamentalData:
    """Fundamental metrics for a stock."""

    symbol: str
    name: str
    sector: str | None
    industry: str | None

    # Valuation
    market_cap: float | None  # In crores
    pe_ratio: float | None  # Trailing P/E
    forward_pe: float | None
    pb_ratio: float | None  # Price to Book
    ps_ratio: float | None  # Price to Sales
    peg_ratio: float | None  # P/E to Growth

    # Profitability
    profit_margin: float | None
    operating_margin: float | None
    roe: float | None  # Return on Equity
    roa: float | None  # Return on Assets

    # Growth
    revenue_growth: float | None
    earnings_growth: float | None

    # Dividends
    dividend_yield: float | None
    payout_ratio: float | None

    # Financial Health
    debt_to_equity: float | None
    current_ratio: float | None
    quick_ratio: float | None

    # Price levels
    current_price: float
    week_52_high: float | None
    week_52_low: float | None
    avg_volume: int | None

    # Analyst ratings
    target_price: float | None
    recommendation: str | None  # buy, hold, sell
    num_analysts: int | None

    def get_valuation_score(self) -> tuple[float, str]:
        """Calculate valuation score (0-100, higher = more undervalued).

        Returns:
            Tuple of (score, interpretation)
        """
        score = 50.0  # Base score
        reasons = []

        # P/E analysis
        if self.pe_ratio:
            if self.pe_ratio < 15:
                score += 15
                reasons.append("Low P/E (value)")
            elif self.pe_ratio < 25:
                score += 5
                reasons.append("Reasonable P/E")
            elif self.pe_ratio > 40:
                score -= 15
                reasons.append("High P/E (expensive)")
            else:
                score -= 5
                reasons.append("Elevated P/E")

        # P/B analysis
        if self.pb_ratio:
            if self.pb_ratio < 2:
                score += 10
                reasons.append("Low P/B")
            elif self.pb_ratio > 5:
                score -= 10
                reasons.append("High P/B")

        # PEG analysis (best indicator)
        if self.peg_ratio:
            if self.peg_ratio < 1:
                score += 15
                reasons.append("PEG < 1 (undervalued)")
            elif self.peg_ratio < 1.5:
                score += 5
                reasons.append("PEG reasonable")
            elif self.peg_ratio > 2:
                score -= 10
                reasons.append("PEG > 2 (overvalued)")

        # 52-week position
        if self.week_52_high and self.week_52_low:
            range_position = (self.current_price - self.week_52_low) / (
                self.week_52_high - self.week_52_low
            )
            if range_position < 0.3:
                score += 10
                reasons.append("Near 52W low")
            elif range_position > 0.9:
                score -= 5
                reasons.append("Near 52W high")

        score = max(0, min(100, score))
        interpretation = "; ".join(reasons) if reasons else "No clear valuation signals"

        return score, interpretation

    def get_quality_score(self) -> tuple[float, str]:
        """Calculate quality score based on profitability and health.

        Returns:
            Tuple of (score, interpretation)
        """
        score = 50.0
        reasons = []

        # ROE analysis
        if self.roe:
            if self.roe > 20:
                score += 15
                reasons.append(f"Excellent ROE ({self.roe:.1f}%)")
            elif self.roe > 15:
                score += 10
                reasons.append(f"Good ROE ({self.roe:.1f}%)")
            elif self.roe < 10:
                score -= 10
                reasons.append(f"Low ROE ({self.roe:.1f}%)")

        # Profit margin
        if self.profit_margin:
            margin_pct = self.profit_margin * 100
            if margin_pct > 20:
                score += 10
                reasons.append(f"High margin ({margin_pct:.1f}%)")
            elif margin_pct > 10:
                score += 5
                reasons.append(f"Good margin ({margin_pct:.1f}%)")
            elif margin_pct < 5:
                score -= 10
                reasons.append(f"Low margin ({margin_pct:.1f}%)")

        # Debt analysis
        if self.debt_to_equity is not None:
            if self.debt_to_equity < 0.5:
                score += 10
                reasons.append("Low debt")
            elif self.debt_to_equity > 2:
                score -= 15
                reasons.append("High debt risk")

        # Growth
        if self.earnings_growth:
            growth_pct = self.earnings_growth * 100
            if growth_pct > 20:
                score += 10
                reasons.append(f"Strong growth ({growth_pct:.1f}%)")
            elif growth_pct > 10:
                score += 5
                reasons.append(f"Good growth ({growth_pct:.1f}%)")
            elif growth_pct < 0:
                score -= 10
                reasons.append("Declining earnings")

        score = max(0, min(100, score))
        interpretation = "; ".join(reasons) if reasons else "No clear quality signals"

        return score, interpretation

    def to_summary_dict(self) -> dict:
        """Convert to summary dictionary for display."""
        val_score, val_reason = self.get_valuation_score()
        qual_score, qual_reason = self.get_quality_score()

        return {
            "symbol": self.symbol,
            "name": self.name,
            "sector": self.sector or "N/A",
            "market_cap_cr": f"₹{self.market_cap / 100:,.0f} Cr" if self.market_cap else "N/A",
            "current_price": f"₹{self.current_price:.2f}",
            "pe_ratio": f"{self.pe_ratio:.1f}" if self.pe_ratio else "N/A",
            "pb_ratio": f"{self.pb_ratio:.1f}" if self.pb_ratio else "N/A",
            "roe": f"{self.roe:.1f}%" if self.roe else "N/A",
            "debt_to_equity": f"{self.debt_to_equity:.2f}" if self.debt_to_equity else "N/A",
            "dividend_yield": f"{self.dividend_yield * 100:.2f}%" if self.dividend_yield else "N/A",
            "52w_high": f"₹{self.week_52_high:.2f}" if self.week_52_high else "N/A",
            "52w_low": f"₹{self.week_52_low:.2f}" if self.week_52_low else "N/A",
            "target_price": f"₹{self.target_price:.2f}" if self.target_price else "N/A",
            "analyst_rating": self.recommendation or "N/A",
            "valuation_score": val_score,
            "valuation_reason": val_reason,
            "quality_score": qual_score,
            "quality_reason": qual_reason,
        }


class FundamentalFetcher:
    """Fetches fundamental data for stocks."""

    def __init__(self) -> None:
        """Initialize fetcher."""
        pass

    def get_fundamentals(self, symbol: str) -> FundamentalData:
        """Fetch fundamental data for a symbol.

        Args:
            symbol: NSE stock symbol

        Returns:
            FundamentalData object
        """
        yahoo_symbol = f"{symbol}.NS"
        ticker = yf.Ticker(yahoo_symbol)

        try:
            info = ticker.info

            # Get current price from fast_info for accuracy
            try:
                fast_info = ticker.fast_info
                current_price = fast_info.last_price
            except Exception:
                current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)

            return FundamentalData(
                symbol=symbol,
                name=info.get("longName") or info.get("shortName") or symbol,
                sector=info.get("sector"),
                industry=info.get("industry"),
                # Valuation
                market_cap=info.get("marketCap"),
                pe_ratio=info.get("trailingPE"),
                forward_pe=info.get("forwardPE"),
                pb_ratio=info.get("priceToBook"),
                ps_ratio=info.get("priceToSalesTrailing12Months"),
                peg_ratio=info.get("pegRatio"),
                # Profitability
                profit_margin=info.get("profitMargins"),
                operating_margin=info.get("operatingMargins"),
                roe=info.get("returnOnEquity") * 100 if info.get("returnOnEquity") else None,
                roa=info.get("returnOnAssets") * 100 if info.get("returnOnAssets") else None,
                # Growth
                revenue_growth=info.get("revenueGrowth"),
                earnings_growth=info.get("earningsGrowth"),
                # Dividends
                dividend_yield=info.get("dividendYield"),
                payout_ratio=info.get("payoutRatio"),
                # Financial Health
                debt_to_equity=info.get("debtToEquity") / 100 if info.get("debtToEquity") else None,
                current_ratio=info.get("currentRatio"),
                quick_ratio=info.get("quickRatio"),
                # Price levels
                current_price=current_price,
                week_52_high=info.get("fiftyTwoWeekHigh"),
                week_52_low=info.get("fiftyTwoWeekLow"),
                avg_volume=info.get("averageVolume"),
                # Analyst data
                target_price=info.get("targetMeanPrice"),
                recommendation=info.get("recommendationKey"),
                num_analysts=info.get("numberOfAnalystOpinions"),
            )

        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {e}")
            raise

    def get_multiple_fundamentals(self, symbols: list[str]) -> dict[str, FundamentalData]:
        """Fetch fundamentals for multiple symbols.

        Args:
            symbols: List of NSE stock symbols

        Returns:
            Dictionary mapping symbols to FundamentalData
        """
        results = {}

        for symbol in symbols:
            try:
                results[symbol] = self.get_fundamentals(symbol)
            except Exception as e:
                logger.warning(f"Could not fetch fundamentals for {symbol}: {e}")
                continue

        return results
