"""AI-powered stock advisor using Claude."""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime

from anthropic import Anthropic

from stock_predictor.analysis.fundamentals import FundamentalData, FundamentalFetcher
from stock_predictor.analysis.sentiment.analyzer import SentimentAnalyzer
from stock_predictor.analysis.signals.signal_generator import SignalGenerator
from stock_predictor.core.entities.signal import TradingSignal, TradingStyle
from stock_predictor.infrastructure.data_providers.yahoo_provider import YahooDataProvider
from stock_predictor.infrastructure.news_providers.aggregator import NewsAggregator

logger = logging.getLogger(__name__)


@dataclass
class AIRecommendation:
    """AI-generated stock recommendation."""

    symbol: str
    action: str  # STRONG BUY, BUY, HOLD, SELL, AVOID
    confidence: int  # 1-100
    target_price: float | None
    stop_loss: float | None
    time_horizon: str  # Short-term, Medium-term, Long-term
    risk_level: str  # Low, Medium, High
    key_reasons: list[str]
    risks: list[str]
    summary: str


@dataclass
class MarketAnalysis:
    """AI analysis of overall market conditions."""

    market_sentiment: str  # Bullish, Bearish, Neutral
    recommended_sectors: list[str]
    sectors_to_avoid: list[str]
    top_picks: list[AIRecommendation]
    market_summary: str
    strategy_advice: str


class AIAdvisor:
    """AI-powered stock advisor using Claude for intelligent analysis."""

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize AI advisor.

        Args:
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = Anthropic(api_key=self.api_key)
        self.data_provider = YahooDataProvider()
        self.signal_generator = SignalGenerator()
        self.fundamental_fetcher = FundamentalFetcher()
        self.news_aggregator = NewsAggregator()
        self.sentiment_analyzer = SentimentAnalyzer()

    def _gather_stock_data(self, symbol: str) -> dict:
        """Gather all available data for a stock.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with all data
        """
        data = {"symbol": symbol, "timestamp": datetime.now().isoformat()}

        # Technical data
        try:
            historical = self.data_provider.get_historical(symbol, period="6mo")
            signal = self.signal_generator.generate(historical, symbol, TradingStyle.SWING)
            indicators = self.signal_generator.get_indicator_summary(historical)

            data["technical"] = {
                "signal": signal.signal_type.value,
                "signal_strength": signal.strength.value,
                "confidence": f"{signal.confidence:.1%}",
                "entry_price": float(signal.entry_price),
                "target_price": float(signal.target_price) if signal.target_price else None,
                "stop_loss": float(signal.stop_loss) if signal.stop_loss else None,
                "indicators": {
                    "price": indicators["price"],
                    "rsi": indicators["rsi"],
                    "macd": indicators["macd"],
                    "support": indicators["support_resistance"]["nearest_support"],
                    "resistance": indicators["support_resistance"]["nearest_resistance"],
                },
                "reasons": signal.reasons,
            }
        except Exception as e:
            logger.warning(f"Could not fetch technical data for {symbol}: {e}")
            data["technical"] = {"error": str(e)}

        # Fundamental data
        try:
            fundamentals = self.fundamental_fetcher.get_fundamentals(symbol)
            data["fundamentals"] = fundamentals.to_summary_dict()
        except Exception as e:
            logger.warning(f"Could not fetch fundamentals for {symbol}: {e}")
            data["fundamentals"] = {"error": str(e)}

        # News and sentiment
        try:
            articles = self.news_aggregator.fetch_for_symbol(symbol, limit_per_source=5)
            articles = self.sentiment_analyzer.analyze_articles(articles)
            aggregate = self.sentiment_analyzer.get_aggregate_sentiment(articles)

            data["news_sentiment"] = {
                "overall_sentiment": aggregate.category.value,
                "sentiment_score": aggregate.overall_score,
                "article_count": aggregate.article_count,
                "positive": aggregate.positive_count,
                "negative": aggregate.negative_count,
                "recent_headlines": [
                    {
                        "title": a.title,
                        "sentiment": a.sentiment.compound if a.sentiment else 0,
                        "age_hours": a.age_hours,
                    }
                    for a in articles[:5]
                ],
            }
        except Exception as e:
            logger.warning(f"Could not fetch news for {symbol}: {e}")
            data["news_sentiment"] = {"error": str(e)}

        return data

    def analyze_stock(self, symbol: str) -> AIRecommendation:
        """Get AI-powered analysis and recommendation for a stock.

        Args:
            symbol: Stock symbol

        Returns:
            AIRecommendation object
        """
        # Gather all data
        stock_data = self._gather_stock_data(symbol)

        # Create prompt for Claude
        prompt = f"""You are an expert Indian stock market analyst. Analyze this stock and provide a clear investment recommendation.

STOCK DATA FOR {symbol}:
{json.dumps(stock_data, indent=2, default=str)}

Based on this comprehensive data (technical indicators, fundamentals, and news sentiment), provide your analysis in the following JSON format:

{{
    "action": "STRONG BUY" | "BUY" | "HOLD" | "SELL" | "AVOID",
    "confidence": <1-100>,
    "target_price": <price or null>,
    "stop_loss": <price or null>,
    "time_horizon": "Short-term (1-2 weeks)" | "Medium-term (1-3 months)" | "Long-term (6+ months)",
    "risk_level": "Low" | "Medium" | "High",
    "key_reasons": ["reason1", "reason2", "reason3"],
    "risks": ["risk1", "risk2"],
    "summary": "2-3 sentence summary of the recommendation"
}}

IMPORTANT GUIDELINES:
1. Be realistic and conservative - never promise guaranteed returns
2. Consider all three aspects: technicals, fundamentals, and sentiment
3. Factor in current market conditions (it's {datetime.now().strftime('%B %Y')})
4. For Indian markets, consider factors like FII/DII flows, RBI policies
5. If data is insufficient or conflicting, recommend HOLD with lower confidence
6. Always mention key risks

Return ONLY the JSON object, no other text."""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse response
            result_text = response.content[0].text.strip()

            # Handle potential markdown code blocks
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
                result_text = result_text.strip()

            result = json.loads(result_text)

            return AIRecommendation(
                symbol=symbol,
                action=result["action"],
                confidence=result["confidence"],
                target_price=result.get("target_price"),
                stop_loss=result.get("stop_loss"),
                time_horizon=result["time_horizon"],
                risk_level=result["risk_level"],
                key_reasons=result["key_reasons"],
                risks=result["risks"],
                summary=result["summary"],
            )

        except Exception as e:
            logger.error(f"AI analysis failed for {symbol}: {e}")
            raise

    def get_todays_picks(
        self,
        symbols: list[str] | None = None,
        budget: float | None = None,
        risk_tolerance: str = "medium",
    ) -> MarketAnalysis:
        """Get AI-powered stock picks for today.

        Args:
            symbols: List of symbols to analyze (auto-selects based on risk if not provided)
            budget: Investment budget in INR (optional)
            risk_tolerance: "low", "medium", or "high"

        Returns:
            MarketAnalysis with recommendations
        """
        from stock_predictor.infrastructure.config.constants import (
            FNO_STOCKS,
            HIGH_VOLATILITY_STOCKS,
            NIFTY50_SYMBOLS,
            NIFTY_MIDCAP_SYMBOLS,
            NIFTY_NEXT50_SYMBOLS,
        )

        if symbols is None:
            # Select stock universe based on risk tolerance
            if risk_tolerance == "low":
                # Conservative: Blue-chip Nifty 50 stocks
                symbols = NIFTY50_SYMBOLS
            elif risk_tolerance == "high":
                # Aggressive: High volatility + midcap stocks
                symbols = list(set(HIGH_VOLATILITY_STOCKS + NIFTY_MIDCAP_SYMBOLS))
            else:
                # Medium: F&O stocks (liquid with decent volatility)
                symbols = list(set(FNO_STOCKS + NIFTY_NEXT50_SYMBOLS))

        # First, get market overview
        market_data = self._get_market_overview()

        # Pre-screen stocks using technical analysis
        screened_stocks = self._prescreen_stocks(symbols)

        # Gather detailed data for top candidates
        candidates_data = []
        for symbol in screened_stocks[:10]:  # Analyze top 10 candidates
            try:
                stock_data = self._gather_stock_data(symbol)
                candidates_data.append(stock_data)
            except Exception as e:
                logger.warning(f"Could not gather data for {symbol}: {e}")
                continue

        # Create comprehensive prompt for Claude
        prompt = f"""You are an expert Indian stock market advisor. Based on the following market data and stock analysis, provide investment recommendations for TODAY ({datetime.now().strftime('%d %B %Y')}).

MARKET OVERVIEW:
{json.dumps(market_data, indent=2, default=str)}

CANDIDATE STOCKS (pre-screened for technical strength):
{json.dumps(candidates_data, indent=2, default=str)}

INVESTOR PROFILE:
- Risk Tolerance: {risk_tolerance}
- Budget: {"₹" + f"{budget:,.0f}" if budget else "Not specified"}

Provide your analysis in the following JSON format:

{{
    "market_sentiment": "Bullish" | "Bearish" | "Neutral",
    "recommended_sectors": ["sector1", "sector2"],
    "sectors_to_avoid": ["sector1"],
    "market_summary": "2-3 sentences about current market conditions",
    "strategy_advice": "Specific advice for today based on market conditions",
    "top_picks": [
        {{
            "symbol": "SYMBOL",
            "action": "STRONG BUY" | "BUY",
            "confidence": <1-100>,
            "target_price": <price>,
            "stop_loss": <price>,
            "time_horizon": "Short-term" | "Medium-term" | "Long-term",
            "risk_level": "Low" | "Medium" | "High",
            "key_reasons": ["reason1", "reason2"],
            "risks": ["risk1"],
            "summary": "Why this stock today"
        }}
    ]
}}

GUIDELINES:
1. Select 0-5 top picks that match the investor's risk tolerance
2. For LOW risk: Focus on large-cap, dividend stocks, low volatility
3. For MEDIUM risk: Mix of growth and value, quality mid-caps
4. For HIGH risk: High-growth potential, momentum plays
5. Ensure diversification across sectors
6. Consider current market sentiment
7. Be specific about entry points and stop losses
8. IMPORTANT: If market conditions are unfavourable (high VIX, global risk-off, circuit breaker), return an EMPTY top_picks list and explain in strategy_advice why it's better to stay out today. Do NOT force picks when there are no good setups.

Return ONLY the JSON object."""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )

            result_text = response.content[0].text.strip()

            # Handle potential markdown code blocks
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
                result_text = result_text.strip()

            result = json.loads(result_text)

            # Convert to objects
            top_picks = [
                AIRecommendation(
                    symbol=pick["symbol"],
                    action=pick["action"],
                    confidence=pick["confidence"],
                    target_price=pick.get("target_price"),
                    stop_loss=pick.get("stop_loss"),
                    time_horizon=pick["time_horizon"],
                    risk_level=pick["risk_level"],
                    key_reasons=pick["key_reasons"],
                    risks=pick["risks"],
                    summary=pick["summary"],
                )
                for pick in result["top_picks"]
            ]

            return MarketAnalysis(
                market_sentiment=result["market_sentiment"],
                recommended_sectors=result["recommended_sectors"],
                sectors_to_avoid=result["sectors_to_avoid"],
                top_picks=top_picks,
                market_summary=result["market_summary"],
                strategy_advice=result["strategy_advice"],
            )

        except Exception as e:
            logger.error(f"AI market analysis failed: {e}")
            raise

    def _get_market_overview(self) -> dict:
        """Get current market overview data."""
        overview = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M"),
            "market_open": self.data_provider.is_market_open(),
        }

        # Get index data
        try:
            nifty = self.data_provider.get_index_data("NIFTY50")
            overview["nifty50"] = {
                "value": float(nifty.value),
                "change": float(nifty.change),
                "change_pct": float(nifty.change_percent),
            }
        except Exception:
            pass

        try:
            banknifty = self.data_provider.get_index_data("BANKNIFTY")
            overview["banknifty"] = {
                "value": float(banknifty.value),
                "change": float(banknifty.change),
                "change_pct": float(banknifty.change_percent),
            }
        except Exception:
            pass

        # Get market news sentiment
        try:
            articles = self.news_aggregator.fetch_market_news(limit_per_source=5)
            articles = self.sentiment_analyzer.analyze_articles(articles)
            aggregate = self.sentiment_analyzer.get_aggregate_sentiment(articles)
            overview["market_sentiment"] = {
                "score": aggregate.overall_score,
                "category": aggregate.category.value,
            }
        except Exception:
            pass

        return overview

    def _prescreen_stocks(self, symbols: list[str], limit: int = 15) -> list[str]:
        """Pre-screen stocks based on technical signals.

        Args:
            symbols: List of symbols to screen
            limit: Maximum number to return

        Returns:
            List of promising symbols
        """
        scored = []

        for symbol in symbols:
            try:
                data = self.data_provider.get_historical(symbol, period="3mo")
                if len(data) < 50:
                    continue

                signal = self.signal_generator.generate(data, symbol, TradingStyle.SWING)

                # Score based on signal
                score = signal.confidence
                if signal.signal_type.value in ["strong_buy", "buy"]:
                    score += 0.3
                elif signal.signal_type.value in ["sell", "strong_sell"]:
                    score -= 0.3

                scored.append((symbol, score))

            except Exception:
                continue

        # Sort by score and return top symbols
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored[:limit]]
