"""News aggregator combining multiple news sources."""

import logging
from datetime import datetime

from stock_predictor.core.entities.news import NewsArticle
from stock_predictor.infrastructure.news_providers.google_news import GoogleNewsProvider
from stock_predictor.infrastructure.news_providers.rss_providers import (
    EconomicTimesRSSProvider,
    MoneyControlRSSProvider,
)

logger = logging.getLogger(__name__)


class NewsAggregator:
    """Aggregates news from multiple providers."""

    def __init__(self) -> None:
        """Initialize news aggregator with all providers."""
        self.google = GoogleNewsProvider()
        self.moneycontrol = MoneyControlRSSProvider()
        self.et = EconomicTimesRSSProvider()

    def _deduplicate(self, articles: list[NewsArticle]) -> list[NewsArticle]:
        """Remove duplicate articles based on URL and similar titles.

        Args:
            articles: List of articles

        Returns:
            Deduplicated list
        """
        seen_urls = set()
        seen_titles = set()
        unique = []

        for article in articles:
            # Check URL
            if article.url in seen_urls:
                continue

            # Check for similar titles (first 50 chars)
            title_key = article.title[:50].lower() if article.title else ""
            if title_key in seen_titles:
                continue

            seen_urls.add(article.url)
            seen_titles.add(title_key)
            unique.append(article)

        return unique

    def fetch_for_symbol(
        self,
        symbol: str,
        limit_per_source: int = 10,
    ) -> list[NewsArticle]:
        """Fetch news for a specific stock symbol.

        Args:
            symbol: Stock symbol
            limit_per_source: Max articles per source

        Returns:
            Aggregated list of NewsArticle objects
        """
        all_articles = []

        # Google News is best for symbol-specific search
        try:
            google_articles = self.google.fetch_news(
                symbol=symbol, limit=limit_per_source * 2
            )
            all_articles.extend(google_articles)
        except Exception as e:
            logger.warning(f"Google News fetch failed: {e}")

        # Deduplicate and sort
        unique = self._deduplicate(all_articles)
        unique.sort(key=lambda x: x.published_at, reverse=True)

        # Add symbol to related_symbols
        for article in unique:
            if symbol not in article.related_symbols:
                article.related_symbols.append(symbol)

        return unique[:limit_per_source * 2]

    def fetch_market_news(
        self,
        limit_per_source: int = 10,
    ) -> list[NewsArticle]:
        """Fetch general market news from all sources.

        Args:
            limit_per_source: Max articles per source

        Returns:
            Aggregated list of NewsArticle objects
        """
        all_articles = []

        # Google News - market news
        try:
            google_articles = self.google.fetch_market_news(limit=limit_per_source)
            all_articles.extend(google_articles)
        except Exception as e:
            logger.warning(f"Google News fetch failed: {e}")

        # MoneyControl
        try:
            mc_articles = self.moneycontrol.fetch_news("market", limit_per_source)
            all_articles.extend(mc_articles)
        except Exception as e:
            logger.warning(f"MoneyControl fetch failed: {e}")

        # Economic Times
        try:
            et_articles = self.et.fetch_news("markets", limit_per_source)
            all_articles.extend(et_articles)
        except Exception as e:
            logger.warning(f"Economic Times fetch failed: {e}")

        # Deduplicate and sort
        unique = self._deduplicate(all_articles)
        unique.sort(key=lambda x: x.published_at, reverse=True)

        return unique

    def fetch_all(
        self,
        symbol: str | None = None,
        limit_per_source: int = 10,
    ) -> list[NewsArticle]:
        """Fetch news from all sources.

        Args:
            symbol: Optional stock symbol to filter by
            limit_per_source: Max articles per source

        Returns:
            Aggregated list of NewsArticle objects
        """
        if symbol:
            return self.fetch_for_symbol(symbol, limit_per_source)
        return self.fetch_market_news(limit_per_source)

    def get_recent_news(
        self,
        hours: int = 24,
        symbol: str | None = None,
        limit: int = 30,
    ) -> list[NewsArticle]:
        """Get news from the last N hours.

        Args:
            hours: Number of hours to look back
            symbol: Optional stock symbol
            limit: Maximum articles to return

        Returns:
            List of recent NewsArticle objects
        """
        articles = self.fetch_all(symbol=symbol, limit_per_source=limit // 2)

        # Filter by time
        cutoff = datetime.now().timestamp() - (hours * 3600)
        recent = [
            a for a in articles
            if a.published_at.timestamp() > cutoff
        ]

        return recent[:limit]
