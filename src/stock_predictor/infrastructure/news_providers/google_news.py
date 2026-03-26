"""Google News provider for stock-related news."""

import logging
from datetime import datetime
from urllib.parse import quote_plus

import feedparser

from stock_predictor.core.entities.news import NewsArticle

logger = logging.getLogger(__name__)


class GoogleNewsProvider:
    """Fetches news from Google News RSS feeds."""

    BASE_URL = "https://news.google.com/rss/search"

    def __init__(self, language: str = "en", region: str = "IN") -> None:
        """Initialize Google News provider.

        Args:
            language: Language code (en, hi, etc.)
            region: Region code (IN for India)
        """
        self.language = language
        self.region = region

    def _build_url(self, query: str) -> str:
        """Build Google News RSS URL.

        Args:
            query: Search query

        Returns:
            Complete RSS URL
        """
        encoded_query = quote_plus(query)
        return f"{self.BASE_URL}?q={encoded_query}&hl={self.language}-{self.region}&gl={self.region}&ceid={self.region}:{self.language}"

    def _parse_entry(self, entry: dict) -> NewsArticle:
        """Parse RSS entry into NewsArticle.

        Args:
            entry: feedparser entry dict

        Returns:
            NewsArticle object
        """
        # Parse published date
        published = entry.get("published_parsed")
        if published:
            published_at = datetime(*published[:6])
        else:
            published_at = datetime.now()

        # Extract source from title (Google News format: "Title - Source")
        title = entry.get("title", "")
        source = "Google News"
        if " - " in title:
            parts = title.rsplit(" - ", 1)
            title = parts[0]
            source = parts[1] if len(parts) > 1 else source

        return NewsArticle(
            title=title,
            source=source,
            url=entry.get("link", ""),
            published_at=published_at,
            summary=entry.get("summary"),
        )

    def fetch_news(
        self,
        symbol: str | None = None,
        keywords: list[str] | None = None,
        limit: int = 20,
    ) -> list[NewsArticle]:
        """Fetch news articles.

        Args:
            symbol: Stock symbol to search for
            keywords: Additional keywords
            limit: Maximum number of articles

        Returns:
            List of NewsArticle objects
        """
        # Build search query
        query_parts = []

        if symbol:
            # Add symbol and common variations
            query_parts.append(f"{symbol} NSE")
            query_parts.append(f"{symbol} stock")

        if keywords:
            query_parts.extend(keywords)

        if not query_parts:
            # Default to Indian stock market news
            query_parts = ["Indian stock market", "NSE", "Nifty"]

        query = " OR ".join(query_parts)
        url = self._build_url(query)

        try:
            feed = feedparser.parse(url)

            if feed.bozo:
                logger.warning(f"Feed parsing warning: {feed.bozo_exception}")

            articles = []
            for entry in feed.entries[:limit]:
                try:
                    article = self._parse_entry(entry)
                    if symbol:
                        article.related_symbols.append(symbol)
                    articles.append(article)
                except Exception as e:
                    logger.warning(f"Error parsing entry: {e}")
                    continue

            return articles

        except Exception as e:
            logger.error(f"Error fetching Google News: {e}")
            return []

    def fetch_market_news(self, limit: int = 20) -> list[NewsArticle]:
        """Fetch general Indian market news.

        Args:
            limit: Maximum number of articles

        Returns:
            List of NewsArticle objects
        """
        queries = [
            "Nifty 50 today",
            "NSE BSE market",
            "Indian stock market news",
        ]

        all_articles = []

        for query in queries:
            url = self._build_url(query)
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[: limit // len(queries)]:
                    try:
                        article = self._parse_entry(entry)
                        all_articles.append(article)
                    except Exception:
                        continue
            except Exception as e:
                logger.warning(f"Error fetching query '{query}': {e}")

        # Remove duplicates based on URL
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)

        # Sort by publish date
        unique_articles.sort(key=lambda x: x.published_at, reverse=True)

        return unique_articles[:limit]
