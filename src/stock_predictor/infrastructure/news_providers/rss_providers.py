"""RSS-based news providers for Indian financial news."""

import logging
from datetime import datetime

import feedparser

from stock_predictor.core.entities.news import NewsArticle

logger = logging.getLogger(__name__)


class MoneyControlRSSProvider:
    """Fetches news from MoneyControl RSS feeds."""

    RSS_FEEDS = {
        "market": "https://www.moneycontrol.com/rss/marketreports.xml",
        "business": "https://www.moneycontrol.com/rss/business.xml",
        "stocks": "https://www.moneycontrol.com/rss/stocksnews.xml",
        "economy": "https://www.moneycontrol.com/rss/economy.xml",
        "results": "https://www.moneycontrol.com/rss/results.xml",
    }

    def __init__(self) -> None:
        """Initialize MoneyControl RSS provider."""
        self.source_name = "MoneyControl"

    def _parse_entry(self, entry: dict) -> NewsArticle:
        """Parse RSS entry into NewsArticle.

        Args:
            entry: feedparser entry dict

        Returns:
            NewsArticle object
        """
        published = entry.get("published_parsed")
        if published:
            published_at = datetime(*published[:6])
        else:
            published_at = datetime.now()

        # Clean up summary (remove HTML tags)
        summary = entry.get("summary", "")
        if summary:
            # Basic HTML tag removal
            import re
            summary = re.sub(r"<[^>]+>", "", summary)
            summary = summary[:500] if len(summary) > 500 else summary

        return NewsArticle(
            title=entry.get("title", ""),
            source=self.source_name,
            url=entry.get("link", ""),
            published_at=published_at,
            summary=summary,
        )

    def fetch_news(
        self,
        category: str = "market",
        limit: int = 20,
    ) -> list[NewsArticle]:
        """Fetch news from specified category.

        Args:
            category: Feed category (market, business, stocks, economy, results)
            limit: Maximum number of articles

        Returns:
            List of NewsArticle objects
        """
        if category not in self.RSS_FEEDS:
            logger.warning(f"Unknown category: {category}, using 'market'")
            category = "market"

        url = self.RSS_FEEDS[category]

        try:
            feed = feedparser.parse(url)

            if feed.bozo:
                logger.warning(f"Feed parsing warning: {feed.bozo_exception}")

            articles = []
            for entry in feed.entries[:limit]:
                try:
                    articles.append(self._parse_entry(entry))
                except Exception as e:
                    logger.warning(f"Error parsing entry: {e}")
                    continue

            return articles

        except Exception as e:
            logger.error(f"Error fetching MoneyControl RSS: {e}")
            return []

    def fetch_all_categories(self, limit_per_category: int = 10) -> list[NewsArticle]:
        """Fetch news from all categories.

        Args:
            limit_per_category: Maximum articles per category

        Returns:
            List of NewsArticle objects from all categories
        """
        all_articles = []

        for category in self.RSS_FEEDS:
            articles = self.fetch_news(category, limit_per_category)
            all_articles.extend(articles)

        # Remove duplicates
        seen_urls = set()
        unique = []
        for article in all_articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique.append(article)

        # Sort by date
        unique.sort(key=lambda x: x.published_at, reverse=True)
        return unique


class EconomicTimesRSSProvider:
    """Fetches news from Economic Times RSS feeds."""

    RSS_FEEDS = {
        "markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "stocks": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
        "companies": "https://economictimes.indiatimes.com/news/company/rssfeeds/2143429.cms",
        "economy": "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms",
        "industry": "https://economictimes.indiatimes.com/industry/rssfeeds/13352306.cms",
    }

    def __init__(self) -> None:
        """Initialize Economic Times RSS provider."""
        self.source_name = "Economic Times"

    def _parse_entry(self, entry: dict) -> NewsArticle:
        """Parse RSS entry into NewsArticle.

        Args:
            entry: feedparser entry dict

        Returns:
            NewsArticle object
        """
        published = entry.get("published_parsed")
        if published:
            published_at = datetime(*published[:6])
        else:
            published_at = datetime.now()

        # Clean up summary
        summary = entry.get("summary", "")
        if summary:
            import re
            summary = re.sub(r"<[^>]+>", "", summary)
            summary = summary[:500] if len(summary) > 500 else summary

        return NewsArticle(
            title=entry.get("title", ""),
            source=self.source_name,
            url=entry.get("link", ""),
            published_at=published_at,
            summary=summary,
        )

    def fetch_news(
        self,
        category: str = "markets",
        limit: int = 20,
    ) -> list[NewsArticle]:
        """Fetch news from specified category.

        Args:
            category: Feed category
            limit: Maximum number of articles

        Returns:
            List of NewsArticle objects
        """
        if category not in self.RSS_FEEDS:
            logger.warning(f"Unknown category: {category}, using 'markets'")
            category = "markets"

        url = self.RSS_FEEDS[category]

        try:
            feed = feedparser.parse(url)

            if feed.bozo:
                logger.warning(f"Feed parsing warning: {feed.bozo_exception}")

            articles = []
            for entry in feed.entries[:limit]:
                try:
                    articles.append(self._parse_entry(entry))
                except Exception as e:
                    logger.warning(f"Error parsing entry: {e}")
                    continue

            return articles

        except Exception as e:
            logger.error(f"Error fetching Economic Times RSS: {e}")
            return []

    def fetch_all_categories(self, limit_per_category: int = 10) -> list[NewsArticle]:
        """Fetch news from all categories.

        Args:
            limit_per_category: Maximum articles per category

        Returns:
            List of NewsArticle objects from all categories
        """
        all_articles = []

        for category in self.RSS_FEEDS:
            articles = self.fetch_news(category, limit_per_category)
            all_articles.extend(articles)

        # Remove duplicates
        seen_urls = set()
        unique = []
        for article in all_articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique.append(article)

        # Sort by date
        unique.sort(key=lambda x: x.published_at, reverse=True)
        return unique
