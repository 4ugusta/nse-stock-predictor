"""Sentiment analysis for financial news."""

import logging

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from stock_predictor.core.entities.news import AggregateSentiment, NewsArticle, SentimentResult

logger = logging.getLogger(__name__)


# Financial lexicon additions for VADER
FINANCIAL_LEXICON = {
    # Bullish terms
    "bullish": 3.0,
    "rally": 2.5,
    "surge": 2.5,
    "soar": 2.5,
    "breakout": 2.0,
    "outperform": 2.0,
    "upgrade": 2.5,
    "beat": 1.5,
    "buy": 1.5,
    "accumulate": 1.5,
    "growth": 1.5,
    "profit": 1.5,
    "gains": 1.5,
    "positive": 1.0,
    "recovery": 1.5,
    "rebound": 2.0,
    "momentum": 1.0,
    "upside": 1.5,
    "strong": 1.0,
    "robust": 1.5,
    "expansion": 1.5,
    "boom": 2.0,
    "dividend": 1.0,
    "bonus": 1.5,
    "split": 0.5,  # Stock split is generally positive
    "ipo": 0.5,
    "fii": 0.5,  # FII buying is generally positive context
    "dii": 0.5,

    # Bearish terms
    "bearish": -3.0,
    "crash": -3.0,
    "plunge": -2.5,
    "tumble": -2.5,
    "slump": -2.5,
    "breakdown": -2.0,
    "underperform": -2.0,
    "downgrade": -2.5,
    "miss": -1.5,
    "sell": -1.5,
    "reduce": -1.5,
    "loss": -2.0,
    "losses": -2.0,
    "decline": -1.5,
    "negative": -1.0,
    "weak": -1.5,
    "correction": -1.0,
    "downside": -1.5,
    "volatility": -0.5,
    "volatile": -0.5,
    "uncertainty": -1.0,
    "risk": -0.5,
    "recession": -2.5,
    "inflation": -1.0,
    "default": -2.5,
    "debt": -0.5,
    "fraud": -3.0,
    "scam": -3.0,
    "investigation": -1.5,
    "probe": -1.0,
    "warning": -1.5,
    "caution": -1.0,

    # Neutral but contextually important
    "flat": 0.0,
    "unchanged": 0.0,
    "steady": 0.3,
    "stable": 0.3,
    "consolidation": 0.0,
    "range-bound": 0.0,

    # Indian market specific
    "nifty": 0.0,
    "sensex": 0.0,
    "sebi": 0.0,
    "rbi": 0.0,
    "repo": -0.3,  # Rate hikes are often bearish
    "rate hike": -1.0,
    "rate cut": 1.0,
}


class SentimentAnalyzer:
    """Analyzes sentiment of financial text using VADER."""

    def __init__(self) -> None:
        """Initialize sentiment analyzer with financial lexicon."""
        self.vader = SentimentIntensityAnalyzer()
        self._load_financial_lexicon()

    def _load_financial_lexicon(self) -> None:
        """Add financial terms to VADER lexicon."""
        self.vader.lexicon.update(FINANCIAL_LEXICON)

    def analyze_text(self, text: str) -> SentimentResult:
        """Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            SentimentResult object
        """
        if not text:
            return SentimentResult(
                compound=0.0,
                positive=0.0,
                negative=0.0,
                neutral=1.0,
            )

        scores = self.vader.polarity_scores(text)

        return SentimentResult(
            compound=scores["compound"],
            positive=scores["pos"],
            negative=scores["neg"],
            neutral=scores["neu"],
        )

    def analyze_article(self, article: NewsArticle) -> NewsArticle:
        """Analyze sentiment of a news article.

        Args:
            article: NewsArticle to analyze

        Returns:
            Same article with sentiment added
        """
        # Combine title and summary for analysis
        text = article.title
        if article.summary:
            text += " " + article.summary

        sentiment = self.analyze_text(text)
        article.sentiment = sentiment

        return article

    def analyze_articles(self, articles: list[NewsArticle]) -> list[NewsArticle]:
        """Analyze sentiment of multiple articles.

        Args:
            articles: List of NewsArticle objects

        Returns:
            Same articles with sentiment added
        """
        return [self.analyze_article(article) for article in articles]

    def get_aggregate_sentiment(self, articles: list[NewsArticle]) -> AggregateSentiment:
        """Calculate aggregate sentiment across articles.

        Args:
            articles: List of NewsArticle objects (with sentiment analyzed)

        Returns:
            AggregateSentiment object
        """
        if not articles:
            return AggregateSentiment(
                overall_score=0.0,
                article_count=0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
            )

        # Ensure all articles have sentiment
        analyzed = []
        for article in articles:
            if article.sentiment is None:
                article = self.analyze_article(article)
            analyzed.append(article)

        # Calculate aggregate
        total_compound = sum(a.sentiment.compound for a in analyzed if a.sentiment)
        avg_compound = total_compound / len(analyzed)

        # Count by category
        positive_count = sum(
            1 for a in analyzed if a.sentiment and a.sentiment.compound > 0.05
        )
        negative_count = sum(
            1 for a in analyzed if a.sentiment and a.sentiment.compound < -0.05
        )
        neutral_count = len(analyzed) - positive_count - negative_count

        return AggregateSentiment(
            overall_score=avg_compound,
            article_count=len(analyzed),
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
        )

    def get_sentiment_summary(
        self, articles: list[NewsArticle]
    ) -> dict:
        """Get a detailed sentiment summary.

        Args:
            articles: List of NewsArticle objects

        Returns:
            Dictionary with sentiment details
        """
        analyzed = self.analyze_articles(articles)
        aggregate = self.get_aggregate_sentiment(analyzed)

        # Get most positive and negative articles
        sorted_by_sentiment = sorted(
            [a for a in analyzed if a.sentiment],
            key=lambda x: x.sentiment.compound,
            reverse=True,
        )

        most_positive = sorted_by_sentiment[:3] if sorted_by_sentiment else []
        most_negative = sorted_by_sentiment[-3:][::-1] if len(sorted_by_sentiment) > 3 else []

        return {
            "aggregate": aggregate.to_dict(),
            "most_positive": [
                {"title": a.title, "sentiment": a.sentiment.compound}
                for a in most_positive
            ],
            "most_negative": [
                {"title": a.title, "sentiment": a.sentiment.compound}
                for a in most_negative
            ],
            "recent_articles": [
                {
                    "title": a.title,
                    "source": a.source,
                    "sentiment": a.sentiment.display_text if a.sentiment else "N/A",
                    "score": f"{a.sentiment.compound:+.2f}" if a.sentiment else "N/A",
                    "age": f"{a.age_hours:.1f}h ago",
                }
                for a in analyzed[:10]
            ],
        }
