"""News and sentiment domain entities."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class SentimentCategory(Enum):
    """Sentiment interpretation categories."""

    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


@dataclass
class SentimentResult:
    """Sentiment analysis result for text."""

    compound: float  # -1.0 to 1.0
    positive: float  # 0.0 to 1.0
    negative: float  # 0.0 to 1.0
    neutral: float  # 0.0 to 1.0

    @property
    def category(self) -> SentimentCategory:
        """Get sentiment category based on compound score."""
        if self.compound >= 0.5:
            return SentimentCategory.VERY_BULLISH
        elif self.compound >= 0.1:
            return SentimentCategory.BULLISH
        elif self.compound <= -0.5:
            return SentimentCategory.VERY_BEARISH
        elif self.compound <= -0.1:
            return SentimentCategory.BEARISH
        return SentimentCategory.NEUTRAL

    @property
    def display_text(self) -> str:
        """Human-readable sentiment display."""
        return self.category.value.replace("_", " ").title()


@dataclass
class NewsArticle:
    """A news article from any source."""

    title: str
    source: str
    url: str
    published_at: datetime
    summary: str | None = None
    content: str | None = None
    sentiment: SentimentResult | None = None
    related_symbols: list[str] = field(default_factory=list)

    @property
    def age_hours(self) -> float:
        """Get article age in hours."""
        delta = datetime.now() - self.published_at
        return delta.total_seconds() / 3600

    @property
    def is_recent(self) -> bool:
        """Check if article is less than 24 hours old."""
        return self.age_hours < 24

    def to_dict(self) -> dict:
        """Convert to dictionary for display."""
        return {
            "title": self.title,
            "source": self.source,
            "url": self.url,
            "published": self.published_at.strftime("%Y-%m-%d %H:%M"),
            "age": f"{self.age_hours:.1f}h ago",
            "sentiment": self.sentiment.display_text if self.sentiment else "N/A",
            "sentiment_score": f"{self.sentiment.compound:+.2f}" if self.sentiment else "N/A",
        }


@dataclass
class AggregateSentiment:
    """Aggregated sentiment across multiple articles."""

    overall_score: float  # -1.0 to 1.0
    article_count: int
    positive_count: int
    negative_count: int
    neutral_count: int

    @property
    def category(self) -> SentimentCategory:
        """Get overall sentiment category."""
        if self.overall_score >= 0.5:
            return SentimentCategory.VERY_BULLISH
        elif self.overall_score >= 0.1:
            return SentimentCategory.BULLISH
        elif self.overall_score <= -0.5:
            return SentimentCategory.VERY_BEARISH
        elif self.overall_score <= -0.1:
            return SentimentCategory.BEARISH
        return SentimentCategory.NEUTRAL

    @property
    def sentiment_ratio(self) -> str:
        """Display ratio of positive/negative/neutral articles."""
        return f"+{self.positive_count} / -{self.negative_count} / ~{self.neutral_count}"

    def to_dict(self) -> dict:
        """Convert to dictionary for display."""
        return {
            "overall": self.category.value.replace("_", " ").title(),
            "score": f"{self.overall_score:+.2f}",
            "articles": self.article_count,
            "breakdown": self.sentiment_ratio,
        }
