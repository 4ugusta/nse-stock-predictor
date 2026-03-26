"""Application configuration using Pydantic Settings."""

from pathlib import Path
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class KiteSettings(BaseSettings):
    """Zerodha Kite Connect settings."""

    model_config = SettingsConfigDict(env_prefix="KITE_")

    api_key: str = ""
    api_secret: SecretStr = SecretStr("")
    user_id: str = ""
    password: SecretStr = SecretStr("")
    totp_key: SecretStr = SecretStr("")
    redirect_url: str = "http://localhost:8000/callback"

    @property
    def is_configured(self) -> bool:
        """Check if Kite credentials are configured."""
        return bool(self.api_key and self.api_secret.get_secret_value())


class YahooSettings(BaseSettings):
    """Yahoo Finance settings."""

    model_config = SettingsConfigDict(env_prefix="YAHOO_")

    rate_limit_per_minute: int = 60
    timeout_seconds: int = 30


class NewsSettings(BaseSettings):
    """News provider settings."""

    model_config = SettingsConfigDict(env_prefix="NEWS_")

    cache_ttl_minutes: int = 15
    max_articles_per_source: int = 20


class AppSettings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="APP_",
        extra="ignore",
    )

    debug: bool = False
    log_level: str = "INFO"
    data_dir: Path = Field(default=Path("data"))
    cache_ttl_minutes: int = 5

    # Nested settings
    kite: KiteSettings = Field(default_factory=KiteSettings)
    yahoo: YahooSettings = Field(default_factory=YahooSettings)
    news: NewsSettings = Field(default_factory=NewsSettings)

    def model_post_init(self, __context: object) -> None:
        """Create data directory if it doesn't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "cache").mkdir(exist_ok=True)
        (self.data_dir / "watchlists").mkdir(exist_ok=True)

    @property
    def use_kite(self) -> bool:
        """Check if Kite Connect should be used."""
        return self.kite.is_configured

    @property
    def cache_dir(self) -> Path:
        """Get cache directory path."""
        return self.data_dir / "cache"

    @property
    def watchlist_dir(self) -> Path:
        """Get watchlist directory path."""
        return self.data_dir / "watchlists"


@lru_cache
def get_settings() -> AppSettings:
    """Get cached application settings singleton."""
    return AppSettings()
