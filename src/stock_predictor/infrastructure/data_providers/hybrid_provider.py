"""Hybrid data provider with automatic fallback."""

import logging
from datetime import datetime
from decimal import Decimal

import pandas as pd

from stock_predictor.core.entities.stock import IndexData, Quote, StockData
from stock_predictor.infrastructure.config.settings import get_settings
from stock_predictor.infrastructure.data_providers.kite_provider import KiteDataProvider
from stock_predictor.infrastructure.data_providers.yahoo_provider import YahooDataProvider

logger = logging.getLogger(__name__)


class HybridDataProvider:
    """Data provider that uses Kite Connect as primary with Yahoo Finance fallback.

    This provider attempts to use Kite Connect for real-time data when authenticated,
    and falls back to Yahoo Finance when Kite is unavailable or for historical data.
    """

    def __init__(
        self,
        kite_provider: KiteDataProvider | None = None,
        yahoo_provider: YahooDataProvider | None = None,
    ) -> None:
        """Initialize hybrid provider.

        Args:
            kite_provider: Optional Kite Connect provider (created from settings if None)
            yahoo_provider: Optional Yahoo Finance provider (created if None)
        """
        self.yahoo = yahoo_provider or YahooDataProvider()
        self.kite: KiteDataProvider | None = None
        self._last_data_source: str = "yahoo"  # Track which provider served the last request

        settings = get_settings()

        if kite_provider:
            self.kite = kite_provider
        elif settings.use_kite:
            self.kite = KiteDataProvider(settings.kite)

    @property
    def has_kite(self) -> bool:
        """Check if Kite provider is available and authenticated."""
        return self.kite is not None and self.kite.is_authenticated

    @property
    def primary_source(self) -> str:
        """Get name of the current primary data source."""
        return "Kite Connect" if self.has_kite else "Yahoo Finance"

    def get_quote(self, symbol: str) -> Quote:
        """Get real-time quote, trying Kite first then Yahoo.

        Args:
            symbol: NSE stock symbol

        Returns:
            Quote object
        """
        if self.has_kite:
            try:
                result = self.kite.get_quote(symbol)
                self._last_data_source = "kite"
                return result
            except Exception as e:
                logger.warning(
                    f"Kite quote failed for {symbol}, falling back to Yahoo: {e}. "
                    "Kite data may differ from Yahoo (real-time vs delayed)."
                )

        self._last_data_source = "yahoo"
        return self.yahoo.get_quote(symbol)

    def get_historical(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Get historical data, trying Kite first then Yahoo.

        Args:
            symbol: NSE stock symbol
            period: Data period (if start/end not provided)
            interval: Data interval
            start: Start date (optional)
            end: End date (optional)

        Returns:
            DataFrame with OHLCV data
        """
        # Map interval formats between providers
        kite_interval_map = {
            "1m": "minute",
            "5m": "5minute",
            "15m": "15minute",
            "30m": "30minute",
            "1h": "60minute",
            "1d": "day",
            "1wk": "week",
            "1mo": "month",
        }

        if self.has_kite:
            try:
                kite_interval = kite_interval_map.get(interval, "day")

                if start and end:
                    return self.kite.get_historical(symbol, start, end, kite_interval)
                else:
                    return self.kite.get_historical_by_period(symbol, period, kite_interval)

            except Exception as e:
                logger.warning(
                    f"Kite historical failed for {symbol}, falling back to Yahoo: {e}. "
                    "Note: Yahoo data may have different adjustment methodology."
                )

        self._last_data_source = "yahoo"
        return self.yahoo.get_historical(symbol, period, interval, start, end)

    def get_stock_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> StockData:
        """Get complete stock data.

        Args:
            symbol: NSE stock symbol
            period: Data period
            interval: Data interval

        Returns:
            StockData object
        """
        if self.has_kite:
            try:
                kite_interval = {"1d": "day", "1wk": "week"}.get(interval, "day")
                return self.kite.get_stock_data(symbol, period, kite_interval)
            except Exception as e:
                logger.warning(f"Kite stock data failed for {symbol}, falling back to Yahoo: {e}")

        return self.yahoo.get_stock_data(symbol, period, interval)

    def get_index_data(self, index: str) -> IndexData:
        """Get index data.

        Args:
            index: Index identifier

        Returns:
            IndexData object
        """
        if self.has_kite:
            try:
                return self.kite.get_index_data(index)
            except Exception as e:
                logger.warning(f"Kite index data failed for {index}, falling back to Yahoo: {e}")

        return self.yahoo.get_index_data(index)

    def get_multiple_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        """Get quotes for multiple symbols.

        Args:
            symbols: List of NSE stock symbols

        Returns:
            Dictionary mapping symbols to Quote objects
        """
        if self.has_kite:
            try:
                return self.kite.get_multiple_quotes(symbols)
            except Exception as e:
                logger.warning(f"Kite multiple quotes failed, falling back to Yahoo: {e}")

        return self.yahoo.get_multiple_quotes(symbols)

    def is_market_open(self) -> bool:
        """Check if market is open."""
        # Both providers have the same logic
        return self.yahoo.is_market_open()

    def authenticate_kite(self, request_token: str) -> None:
        """Authenticate Kite Connect with request token.

        Args:
            request_token: Token from Kite login redirect
        """
        if self.kite is None:
            settings = get_settings()
            if not settings.use_kite:
                raise RuntimeError("Kite Connect is not configured")
            self.kite = KiteDataProvider(settings.kite)

        self.kite.authenticate_with_request_token(request_token)
        logger.info(f"Kite authenticated. Primary source is now: {self.primary_source}")

    def set_kite_access_token(self, access_token: str) -> None:
        """Set Kite access token directly.

        Args:
            access_token: Valid access token
        """
        if self.kite is None:
            settings = get_settings()
            if not settings.use_kite:
                raise RuntimeError("Kite Connect is not configured")
            self.kite = KiteDataProvider(settings.kite)

        self.kite.set_access_token(access_token)
        logger.info(f"Kite access token set. Primary source is now: {self.primary_source}")

    def get_kite_login_url(self) -> str | None:
        """Get Kite Connect login URL if configured.

        Returns:
            Login URL or None if Kite is not configured
        """
        if self.kite:
            return self.kite.get_login_url()

        settings = get_settings()
        if settings.use_kite:
            temp_kite = KiteDataProvider(settings.kite)
            return temp_kite.get_login_url()

        return None
