"""Zerodha Kite Connect data provider for real-time NSE data."""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import pandas as pd
import pyotp
from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException

from stock_predictor.core.entities.stock import OHLCV, IndexData, Quote, StockData
from stock_predictor.infrastructure.config.settings import KiteSettings

logger = logging.getLogger(__name__)


class KiteDataProvider:
    """Data provider using Zerodha Kite Connect API.

    Requires a valid Kite Connect subscription and API credentials.
    See: https://kite.trade/docs/connect/v3/
    """

    def __init__(self, settings: KiteSettings) -> None:
        """Initialize Kite Connect provider.

        Args:
            settings: Kite Connect API settings
        """
        self.settings = settings
        self.kite = KiteConnect(api_key=settings.api_key)
        self._access_token: str | None = None
        self._instruments: dict[str, int] = {}  # symbol -> instrument_token mapping

    @property
    def is_authenticated(self) -> bool:
        """Check if we have a valid access token."""
        return self._access_token is not None

    def get_login_url(self) -> str:
        """Get the Kite Connect login URL.

        Returns:
            Login URL for user authentication
        """
        return self.kite.login_url()

    def authenticate_with_request_token(self, request_token: str) -> None:
        """Authenticate using request token from redirect.

        Args:
            request_token: Token received from login redirect
        """
        try:
            data = self.kite.generate_session(
                request_token,
                api_secret=self.settings.api_secret.get_secret_value(),
            )
            self._access_token = data["access_token"]
            self.kite.set_access_token(self._access_token)
            logger.info("Kite Connect authenticated successfully")
        except KiteException as e:
            logger.error(f"Kite authentication failed: {e}")
            raise

    def authenticate_with_totp(self) -> None:
        """Authenticate using TOTP (for automated login).

        This method attempts to automate the login process using TOTP.
        Note: This requires additional setup and may not work in all cases.
        """
        if not self.settings.totp_key.get_secret_value():
            raise ValueError("TOTP key not configured")

        # Generate TOTP
        totp = pyotp.TOTP(self.settings.totp_key.get_secret_value())
        otp = totp.now()

        logger.info(f"Generated TOTP: {otp}")
        logger.info("Please complete login at the URL and provide request_token")
        logger.info(f"Login URL: {self.get_login_url()}")

        # Note: Full automated login requires additional web scraping
        # which is not recommended. Use authenticate_with_request_token()
        # after manual login.

    def set_access_token(self, access_token: str) -> None:
        """Set access token directly.

        Args:
            access_token: Valid access token
        """
        self._access_token = access_token
        self.kite.set_access_token(access_token)

    def _load_instruments(self) -> None:
        """Load instrument list for token mapping."""
        if self._instruments:
            return

        try:
            instruments = self.kite.instruments("NSE")
            for inst in instruments:
                self._instruments[inst["tradingsymbol"]] = inst["instrument_token"]
            logger.info(f"Loaded {len(self._instruments)} NSE instruments")
        except KiteException as e:
            logger.error(f"Failed to load instruments: {e}")
            raise

    def _get_instrument_token(self, symbol: str) -> int:
        """Get instrument token for a symbol.

        Args:
            symbol: NSE trading symbol

        Returns:
            Instrument token for Kite API
        """
        if not self._instruments:
            self._load_instruments()

        if symbol not in self._instruments:
            raise ValueError(f"Unknown symbol: {symbol}")

        return self._instruments[symbol]

    def get_quote(self, symbol: str) -> Quote:
        """Get real-time quote for a stock.

        Args:
            symbol: NSE stock symbol

        Returns:
            Quote object with current price data
        """
        if not self.is_authenticated:
            raise RuntimeError("Not authenticated. Call authenticate first.")

        instrument = f"NSE:{symbol}"

        try:
            quotes = self.kite.quote([instrument])
            data = quotes[instrument]

            return Quote(
                symbol=symbol,
                last_price=Decimal(str(data["last_price"])),
                change=Decimal(str(data["net_change"])),
                change_percent=Decimal(str(data["net_change"] / data["ohlc"]["close"] * 100))
                if data["ohlc"]["close"]
                else Decimal(0),
                open=Decimal(str(data["ohlc"]["open"])),
                high=Decimal(str(data["ohlc"]["high"])),
                low=Decimal(str(data["ohlc"]["low"])),
                close=Decimal(str(data["ohlc"]["close"])),
                volume=data["volume"],
                timestamp=datetime.now(),
                bid=Decimal(str(data["depth"]["buy"][0]["price"])) if data["depth"]["buy"] else None,
                ask=Decimal(str(data["depth"]["sell"][0]["price"])) if data["depth"]["sell"] else None,
                bid_qty=data["depth"]["buy"][0]["quantity"] if data["depth"]["buy"] else None,
                ask_qty=data["depth"]["sell"][0]["quantity"] if data["depth"]["sell"] else None,
            )

        except KiteException as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            raise

    def get_historical(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "day",
    ) -> pd.DataFrame:
        """Get historical OHLCV data.

        Args:
            symbol: NSE stock symbol
            start: Start date
            end: End date
            interval: Candle interval (minute, 3minute, 5minute, 10minute,
                     15minute, 30minute, 60minute, day, week, month)

        Returns:
            DataFrame with OHLCV data
        """
        if not self.is_authenticated:
            raise RuntimeError("Not authenticated. Call authenticate first.")

        try:
            instrument_token = self._get_instrument_token(symbol)

            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=start,
                to_date=end,
                interval=interval,
            )

            if not data:
                raise ValueError(f"No historical data for {symbol}")

            df = pd.DataFrame(data)
            df = df.rename(columns={"date": "timestamp"})

            # Ensure timestamp is datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Remove timezone info
            if df["timestamp"].dt.tz is not None:
                df["timestamp"] = df["timestamp"].dt.tz_localize(None)

            return df[["timestamp", "open", "high", "low", "close", "volume"]]

        except KiteException as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            raise

    def get_historical_by_period(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "day",
    ) -> pd.DataFrame:
        """Get historical data by period string.

        Args:
            symbol: NSE stock symbol
            period: Period string (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y)
            interval: Candle interval

        Returns:
            DataFrame with OHLCV data
        """
        period_map = {
            "1d": timedelta(days=1),
            "5d": timedelta(days=5),
            "1mo": timedelta(days=30),
            "3mo": timedelta(days=90),
            "6mo": timedelta(days=180),
            "1y": timedelta(days=365),
            "2y": timedelta(days=730),
        }

        if period not in period_map:
            raise ValueError(f"Invalid period: {period}")

        end = datetime.now()
        start = end - period_map[period]

        return self.get_historical(symbol, start, end, interval)

    def get_stock_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "day",
    ) -> StockData:
        """Get complete stock data.

        Args:
            symbol: NSE stock symbol
            period: Data period
            interval: Candle interval

        Returns:
            StockData object
        """
        df = self.get_historical_by_period(symbol, period, interval)
        quote = self.get_quote(symbol)

        return StockData(
            symbol=symbol,
            name=symbol,  # Kite doesn't provide company name in quote
            data=df,
            quote=quote,
        )

    def get_index_data(self, index: str) -> IndexData:
        """Get index data.

        Args:
            index: Index symbol (NIFTY 50, NIFTY BANK, etc.)

        Returns:
            IndexData object
        """
        if not self.is_authenticated:
            raise RuntimeError("Not authenticated. Call authenticate first.")

        # Map common names to Kite symbols
        index_map = {
            "NIFTY50": "NIFTY 50",
            "BANKNIFTY": "NIFTY BANK",
            "NIFTYIT": "NIFTY IT",
        }

        kite_symbol = index_map.get(index, index)
        instrument = f"NSE:{kite_symbol}"

        try:
            quotes = self.kite.quote([instrument])
            data = quotes[instrument]

            value = Decimal(str(data["last_price"]))
            prev_close = Decimal(str(data["ohlc"]["close"]))
            change = value - prev_close
            change_percent = (change / prev_close * 100) if prev_close else Decimal(0)

            return IndexData(
                symbol=index,
                name=kite_symbol,
                value=value,
                change=change,
                change_percent=change_percent,
                open=Decimal(str(data["ohlc"]["open"])),
                high=Decimal(str(data["ohlc"]["high"])),
                low=Decimal(str(data["ohlc"]["low"])),
                previous_close=prev_close,
                timestamp=datetime.now(),
            )

        except KiteException as e:
            logger.error(f"Error fetching index data for {index}: {e}")
            raise

    def get_multiple_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        """Get quotes for multiple symbols.

        Args:
            symbols: List of NSE stock symbols

        Returns:
            Dictionary mapping symbols to Quote objects
        """
        if not self.is_authenticated:
            raise RuntimeError("Not authenticated. Call authenticate first.")

        instruments = [f"NSE:{s}" for s in symbols]

        try:
            quotes = self.kite.quote(instruments)

            results = {}
            for symbol in symbols:
                instrument = f"NSE:{symbol}"
                if instrument in quotes:
                    data = quotes[instrument]
                    results[symbol] = Quote(
                        symbol=symbol,
                        last_price=Decimal(str(data["last_price"])),
                        change=Decimal(str(data["net_change"])),
                        change_percent=Decimal(
                            str(data["net_change"] / data["ohlc"]["close"] * 100)
                        )
                        if data["ohlc"]["close"]
                        else Decimal(0),
                        open=Decimal(str(data["ohlc"]["open"])),
                        high=Decimal(str(data["ohlc"]["high"])),
                        low=Decimal(str(data["ohlc"]["low"])),
                        close=Decimal(str(data["ohlc"]["close"])),
                        volume=data["volume"],
                        timestamp=datetime.now(),
                    )

            return results

        except KiteException as e:
            logger.error(f"Error fetching multiple quotes: {e}")
            raise

    @staticmethod
    def is_market_open() -> bool:
        """Check if NSE market is currently open.

        Returns:
            True if market is open
        """
        from stock_predictor.infrastructure.config.constants import (
            MARKET_CLOSE_HOUR,
            MARKET_CLOSE_MINUTE,
            MARKET_OPEN_HOUR,
            MARKET_OPEN_MINUTE,
        )

        now = datetime.now()

        # Check if weekday
        if now.weekday() >= 5:
            return False

        # Check market hours
        market_open = now.replace(
            hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0, microsecond=0
        )
        market_close = now.replace(
            hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0, microsecond=0
        )

        return market_open <= now <= market_close
