"""Yahoo Finance data provider for NSE stocks."""

import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from stock_predictor.core.entities.stock import OHLCV, IndexData, Quote, Stock, StockData
from stock_predictor.core.validation import adjust_for_dividends, adjust_for_splits
from stock_predictor.infrastructure.config.constants import YAHOO_INDEX_SYMBOLS

logger = logging.getLogger(__name__)


class YahooDataProvider:
    """Data provider using Yahoo Finance API."""

    NSE_SUFFIX = ".NS"
    BSE_SUFFIX = ".BO"

    IST = ZoneInfo("Asia/Kolkata")
    MAX_RETRIES = 3

    def __init__(self, timeout: int = 30) -> None:
        """Initialize Yahoo data provider.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout

    def _retry_fetch(self, fetch_fn, retries: int = 3) -> pd.DataFrame:
        """Retry a fetch function with exponential backoff.

        Args:
            fetch_fn: Callable that returns a DataFrame
            retries: Number of retries

        Returns:
            DataFrame from successful fetch

        Raises:
            Last exception if all retries fail
        """
        last_error = None
        for attempt in range(retries):
            try:
                result = fetch_fn()
                if result is not None and not result.empty:
                    return result
            except Exception as e:
                last_error = e
                wait = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(f"Fetch attempt {attempt + 1}/{retries} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
        raise last_error or ValueError("Fetch returned empty data after retries")

    @staticmethod
    def _validate_symbol(symbol: str) -> str:
        """Sanitize and validate an NSE symbol.

        Strips whitespace, uppercases, and rejects invalid characters.

        Args:
            symbol: Raw symbol string

        Returns:
            Cleaned symbol

        Raises:
            ValueError: If symbol contains invalid characters
        """
        cleaned = symbol.strip().upper()
        if not cleaned:
            raise ValueError("Empty symbol")
        # NSE symbols: alphanumeric, hyphens, ampersands (M&M), dots (.NS suffix)
        import re
        if not re.match(r'^[A-Z0-9&\-\.]+$', cleaned):
            raise ValueError(f"Invalid symbol characters in '{symbol}'")
        return cleaned

    def _get_yahoo_symbol(self, symbol: str, is_index: bool = False) -> str:
        """Convert NSE symbol to Yahoo Finance format.

        Args:
            symbol: NSE symbol (e.g., 'RELIANCE')
            is_index: Whether this is an index symbol

        Returns:
            Yahoo Finance compatible symbol
        """
        if is_index:
            return YAHOO_INDEX_SYMBOLS.get(symbol, f"^{symbol}")

        symbol = self._validate_symbol(symbol)

        # Check if already has suffix
        if symbol.endswith(self.NSE_SUFFIX) or symbol.endswith(self.BSE_SUFFIX):
            return symbol

        return f"{symbol}{self.NSE_SUFFIX}"

    def get_quote(self, symbol: str) -> Quote:
        """Get real-time quote for a stock.

        Args:
            symbol: NSE stock symbol

        Returns:
            Quote object with current price data
        """
        yahoo_symbol = self._get_yahoo_symbol(symbol)
        ticker = yf.Ticker(yahoo_symbol)

        try:
            info = ticker.fast_info
            history = ticker.history(period="2d")

            if history.empty:
                raise ValueError(f"No data available for {symbol}")

            last_row = history.iloc[-1]
            prev_close = history.iloc[-2]["Close"] if len(history) > 1 else last_row["Close"]

            last_price = Decimal(str(last_row["Close"]))
            change = last_price - Decimal(str(prev_close))
            change_percent = (change / Decimal(str(prev_close))) * 100 if prev_close else Decimal(0)

            vol = last_row["Volume"]
            volume = int(vol) if pd.notna(vol) else 0

            return Quote(
                symbol=symbol,
                last_price=last_price,
                change=change,
                change_percent=change_percent,
                open=Decimal(str(last_row["Open"])),
                high=Decimal(str(last_row["High"])),
                low=Decimal(str(last_row["Low"])),
                close=Decimal(str(prev_close)),
                volume=volume,
                timestamp=datetime.now(self.IST),
            )
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            raise

    def get_historical(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Get historical OHLCV data.

        Args:
            symbol: NSE stock symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)
            start: Start date (overrides period if provided)
            end: End date (defaults to today)

        Returns:
            DataFrame with OHLCV data
        """
        yahoo_symbol = self._get_yahoo_symbol(symbol)
        ticker = yf.Ticker(yahoo_symbol)

        try:
            # Retry with exponential backoff for transient failures
            def _fetch():
                if start:
                    _end = end or datetime.now()
                    return ticker.history(start=start, end=_end, interval=interval)
                return ticker.history(period=period, interval=interval)

            df = self._retry_fetch(_fetch, retries=self.MAX_RETRIES)

            if df.empty:
                raise ValueError(f"No historical data available for {symbol}")

            # Rename columns to lowercase and standardize
            df = df.reset_index()
            df.columns = [col.lower() for col in df.columns]

            # Handle datetime index
            if "date" in df.columns:
                df = df.rename(columns={"date": "timestamp"})
            elif "datetime" in df.columns:
                df = df.rename(columns={"datetime": "timestamp"})

            # Ensure timestamp is datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Convert to IST then strip timezone for consistent naive timestamps
            if df["timestamp"].dt.tz is not None:
                df["timestamp"] = df["timestamp"].dt.tz_convert(self.IST).dt.tz_localize(None)

            # Select and order columns
            columns = ["timestamp", "open", "high", "low", "close", "volume"]
            df = df[columns]

            # Auto-detect and adjust for stock splits in historical data
            df = adjust_for_splits(df)

            # Adjust for dividends if available from yfinance
            try:
                dividends = ticker.dividends
                if dividends is not None and not dividends.empty:
                    div_dict = {
                        d.strftime("%Y-%m-%d"): float(v)
                        for d, v in dividends.items()
                        if v > 0
                    }
                    if div_dict:
                        df = adjust_for_dividends(df, div_dict)
            except Exception as div_err:
                logger.debug(f"Dividend adjustment skipped for {symbol}: {div_err}")

            return df

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            raise

    def get_stock_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> StockData:
        """Get complete stock data including quote and history.

        Args:
            symbol: NSE stock symbol
            period: Data period
            interval: Data interval

        Returns:
            StockData object with all available data
        """
        yahoo_symbol = self._get_yahoo_symbol(symbol)
        ticker = yf.Ticker(yahoo_symbol)

        try:
            # Get stock info
            info = ticker.info
            name = info.get("longName") or info.get("shortName") or symbol

            # Get historical data
            df = self.get_historical(symbol, period=period, interval=interval)

            # Get current quote
            quote = self.get_quote(symbol)

            return StockData(
                symbol=symbol,
                name=name,
                data=df,
                quote=quote,
                metadata={
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    "market_cap": info.get("marketCap"),
                    "pe_ratio": info.get("trailingPE"),
                    "pb_ratio": info.get("priceToBook"),
                    "dividend_yield": info.get("dividendYield"),
                    "52_week_high": info.get("fiftyTwoWeekHigh"),
                    "52_week_low": info.get("fiftyTwoWeekLow"),
                },
            )

        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {e}")
            raise

    def get_index_data(self, index: str) -> IndexData:
        """Get index data (Nifty 50, Bank Nifty, etc.).

        Args:
            index: Index identifier (NIFTY50, BANKNIFTY, etc.)

        Returns:
            IndexData object
        """
        yahoo_symbol = self._get_yahoo_symbol(index, is_index=True)
        ticker = yf.Ticker(yahoo_symbol)

        try:
            history = ticker.history(period="2d")

            if history.empty:
                raise ValueError(f"No data available for index {index}")

            last_row = history.iloc[-1]
            prev_close = history.iloc[-2]["Close"] if len(history) > 1 else last_row["Close"]

            value = Decimal(str(last_row["Close"]))
            change = value - Decimal(str(prev_close))
            change_percent = (change / Decimal(str(prev_close))) * 100 if prev_close else Decimal(0)

            # Get index name from constants or use symbol
            from stock_predictor.infrastructure.config.constants import NSE_INDICES
            name = NSE_INDICES.get(index, index)

            return IndexData(
                symbol=index,
                name=name,
                value=value,
                change=change,
                change_percent=change_percent,
                open=Decimal(str(last_row["Open"])),
                high=Decimal(str(last_row["High"])),
                low=Decimal(str(last_row["Low"])),
                previous_close=Decimal(str(prev_close)),
                timestamp=datetime.now(self.IST),
            )

        except Exception as e:
            logger.error(f"Error fetching index data for {index}: {e}")
            raise

    def get_multiple_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        """Get quotes for multiple symbols efficiently.

        Args:
            symbols: List of NSE stock symbols

        Returns:
            Dictionary mapping symbols to Quote objects
        """
        results = {}
        yahoo_symbols = [self._get_yahoo_symbol(s) for s in symbols]

        try:
            # Use yfinance download for batch fetching
            data = yf.download(
                yahoo_symbols,
                period="2d",
                group_by="ticker",
                progress=False,
                threads=True,
            )

            for symbol, yahoo_symbol in zip(symbols, yahoo_symbols):
                try:
                    if len(symbols) == 1:
                        ticker_data = data
                    else:
                        ticker_data = data[yahoo_symbol] if yahoo_symbol in data.columns.levels[0] else None

                    if ticker_data is not None and not ticker_data.empty:
                        last_row = ticker_data.iloc[-1]
                        # Skip symbols with NaN price data
                        if pd.isna(last_row["Close"]):
                            logger.warning(f"No price data for {symbol} - skipping")
                            continue
                        prev_close = ticker_data.iloc[-2]["Close"] if len(ticker_data) > 1 else last_row["Close"]

                        last_price = Decimal(str(last_row["Close"]))
                        change = last_price - Decimal(str(prev_close))
                        change_percent = (change / Decimal(str(prev_close))) * 100 if prev_close else Decimal(0)

                        vol = last_row["Volume"]
                        volume = int(vol) if pd.notna(vol) else 0

                        results[symbol] = Quote(
                            symbol=symbol,
                            last_price=last_price,
                            change=change,
                            change_percent=change_percent,
                            open=Decimal(str(last_row["Open"])),
                            high=Decimal(str(last_row["High"])),
                            low=Decimal(str(last_row["Low"])),
                            close=Decimal(str(prev_close)),
                            volume=volume,
                            timestamp=datetime.now(self.IST),
                        )
                except Exception as e:
                    logger.warning(f"Error processing {symbol}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error fetching multiple quotes: {e}")

        return results

    def search_symbol(self, query: str) -> list[Stock]:
        """Search for stocks by name or symbol.

        Args:
            query: Search query

        Returns:
            List of matching Stock objects
        """
        # Yahoo Finance doesn't have a great search API
        # This is a basic implementation
        from stock_predictor.infrastructure.config.constants import NIFTY50_SYMBOLS

        results = []
        query_upper = query.upper()

        for symbol in NIFTY50_SYMBOLS:
            if query_upper in symbol:
                try:
                    yahoo_symbol = self._get_yahoo_symbol(symbol)
                    ticker = yf.Ticker(yahoo_symbol)
                    info = ticker.info

                    results.append(
                        Stock(
                            symbol=symbol,
                            name=info.get("longName") or info.get("shortName") or symbol,
                            exchange="NSE",
                            sector=info.get("sector"),
                            industry=info.get("industry"),
                        )
                    )
                except Exception:
                    results.append(
                        Stock(symbol=symbol, name=symbol, exchange="NSE")
                    )

        return results

    @staticmethod
    def is_market_open() -> bool:
        """Check if NSE market is currently open.

        Accounts for weekends AND NSE holidays.

        Returns:
            True if market is open, False otherwise
        """
        from stock_predictor.infrastructure.config.constants import (
            MARKET_CLOSE_HOUR,
            MARKET_CLOSE_MINUTE,
            MARKET_OPEN_HOUR,
            MARKET_OPEN_MINUTE,
        )
        from stock_predictor.infrastructure.config.market_holidays import is_trading_day

        # Always use IST regardless of system timezone
        ist = ZoneInfo("Asia/Kolkata")
        now = datetime.now(ist)

        # Check if today is a trading day (weekday + not a holiday)
        if not is_trading_day(now.date()):
            return False

        # Check market hours (9:15 AM to 3:30 PM IST)
        market_open = now.replace(
            hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0, microsecond=0
        )
        market_close = now.replace(
            hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0, microsecond=0
        )

        return market_open <= now <= market_close

    def validate_data(self, data: pd.DataFrame, symbol: str) -> tuple[bool, list[str]]:
        """Validate fetched data for quality issues.

        Checks for:
        - Staleness (data too old)
        - Potential stock splits (sudden 30%+ price changes)
        - Missing data (gaps in trading days)
        - Zero or negative prices

        Args:
            data: DataFrame with OHLCV data
            symbol: Stock symbol for context

        Returns:
            Tuple of (is_valid, list of warnings)
        """
        from stock_predictor.infrastructure.config.market_holidays import (
            detect_potential_split,
            validate_data_freshness,
        )

        warnings = []
        is_valid = True

        if data.empty:
            return False, [f"{symbol}: No data returned"]

        # Check data freshness
        if "timestamp" in data.columns:
            last_ts = data["timestamp"].iloc[-1]
            if isinstance(last_ts, pd.Timestamp):
                fresh, msg = validate_data_freshness(last_ts.to_pydatetime())
                if not fresh:
                    warnings.append(f"{symbol}: {msg}")

        # Check for potential stock splits
        closes = data["close"].tolist()
        split_detected, split_idx = detect_potential_split(closes)
        if split_detected:
            warnings.append(
                f"{symbol}: Potential stock split detected at index {split_idx}. "
                "Indicator calculations may be unreliable."
            )

        # Check for zero/negative prices
        if (data["close"] <= 0).any():
            warnings.append(f"{symbol}: Zero or negative prices detected")
            is_valid = False

        # Check for sufficient data
        if len(data) < 30:
            warnings.append(f"{symbol}: Only {len(data)} data points (need 30+ for indicators)")

        return is_valid, warnings
