"""NSE India data provider for institutional flows, promoter holdings, and bulk deals.

Fetches data directly from NSE's internal API endpoints using requests.
NSE requires a session cookie from the homepage before allowing API access.

Data provided:
- FII/DII daily trading activity (buy/sell values)
- Promoter shareholding patterns (holding %, pledge %)
- Bulk and block deal data
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

# NSE rotates user-agent checks; a standard browser UA works
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
}


@dataclass
class FIIDIIActivity:
    """FII and DII daily trading activity."""

    date: str
    fii_buy_value: float  # In crores
    fii_sell_value: float
    fii_net_value: float
    dii_buy_value: float
    dii_sell_value: float
    dii_net_value: float

    @property
    def fii_net_positive(self) -> bool:
        return self.fii_net_value > 0

    @property
    def dii_net_positive(self) -> bool:
        return self.dii_net_value > 0


@dataclass
class PromoterHolding:
    """Promoter shareholding pattern for a stock."""

    symbol: str
    promoter_holding_pct: float  # Total promoter holding %
    promoter_pledge_pct: float  # % of promoter shares pledged
    public_holding_pct: float
    institution_holding_pct: float  # FII + DII + MF combined
    quarter: str  # e.g., "Dec 2025"

    @property
    def pledge_risk(self) -> str:
        """Assess pledge risk level."""
        if self.promoter_pledge_pct > 50:
            return "critical"
        if self.promoter_pledge_pct > 30:
            return "high"
        if self.promoter_pledge_pct > 10:
            return "moderate"
        return "low"


@dataclass
class BulkDeal:
    """A bulk or block deal on NSE."""

    symbol: str
    deal_date: str
    client_name: str
    deal_type: str  # "Buy" or "Sell"
    quantity: int
    price: float


@dataclass
class NSEInstitutionalData:
    """Combined institutional data for scoring."""

    fii_dii: list[FIIDIIActivity] = field(default_factory=list)
    promoter: PromoterHolding | None = None
    bulk_deals: list[BulkDeal] = field(default_factory=list)
    fetch_errors: list[str] = field(default_factory=list)


class NSEDataProvider:
    """Fetches institutional and corporate data from NSE India's API.

    NSE requires a valid session (cookies from the homepage) before
    allowing access to internal API endpoints. This provider handles
    the session management transparently.
    """

    BASE_URL = "https://www.nseindia.com"
    API_FII_DII = "/api/fiidiiTradeReact"
    API_SHAREHOLDING = "/api/corporat-information"
    API_BULK_DEALS = "/api/block-deal"

    def __init__(self, timeout: int = 15) -> None:
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)
        self._cookies_ts: float = 0
        self._cookie_max_age = 120  # Refresh cookies every 2 minutes

    def _ensure_cookies(self) -> None:
        """Hit NSE homepage to obtain session cookies.

        NSE invalidates cookies quickly, so we refresh periodically.
        """
        if time.time() - self._cookies_ts < self._cookie_max_age:
            return

        try:
            resp = self._session.get(
                self.BASE_URL,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            self._cookies_ts = time.time()
        except requests.RequestException as e:
            logger.warning(f"Failed to refresh NSE cookies: {e}")

    def _get(self, path: str, params: dict | None = None) -> dict | list:
        """Make an authenticated GET request to NSE API."""
        self._ensure_cookies()

        url = f"{self.BASE_URL}{path}"
        try:
            resp = self._session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"NSE API request failed ({path}): {e}")
            raise
        except ValueError as e:
            logger.error(f"NSE API returned non-JSON ({path}): {e}")
            raise

    def get_fii_dii_activity(self, days: int = 5) -> list[FIIDIIActivity]:
        """Get recent FII/DII trading activity.

        Args:
            days: Number of recent trading days to fetch (NSE returns ~current day)

        Returns:
            List of FIIDIIActivity, most recent first
        """
        try:
            data = self._get(self.API_FII_DII)
        except Exception as e:
            logger.error(f"Failed to fetch FII/DII data: {e}")
            return []

        results = []

        # NSE returns a list with category-wise data
        # Structure: [{"category": "FII/FPI *", "buyValue": ..., "sellValue": ...}, ...]
        if not isinstance(data, list):
            logger.warning(f"Unexpected FII/DII response format: {type(data)}")
            return []

        fii_buy = fii_sell = dii_buy = dii_sell = 0.0
        date_str = datetime.now().strftime("%d-%b-%Y")

        for entry in data:
            category = entry.get("category", "").upper()
            buy_val = _parse_float(entry.get("buyValue", 0))
            sell_val = _parse_float(entry.get("sellValue", 0))
            net_val = _parse_float(entry.get("netValue", 0))

            if "FII" in category or "FPI" in category:
                fii_buy += buy_val
                fii_sell += sell_val
                if entry.get("date"):
                    date_str = entry["date"]
            elif "DII" in category:
                dii_buy += buy_val
                dii_sell += sell_val

        if fii_buy > 0 or dii_buy > 0:
            results.append(
                FIIDIIActivity(
                    date=date_str,
                    fii_buy_value=fii_buy,
                    fii_sell_value=fii_sell,
                    fii_net_value=fii_buy - fii_sell,
                    dii_buy_value=dii_buy,
                    dii_sell_value=dii_sell,
                    dii_net_value=dii_buy - dii_sell,
                )
            )

        return results

    def get_promoter_holding(self, symbol: str) -> PromoterHolding | None:
        """Get latest promoter shareholding pattern for a stock.

        Args:
            symbol: NSE stock symbol (e.g., 'RELIANCE')

        Returns:
            PromoterHolding or None if unavailable
        """
        try:
            params = {"index": "equityShare", "symbol": symbol}
            data = self._get(self.API_SHAREHOLDING, params=params)
        except Exception as e:
            logger.error(f"Failed to fetch shareholding for {symbol}: {e}")
            return None

        if not isinstance(data, list) or not data:
            logger.warning(f"No shareholding data for {symbol}")
            return None

        # Get the latest quarter's data
        latest = data[0] if data else None
        if not latest:
            return None

        try:
            promoter_pct = _parse_float(latest.get("promoterAndPromoterGroup", 0))
            pledge_pct = _parse_float(latest.get("pledgedPercentage", 0))
            public_pct = _parse_float(latest.get("public", 0))
            institution_pct = _parse_float(
                latest.get("institutionsTotal", 0)
            ) or _parse_float(latest.get("total", 0)) - promoter_pct - public_pct

            quarter = latest.get("period", "Unknown")

            return PromoterHolding(
                symbol=symbol,
                promoter_holding_pct=promoter_pct,
                promoter_pledge_pct=pledge_pct,
                public_holding_pct=public_pct,
                institution_holding_pct=max(0, institution_pct),
                quarter=quarter,
            )
        except (KeyError, TypeError) as e:
            logger.error(f"Error parsing shareholding for {symbol}: {e}")
            return None

    def get_bulk_deals(self, symbol: str | None = None) -> list[BulkDeal]:
        """Get recent bulk/block deals.

        Args:
            symbol: Optional filter by symbol

        Returns:
            List of BulkDeal objects
        """
        try:
            data = self._get(self.API_BULK_DEALS)
        except Exception as e:
            logger.error(f"Failed to fetch bulk deals: {e}")
            return []

        if not isinstance(data, list):
            return []

        results = []
        for entry in data:
            deal_symbol = entry.get("symbol", "")
            if symbol and deal_symbol != symbol:
                continue

            try:
                results.append(
                    BulkDeal(
                        symbol=deal_symbol,
                        deal_date=entry.get("dealDate", ""),
                        client_name=entry.get("clientName", "Unknown"),
                        deal_type=entry.get("buySell", ""),
                        quantity=int(_parse_float(entry.get("quantity", 0))),
                        price=_parse_float(entry.get("tradedPrice", 0)),
                    )
                )
            except (KeyError, ValueError) as e:
                logger.warning(f"Skipping malformed bulk deal: {e}")

        return results

    def get_all_data(self, symbol: str) -> NSEInstitutionalData:
        """Fetch all available institutional data for a symbol.

        This is the main entry point — fetches FII/DII flows,
        promoter holdings, and bulk deals in one call.

        Args:
            symbol: NSE stock symbol

        Returns:
            NSEInstitutionalData with all fetched data and any errors
        """
        result = NSEInstitutionalData()

        # FII/DII (market-wide, not per-stock)
        try:
            result.fii_dii = self.get_fii_dii_activity()
        except Exception as e:
            result.fetch_errors.append(f"FII/DII: {e}")

        # Promoter holding (per-stock)
        try:
            result.promoter = self.get_promoter_holding(symbol)
        except Exception as e:
            result.fetch_errors.append(f"Promoter holding: {e}")

        # Bulk deals (filter by symbol)
        try:
            result.bulk_deals = self.get_bulk_deals(symbol)
        except Exception as e:
            result.fetch_errors.append(f"Bulk deals: {e}")

        return result


def _parse_float(value: str | float | int | None) -> float:
    """Safely parse a numeric value from NSE response."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        # NSE sometimes returns comma-formatted numbers like "1,234.56"
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return 0.0
