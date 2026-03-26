"""NSE market holidays calendar and data validation utilities.

Provides holiday detection for the National Stock Exchange of India,
data freshness validation, and corporate action awareness.
"""

import logging
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# All date defaults must use IST regardless of server timezone
_IST = ZoneInfo("Asia/Kolkata")

# NSE holidays for 2025 and 2026.
# Update annually from: https://www.nseindia.com/regulations/holiday-list
NSE_HOLIDAYS = {
    2025: [
        date(2025, 1, 26),   # Republic Day
        date(2025, 2, 26),   # Maha Shivaratri
        date(2025, 3, 14),   # Holi
        date(2025, 3, 31),   # Id-Ul-Fitr (Ramadan)
        date(2025, 4, 10),   # Shri Mahavir Jayanti
        date(2025, 4, 14),   # Dr. Baba Saheb Ambedkar Jayanti
        date(2025, 4, 18),   # Good Friday
        date(2025, 5, 1),    # Maharashtra Day
        date(2025, 5, 12),   # Buddha Purnima
        date(2025, 6, 7),    # Bakri Id
        date(2025, 7, 6),    # Moharram
        date(2025, 8, 15),   # Independence Day
        date(2025, 8, 16),   # Ashura
        date(2025, 8, 27),   # Janmashtami
        date(2025, 9, 5),    # Milad-un-Nabi (Birthday of Prophet Mohammad)
        date(2025, 10, 2),   # Mahatma Gandhi Jayanti / Dussehra
        date(2025, 10, 21),  # Diwali (Laxmi Puja)
        date(2025, 10, 22),  # Diwali Balipratipada
        date(2025, 11, 5),   # Prakash Gurpurb Sri Guru Nanak Dev
        date(2025, 11, 26),  # Guru Nanak Jayanti
        date(2025, 12, 25),  # Christmas
    ],
    2026: [
        date(2026, 1, 26),   # Republic Day
        date(2026, 2, 17),   # Maha Shivaratri
        date(2026, 3, 3),    # Holi
        date(2026, 3, 20),   # Id-Ul-Fitr (Ramadan)
        date(2026, 3, 30),   # Shri Mahavir Jayanti
        date(2026, 4, 3),    # Good Friday
        date(2026, 4, 14),   # Dr. Baba Saheb Ambedkar Jayanti
        date(2026, 5, 1),    # Maharashtra Day
        date(2026, 5, 26),   # Buddha Purnima (fixed: was duplicate of May 1)
        date(2026, 5, 27),   # Bakri Id
        date(2026, 6, 25),   # Moharram
        date(2026, 8, 15),   # Independence Day
        date(2026, 8, 25),   # Milad-un-Nabi
        date(2026, 9, 4),    # Janmashtami
        date(2026, 10, 2),   # Mahatma Gandhi Jayanti
        date(2026, 10, 12),  # Dussehra
        date(2026, 11, 9),   # Diwali (Laxmi Puja)
        date(2026, 11, 10),  # Diwali Balipratipada
        date(2026, 11, 16),  # Guru Nanak Jayanti
        date(2026, 12, 25),  # Christmas
    ],
    # 2027: Provisional dates — verify from https://www.nseindia.com/regulations/holiday-list
    # Lunar holidays are approximated; fixed holidays are exact
    2027: [
        date(2027, 1, 26),   # Republic Day
        date(2027, 3, 8),    # Maha Shivaratri (approx)
        date(2027, 3, 22),   # Holi (approx)
        date(2027, 3, 10),   # Id-Ul-Fitr (approx)
        date(2027, 4, 2),    # Good Friday
        date(2027, 4, 14),   # Dr. Baba Saheb Ambedkar Jayanti
        date(2027, 5, 1),    # Maharashtra Day
        date(2027, 5, 14),   # Buddha Purnima (approx)
        date(2027, 5, 17),   # Bakri Id (approx)
        date(2027, 6, 15),   # Moharram (approx)
        date(2027, 8, 15),   # Independence Day
        date(2027, 8, 25),   # Janmashtami (approx)
        date(2027, 9, 15),   # Milad-un-Nabi (approx)
        date(2027, 10, 2),   # Mahatma Gandhi Jayanti
        date(2027, 10, 1),   # Dussehra (approx)
        date(2027, 10, 29),  # Diwali (approx)
        date(2027, 10, 30),  # Diwali Balipratipada (approx)
        date(2027, 11, 8),   # Guru Nanak Jayanti (approx)
        date(2027, 12, 25),  # Christmas
    ],
}


def is_nse_holiday(check_date: date | None = None) -> bool:
    """Check if a given date is an NSE holiday.

    Args:
        check_date: Date to check (defaults to today)

    Returns:
        True if the date is a market holiday
    """
    if check_date is None:
        check_date = datetime.now(_IST).date()

    year = check_date.year
    if year not in NSE_HOLIDAYS:
        logger.warning(
            f"No holiday data for {year}. Update market_holidays.py from "
            "https://www.nseindia.com/regulations/holiday-list. "
            "Assuming no holidays — market open/close detection may be wrong."
        )
    holidays = NSE_HOLIDAYS.get(year, [])
    return check_date in holidays


def is_trading_day(check_date: date | None = None) -> bool:
    """Check if a given date is a trading day.

    A trading day is a weekday that is not an NSE holiday.

    Args:
        check_date: Date to check (defaults to today)

    Returns:
        True if the market should be open on this date
    """
    if check_date is None:
        check_date = datetime.now(_IST).date()

    # Weekend check
    if check_date.weekday() >= 5:
        return False

    return not is_nse_holiday(check_date)


def get_last_trading_day(from_date: date | None = None) -> date:
    """Get the most recent trading day on or before the given date.

    Args:
        from_date: Starting date (defaults to today)

    Returns:
        Most recent trading day
    """
    if from_date is None:
        from_date = datetime.now(_IST).date()

    current = from_date
    while not is_trading_day(current):
        current -= timedelta(days=1)

    return current


def get_next_trading_day(from_date: date | None = None) -> date:
    """Get the next trading day after the given date.

    Args:
        from_date: Starting date (defaults to today)

    Returns:
        Next trading day
    """
    if from_date is None:
        from_date = datetime.now(_IST).date()

    current = from_date + timedelta(days=1)
    while not is_trading_day(current):
        current += timedelta(days=1)

    return current


def trading_days_between(start: date, end: date) -> int:
    """Count trading days between two dates (exclusive of end).

    Args:
        start: Start date
        end: End date

    Returns:
        Number of trading days
    """
    count = 0
    current = start
    while current < end:
        if is_trading_day(current):
            count += 1
        current += timedelta(days=1)
    return count


def validate_data_freshness(
    data_timestamp: datetime | None,
    max_staleness_hours: float = 18.0,
) -> tuple[bool, str]:
    """Validate that market data is fresh enough to use.

    Uses trading-day logic: staleness is measured in trading days, not
    calendar hours, to avoid false alerts over weekends and holidays.

    Args:
        data_timestamp: Timestamp of the most recent data point
        max_staleness_hours: Maximum acceptable age in trading hours

    Returns:
        Tuple of (is_fresh, message)
    """
    if data_timestamp is None:
        return False, "No data timestamp available"

    now = datetime.now(_IST)
    last_trading = get_last_trading_day(now.date())

    if isinstance(data_timestamp, datetime):
        data_date = data_timestamp.date()
    else:
        data_date = data_timestamp

    trading_days_stale = trading_days_between(data_date, now.date())

    if trading_days_stale <= 1:
        age_hours = (now - data_timestamp).total_seconds() / 3600
        return True, f"Data is {age_hours:.1f}h old (fresh)"

    if trading_days_stale <= 3:
        return False, (
            f"Data is {trading_days_stale} trading days old "
            f"(last update: {data_timestamp.strftime('%Y-%m-%d %H:%M')}). "
            f"Last trading day: {last_trading}"
        )

    return False, (
        f"Data is {trading_days_stale} trading days old — significantly stale. "
        f"Last update: {data_timestamp.strftime('%Y-%m-%d %H:%M')}"
    )


def detect_potential_split(
    data_close: list[float],
    threshold: float = 0.30,
) -> tuple[bool, int | None]:
    """Detect potential stock split from sudden price drops.

    A stock split typically causes a 30%+ overnight price drop without
    a corresponding volume spike. This checks for such anomalies.

    Threshold lowered to 0.30 to catch 3:2 splits (~33% drop) in addition
    to 2:1 (~50%), 3:1 (~67%), and 5:1 (~80%) splits.

    Args:
        data_close: List of closing prices (chronological)
        threshold: Minimum overnight change ratio to flag (0.30 = 30%)

    Returns:
        Tuple of (split_detected, index_of_split)
    """
    for i in range(1, len(data_close)):
        if data_close[i - 1] == 0:
            continue
        change_ratio = abs(data_close[i] - data_close[i - 1]) / data_close[i - 1]
        if change_ratio >= threshold:
            return True, i

    return False, None
