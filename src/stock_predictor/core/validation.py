"""Data validation utilities for stock data quality checks."""

import logging
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

class DataQuality(Enum):
    """Data quality levels."""
    GOOD = "good"
    DEGRADED = "degraded"
    INSUFFICIENT = "insufficient"

@dataclass
class ValidationResult:
    """Result of data validation."""
    quality: DataQuality
    issues: list[str]
    usable_rows: int
    total_rows: int

    @property
    def is_usable(self) -> bool:
        return self.quality != DataQuality.INSUFFICIENT

def validate_ohlcv_data(data: pd.DataFrame, min_rows: int = 50) -> ValidationResult:
    """Validate OHLCV data for quality issues.

    Checks:
    - Minimum row count (default 50 for indicator warmup)
    - OHLC consistency (high >= low, high >= open/close, low <= open/close)
    - NaN/inf values in critical columns
    - Zero or negative prices
    - Duplicate timestamps
    - Large gaps in data (>5 trading days)
    - Frozen prices (same OHLC for 5+ consecutive days)
    """
    issues = []
    required_cols = ["open", "high", "low", "close", "volume"]

    # Check required columns
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        return ValidationResult(
            quality=DataQuality.INSUFFICIENT,
            issues=[f"Missing columns: {missing}"],
            usable_rows=0,
            total_rows=len(data),
        )

    total_rows = len(data)

    if total_rows < min_rows:
        return ValidationResult(
            quality=DataQuality.INSUFFICIENT,
            issues=[f"Only {total_rows} rows, need {min_rows} minimum"],
            usable_rows=total_rows,
            total_rows=total_rows,
        )

    # Check for NaN/inf in price columns
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        nan_count = data[col].isna().sum()
        if nan_count > 0:
            issues.append(f"{col}: {nan_count} NaN values")
        inf_count = np.isinf(data[col].astype(float)).sum()
        if inf_count > 0:
            issues.append(f"{col}: {inf_count} inf values")

    # Check for zero/negative prices
    for col in price_cols:
        bad = (data[col] <= 0).sum()
        if bad > 0:
            issues.append(f"{col}: {bad} zero/negative values")

    # OHLC consistency checks (NaN-safe: dropna before comparison)
    valid_ohlc = data[price_cols].dropna()
    if len(valid_ohlc) > 0:
        inconsistent = (valid_ohlc["high"] < valid_ohlc["low"]).sum()
        if inconsistent > 0:
            issues.append(f"{inconsistent} rows where high < low")

        high_violations = (
            (valid_ohlc["high"] < valid_ohlc["open"]) | (valid_ohlc["high"] < valid_ohlc["close"])
        ).sum()
        if high_violations > 0:
            issues.append(f"{high_violations} rows where high < open or close")

        low_violations = (
            (valid_ohlc["low"] > valid_ohlc["open"]) | (valid_ohlc["low"] > valid_ohlc["close"])
        ).sum()
        if low_violations > 0:
            issues.append(f"{low_violations} rows where low > open or close")

    # Volume checks
    zero_vol = (data["volume"] == 0).sum()
    if zero_vol > total_rows * 0.1:
        issues.append(f"{zero_vol} zero-volume days ({zero_vol/total_rows:.0%})")

    # Duplicate index check
    if data.index.duplicated().any():
        dup_count = data.index.duplicated().sum()
        issues.append(f"{dup_count} duplicate timestamps")

    # Frozen price detection (same close for 5+ days)
    # Also check if volume is zero/constant (true halt vs thin trading)
    if len(data) >= 5:
        close_diff = data["close"].diff()
        frozen_streak = 0
        max_frozen = 0
        frozen_with_zero_vol = 0
        for i, d in enumerate(close_diff):
            if d == 0:
                frozen_streak += 1
                if i < len(data) and data["volume"].iloc[i] == 0:
                    frozen_with_zero_vol += 1
                max_frozen = max(max_frozen, frozen_streak)
            else:
                frozen_streak = 0
        if max_frozen >= 5:
            if frozen_with_zero_vol > max_frozen * 0.5:
                issues.append(
                    f"Likely trading halt: {max_frozen} frozen closes with zero volume"
                )
            else:
                issues.append(
                    f"Frozen prices detected: {max_frozen} consecutive identical closes "
                    "(thin trading, not halted)"
                )

    # Gap detection (if datetime index)
    if isinstance(data.index, pd.DatetimeIndex):
        gaps = data.index.to_series().diff()
        large_gaps = gaps[gaps > pd.Timedelta(days=7)]
        if len(large_gaps) > 0:
            issues.append(f"{len(large_gaps)} data gaps > 7 calendar days")

    # Determine quality
    usable = total_rows - sum(data[col].isna().sum() for col in price_cols)

    severe_issues = [i for i in issues if any(kw in i for kw in ["zero/negative", "high < low", "Missing", "inf"])]

    if severe_issues:
        quality = DataQuality.DEGRADED if usable >= min_rows else DataQuality.INSUFFICIENT
    elif len(issues) > 3:
        quality = DataQuality.DEGRADED
    else:
        quality = DataQuality.GOOD

    return ValidationResult(
        quality=quality,
        issues=issues,
        usable_rows=usable,
        total_rows=total_rows,
    )


def detect_circuit_breaker(data: pd.DataFrame) -> pd.Series:
    """Detect potential circuit breaker hits in Indian market data.

    NSE circuit breaker limits: 10%, 15%, 20% from previous close.

    Returns:
        Boolean Series where True indicates a potential circuit breaker day.
    """
    if len(data) < 2:
        return pd.Series([False] * len(data), index=data.index)

    prev_close = data["close"].shift(1)
    pct_change = ((data["close"] - prev_close) / prev_close).abs()

    # Circuit breaker thresholds
    CB_LOWER = 0.095  # ~10% (with small tolerance)
    CB_UPPER = 0.205  # ~20% (with small tolerance)

    # Also check if high == low (locked circuit)
    locked = data["high"] == data["low"]

    # Large move + locked price = almost certainly circuit breaker
    # Large move + very low volume relative to average = likely circuit breaker
    is_cb = (pct_change >= CB_LOWER) & (locked | (pct_change >= CB_UPPER))

    # First row can't be determined
    is_cb.iloc[0] = False

    return is_cb


def detect_stock_split(data: pd.DataFrame, threshold: float = 0.30) -> pd.Series:
    """Detect potential stock splits/bonuses in price data.

    A split typically shows as a ~50% (2:1), ~67% (3:1), or ~80% (5:1) drop.

    Returns:
        Boolean Series where True indicates a potential split/bonus day.
    """
    if len(data) < 2:
        return pd.Series([False] * len(data), index=data.index)

    prev_close = data["close"].shift(1)
    pct_change = (data["close"] - prev_close) / prev_close

    # Common split ratios cause these approximate changes
    # 2:1 -> -50%, 3:1 -> -67%, 5:1 -> -80%, 1:1 bonus -> -50%
    is_split = pct_change <= -threshold
    is_split.iloc[0] = False

    return is_split


def adjust_for_splits(data: pd.DataFrame, threshold: float = 0.30) -> pd.DataFrame:
    """Adjust historical prices for detected stock splits/bonuses.

    When a split is detected (e.g., 2:1 causes ~50% drop), all prices
    BEFORE the split are divided by the split ratio to create a continuous
    price series.

    Args:
        data: DataFrame with OHLCV data (must have 'close' column)
        threshold: Minimum percentage drop to consider as a split

    Returns:
        DataFrame with adjusted prices (original is not modified)
    """
    adjusted = data.copy()
    if len(adjusted) < 2:
        return adjusted

    price_cols = [c for c in ["open", "high", "low", "close"] if c in adjusted.columns]
    prev_close = adjusted["close"].shift(1)
    pct_change = (adjusted["close"] - prev_close) / prev_close

    # Find split days (large drops)
    split_mask = pct_change <= -threshold

    for idx in adjusted.index[split_mask]:
        pos = adjusted.index.get_loc(idx)
        if pos == 0:
            continue

        # Estimate split ratio from price change
        # Ratios: 5:1 (~80%), 3:1 (~67%), 3:2 (~33%), 2:1/1:1 bonus (~50%)
        drop = abs(pct_change.loc[idx])
        if drop >= 0.75:
            ratio = 5.0
        elif drop >= 0.60:
            ratio = 3.0
        elif drop >= 0.42:
            ratio = 2.0  # Covers 1:1 bonus (50%) with noise tolerance
        elif drop >= 0.30:
            ratio = 1.5  # 3:2 split
        else:
            continue

        logger.info(f"Adjusting for {ratio:.0f}:1 split at index {idx}")

        # Adjust all prices BEFORE the split date
        for col in price_cols:
            adjusted.iloc[:pos, adjusted.columns.get_loc(col)] /= ratio

        # Adjust volume inversely (more shares after split)
        if "volume" in adjusted.columns:
            adjusted.iloc[:pos, adjusted.columns.get_loc("volume")] *= int(ratio)

    return adjusted


def adjust_for_dividends(
    data: pd.DataFrame, dividend_dates: dict[str, float] | None = None
) -> pd.DataFrame:
    """Adjust historical prices for dividend ex-dates.

    On ex-dividend dates, the stock price typically drops by the dividend amount.
    This adjusts pre-dividend prices downward to create a continuous total-return series.

    Args:
        data: DataFrame with OHLCV data
        dividend_dates: Dict mapping ISO date strings to dividend amounts (INR per share).
                       If None, returns data unchanged.

    Returns:
        DataFrame with dividend-adjusted prices
    """
    adjusted = data.copy()
    if dividend_dates is None or not dividend_dates:
        return adjusted

    price_cols = [c for c in ["open", "high", "low", "close"] if c in adjusted.columns]

    for date_str, div_amount in sorted(dividend_dates.items(), reverse=True):
        # Find the bar corresponding to this ex-date
        if isinstance(adjusted.index, pd.DatetimeIndex):
            mask = adjusted.index.strftime("%Y-%m-%d") == date_str
        elif "timestamp" in adjusted.columns:
            mask = adjusted["timestamp"].astype(str).str[:10] == date_str
        else:
            continue

        if not mask.any():
            continue

        loc = adjusted.index.get_loc(adjusted.index[mask][0])

        if loc == 0 or div_amount <= 0:
            continue

        # Adjustment factor: (close_before - dividend) / close_before
        close_before = float(adjusted["close"].iloc[loc - 1])
        if close_before <= 0:
            continue
        factor = (close_before - div_amount) / close_before

        # Multiply all pre-dividend prices by this factor
        for col in price_cols:
            adjusted.iloc[:loc, adjusted.columns.get_loc(col)] *= factor

    return adjusted
