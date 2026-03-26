"""Support and Resistance level calculations."""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class LevelType(Enum):
    """Type of price level."""

    SUPPORT = "support"
    RESISTANCE = "resistance"
    PIVOT = "pivot"


@dataclass
class PriceLevel:
    """A significant price level."""

    price: float
    level_type: LevelType
    strength: float  # 0.0 to 1.0, based on touches/tests
    touches: int  # Number of times price touched this level


@dataclass
class PivotPoints:
    """Classic pivot points."""

    pivot: float
    r1: float  # Resistance 1
    r2: float  # Resistance 2
    r3: float  # Resistance 3
    s1: float  # Support 1
    s2: float  # Support 2
    s3: float  # Support 3


@dataclass
class SupportResistanceAnalysis:
    """Complete support/resistance analysis."""

    pivot_points: PivotPoints
    key_levels: list[PriceLevel]
    nearest_support: float | None
    nearest_resistance: float | None
    current_price: float
    distance_to_support_pct: float | None
    distance_to_resistance_pct: float | None


class SupportResistanceCalculator:
    """Calculator for support and resistance levels."""

    def __init__(self, tolerance: float | None = None) -> None:
        """Initialize calculator.

        Args:
            tolerance: Price tolerance for level clustering. If None, uses
                adaptive tolerance based on price level.
        """
        self._fixed_tolerance = tolerance

    @staticmethod
    def _get_adaptive_tolerance(price: float) -> float:
        """Get price-tier adaptive tolerance for level clustering.

        Args:
            price: Price level to determine tolerance for

        Returns:
            Tolerance as a fraction (e.g. 0.02 = 2%)
        """
        if price < 50:
            return 0.04  # 4% for penny/micro stocks
        elif price < 500:
            return 0.03  # 3% for small/mid cap
        elif price < 2000:
            return 0.02  # 2% for large cap
        else:
            return 0.015  # 1.5% for mega cap

    @property
    def tolerance(self) -> float:
        """Default tolerance (used when no price context is available)."""
        return self._fixed_tolerance if self._fixed_tolerance is not None else 0.02

    def calculate_pivot_points(self, data: pd.DataFrame) -> PivotPoints:
        """Calculate classic pivot points from previous day data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            PivotPoints object
        """
        # Use the previous complete candle
        if len(data) < 2:
            last = data.iloc[-1]
        else:
            last = data.iloc[-2]

        high = float(last["high"])
        low = float(last["low"])
        close = float(last["close"])

        # Classic pivot point formula
        pivot = (high + low + close) / 3

        # Support and resistance levels
        r1 = (2 * pivot) - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)

        s1 = (2 * pivot) - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)

        return PivotPoints(
            pivot=pivot,
            r1=r1,
            r2=r2,
            r3=r3,
            s1=s1,
            s2=s2,
            s3=s3,
        )

    def find_local_extrema(
        self, data: pd.DataFrame, order: int = 5
    ) -> tuple[list[tuple[float, int]], list[tuple[float, int]]]:
        """Find local highs and lows in price data with their indices.

        Returns (price, index) tuples for recency weighting.

        Args:
            data: DataFrame with OHLCV data
            order: Number of points on each side to compare

        Returns:
            Tuple of (local_highs, local_lows) where each is a list of (price, index)
        """
        highs = data["high"].values
        lows = data["low"].values

        local_highs: list[tuple[float, int]] = []
        local_lows: list[tuple[float, int]] = []

        for i in range(order, len(data) - order):
            # Check for local high
            if all(highs[i] >= highs[i - order : i]) and all(
                highs[i] >= highs[i + 1 : i + order + 1]
            ):
                local_highs.append((highs[i], i))

            # Check for local low
            if all(lows[i] <= lows[i - order : i]) and all(
                lows[i] <= lows[i + 1 : i + order + 1]
            ):
                local_lows.append((lows[i], i))

        return local_highs, local_lows

    def cluster_levels(
        self, levels: list[tuple[float, int]], total_bars: int = 0
    ) -> list[PriceLevel]:
        """Cluster nearby price levels together with recency weighting.

        Uses adaptive tolerance based on price level unless a fixed
        tolerance was explicitly provided at initialization.

        Args:
            levels: List of (price, bar_index) tuples
            total_bars: Total number of bars in data (for recency calculation)

        Returns:
            List of clustered PriceLevel objects
        """
        if not levels:
            return []

        sorted_levels = sorted(levels, key=lambda x: x[0])
        clusters: list[list[tuple[float, int]]] = []
        current_cluster: list[tuple[float, int]] = [sorted_levels[0]]

        for price, idx in sorted_levels[1:]:
            # Check if within tolerance of cluster average
            cluster_avg = sum(p for p, _ in current_cluster) / len(current_cluster)
            # Use fixed tolerance if provided, otherwise adaptive
            tol = (
                self._fixed_tolerance
                if self._fixed_tolerance is not None
                else self._get_adaptive_tolerance(cluster_avg)
            )
            if cluster_avg > 0 and abs(price - cluster_avg) / cluster_avg <= tol:
                current_cluster.append((price, idx))
            else:
                clusters.append(current_cluster)
                current_cluster = [(price, idx)]

        clusters.append(current_cluster)

        # Convert to PriceLevel objects with recency-weighted strength
        result = []
        for cluster in clusters:
            avg_price = sum(p for p, _ in cluster) / len(cluster)
            touches = len(cluster)

            # Recency weighting: recent touches count more
            if total_bars > 0:
                recency_weights = []
                for _, idx in cluster:
                    # Linear decay: most recent bar = 1.0, oldest = 0.2
                    recency = 0.2 + 0.8 * (idx / total_bars)
                    recency_weights.append(recency)
                weighted_touches = sum(recency_weights)
                strength = min(1.0, weighted_touches / 4)
            else:
                strength = min(1.0, touches / 5)

            result.append(
                PriceLevel(
                    price=avg_price,
                    level_type=LevelType.SUPPORT,  # Will be updated later
                    strength=strength,
                    touches=touches,
                )
            )

        return result

    def find_key_levels(
        self, data: pd.DataFrame, lookback: int = 100, num_levels: int = 5
    ) -> list[PriceLevel]:
        """Find key support and resistance levels from price action.

        Args:
            data: DataFrame with OHLCV data
            lookback: Number of periods to analyze
            num_levels: Maximum number of levels to return

        Returns:
            List of key PriceLevel objects
        """
        # Use recent data
        recent_data = data.iloc[-lookback:] if len(data) > lookback else data
        current_price = float(data["close"].iloc[-1])

        # Find local extrema (returns (price, index) tuples)
        local_highs, local_lows = self.find_local_extrema(recent_data)
        total_bars = len(recent_data)

        # Cluster support levels (from lows) with recency weighting
        support_levels = self.cluster_levels(local_lows, total_bars)
        for level in support_levels:
            level.level_type = LevelType.SUPPORT

        # Cluster resistance levels (from highs) with recency weighting
        resistance_levels = self.cluster_levels(local_highs, total_bars)
        for level in resistance_levels:
            level.level_type = LevelType.RESISTANCE

        # Combine and sort by strength
        all_levels = support_levels + resistance_levels
        all_levels.sort(key=lambda x: x.strength, reverse=True)

        # Update level types based on current price
        for level in all_levels:
            if level.price < current_price:
                level.level_type = LevelType.SUPPORT
            else:
                level.level_type = LevelType.RESISTANCE

        return all_levels[:num_levels]

    def get_52_week_levels(self, data: pd.DataFrame) -> tuple[float, float]:
        """Get 52-week high and low.

        Args:
            data: DataFrame with OHLCV data (ideally 1 year)

        Returns:
            Tuple of (52_week_high, 52_week_low)
        """
        # Use last 252 trading days (approximately 1 year)
        yearly_data = data.iloc[-252:] if len(data) > 252 else data

        high_52w = float(yearly_data["high"].max())
        low_52w = float(yearly_data["low"].min())

        return high_52w, low_52w

    def analyze(self, data: pd.DataFrame) -> SupportResistanceAnalysis:
        """Complete support/resistance analysis.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            SupportResistanceAnalysis with all metrics
        """
        # NaN guard: drop rows with NaN in price columns before analysis
        clean_data = data.dropna(subset=["high", "low", "close"])
        if len(clean_data) < 2:
            # Not enough data after dropping NaNs
            current_price = float(data["close"].iloc[-1]) if not pd.isna(data["close"].iloc[-1]) else 0.0
            pivot = PivotPoints(pivot=current_price, r1=current_price, r2=current_price,
                                r3=current_price, s1=current_price, s2=current_price, s3=current_price)
            return SupportResistanceAnalysis(
                pivot_points=pivot, key_levels=[], nearest_support=None,
                nearest_resistance=None, current_price=current_price,
                distance_to_support_pct=None, distance_to_resistance_pct=None,
            )

        current_price = float(clean_data["close"].iloc[-1])
        pivot_points = self.calculate_pivot_points(clean_data)
        key_levels = self.find_key_levels(clean_data)

        # Find nearest support and resistance
        supports = [
            l.price for l in key_levels if l.level_type == LevelType.SUPPORT
        ]
        resistances = [
            l.price for l in key_levels if l.level_type == LevelType.RESISTANCE
        ]

        # Also include pivot points
        supports.extend([pivot_points.s1, pivot_points.s2, pivot_points.s3])
        resistances.extend([pivot_points.r1, pivot_points.r2, pivot_points.r3])

        # Filter to levels below/above current price
        supports = [s for s in supports if s < current_price]
        resistances = [r for r in resistances if r > current_price]

        nearest_support = max(supports) if supports else None
        nearest_resistance = min(resistances) if resistances else None

        # Calculate distances
        dist_to_support = None
        dist_to_resistance = None

        if nearest_support:
            dist_to_support = ((current_price - nearest_support) / current_price) * 100

        if nearest_resistance:
            dist_to_resistance = ((nearest_resistance - current_price) / current_price) * 100

        return SupportResistanceAnalysis(
            pivot_points=pivot_points,
            key_levels=key_levels,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            current_price=current_price,
            distance_to_support_pct=dist_to_support,
            distance_to_resistance_pct=dist_to_resistance,
        )

    def get_signal_strength(self, data: pd.DataFrame) -> float:
        """Calculate signal strength based on S/R levels.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Signal strength from -1.0 to 1.0
        """
        analysis = self.analyze(data)

        # Near support = bullish, near resistance = bearish
        if analysis.distance_to_support_pct is not None and analysis.distance_to_support_pct < 2:
            # Very close to support - bullish
            return 0.5 + (2 - analysis.distance_to_support_pct) / 4

        if analysis.distance_to_resistance_pct is not None and analysis.distance_to_resistance_pct < 2:
            # Very close to resistance - bearish
            return -0.5 - (2 - analysis.distance_to_resistance_pct) / 4

        # Between levels - neutral to slight bias
        if analysis.distance_to_support_pct and analysis.distance_to_resistance_pct:
            total_range = analysis.distance_to_support_pct + analysis.distance_to_resistance_pct
            position = analysis.distance_to_support_pct / total_range

            # 0.5 = middle, < 0.5 = closer to support, > 0.5 = closer to resistance
            return (0.5 - position) * 0.5

        return 0.0
