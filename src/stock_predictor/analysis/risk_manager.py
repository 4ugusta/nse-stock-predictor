"""Position sizing and portfolio risk management.

Provides:
- ATR-based position sizing (risk a fixed % of capital per trade)
- Portfolio correlation checks to avoid concentration risk
- Drawdown protection and circuit-breaker logic
- Maximum allocation limits per stock, sector, and portfolio
"""

import logging
from dataclasses import dataclass, field
from decimal import Decimal

import numpy as np
import pandas as pd

from stock_predictor.analysis.indicators.atr import ATRCalculator
from stock_predictor.infrastructure.config.constants import SECTOR_STOCKS

logger = logging.getLogger(__name__)

# Sector correlation matrix (simplified but more accurate than binary same-sector check)
# Keys are frozensets of sector pairs; values are estimated correlation coefficients.
SECTOR_CORRELATION: dict[tuple[str, str], float] = {
    ("banking", "nbfc"): 0.85,
    ("banking", "insurance"): 0.70,
    ("it", "it"): 0.75,       # intra-sector
    ("pharma", "pharma"): 0.65,
    ("auto", "auto"): 0.70,
    ("metal", "metal"): 0.80,
    ("energy", "energy"): 0.75,
    ("fmcg", "fmcg"): 0.60,
    ("realty", "banking"): 0.65,
    ("banking", "banking"): 0.80,
    ("infra", "infra"): 0.70,
    ("defence", "defence"): 0.70,
    ("chemicals", "chemicals"): 0.65,
    ("realty", "realty"): 0.75,
    ("infra", "realty"): 0.60,
    ("metal", "energy"): 0.55,
}
DEFAULT_SAME_SECTOR = 0.65
DEFAULT_CROSS_SECTOR = 0.20
# Correlation above this threshold is considered "high" for blocking
HIGH_CORRELATION_THRESHOLD = 0.60


def _get_sector_correlation(sector_a: str, sector_b: str) -> float:
    """Get estimated correlation between two sectors.

    Looks up the pair in SECTOR_CORRELATION (order-independent).
    Falls back to DEFAULT_SAME_SECTOR or DEFAULT_CROSS_SECTOR.
    """
    a = sector_a.lower()
    b = sector_b.lower()
    # Try both orderings
    corr = SECTOR_CORRELATION.get((a, b)) or SECTOR_CORRELATION.get((b, a))
    if corr is not None:
        return corr
    if a == b:
        return DEFAULT_SAME_SECTOR
    return DEFAULT_CROSS_SECTOR


@dataclass
class PositionSize:
    """Calculated position size for a trade."""

    symbol: str
    shares: int
    entry_price: float
    position_value: float
    stop_loss: float
    risk_per_share: float
    total_risk: float  # Max loss if stop hit
    allocation_percent: float  # % of total capital
    reason: str


@dataclass
class PortfolioPosition:
    """An existing portfolio position."""

    symbol: str
    shares: int
    entry_price: float
    current_price: float
    sector: str | None = None
    stop_loss: float | None = None  # Actual stop loss for risk tracking
    target_price: float | None = None  # Target price for R:R tracking
    entry_date: str | None = None  # ISO date string for position age

    @property
    def position_value(self) -> float:
        return self.shares * self.current_price

    @property
    def pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.shares

    @property
    def pnl_percent(self) -> float:
        return ((self.current_price / self.entry_price) - 1) * 100 if self.entry_price > 0 else 0

    @property
    def risk_percent(self) -> float:
        """Actual risk as % of entry price based on stop loss."""
        if self.stop_loss and self.entry_price > 0:
            return abs(self.entry_price - self.stop_loss) / self.entry_price * 100
        return 2.0  # Default assumption if no stop set


@dataclass
class RiskAssessment:
    """Risk assessment for a proposed trade."""

    can_trade: bool
    position_size: PositionSize | None
    warnings: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)


@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics."""

    total_value: float
    total_invested: float
    cash_available: float
    total_pnl: float
    total_pnl_percent: float
    num_positions: int
    sector_exposure: dict[str, float]  # sector -> % of portfolio
    largest_position_percent: float
    portfolio_heat: float  # Total risk as % of portfolio (sum of all position risks)
    warnings: list[str]


class RiskManager:
    """Portfolio risk manager with position sizing."""

    # Risk limits
    MAX_RISK_PER_TRADE: float = 2.0  # Max 2% of capital at risk per trade
    MAX_POSITION_SIZE: float = 15.0  # Max 15% of capital in one stock
    MAX_SECTOR_EXPOSURE: float = 35.0  # Max 35% in one sector
    MAX_PORTFOLIO_HEAT: float = 10.0  # Max 10% total portfolio risk
    MAX_POSITIONS: int = 15  # Max 15 open positions
    MAX_CORRELATED_POSITIONS: int = 3  # Max 3 highly correlated stocks

    def __init__(
        self,
        total_capital: float,
        risk_per_trade: float = 1.0,
        existing_positions: list[PortfolioPosition] | None = None,
    ) -> None:
        """Initialize risk manager.

        Args:
            total_capital: Total available capital in INR
            risk_per_trade: Risk per trade as % of capital (default 1%)
            existing_positions: Current portfolio positions
        """
        self.total_capital = total_capital
        self.risk_per_trade = min(risk_per_trade, self.MAX_RISK_PER_TRADE)
        self.positions = existing_positions or []
        self.atr_calc = ATRCalculator()
        self._sector_map = self._build_sector_map()

    def _build_sector_map(self) -> dict[str, str]:
        """Build symbol -> sector mapping."""
        sector_map = {}
        for sector, symbols in SECTOR_STOCKS.items():
            for symbol in symbols:
                sector_map[symbol] = sector
        return sector_map

    def _get_sector(self, symbol: str) -> str:
        """Get sector for a symbol."""
        return self._sector_map.get(symbol, "unknown")

    def _get_invested_value(self) -> float:
        """Get total value of existing positions."""
        return sum(p.position_value for p in self.positions)

    def _get_cash_available(self) -> float:
        """Get available cash for new trades."""
        return self.total_capital - self._get_invested_value()

    def _get_sector_exposure(self) -> dict[str, float]:
        """Calculate sector exposure as % of portfolio."""
        sector_values: dict[str, float] = {}
        for p in self.positions:
            sector = p.sector or self._get_sector(p.symbol)
            sector_values[sector] = sector_values.get(sector, 0) + p.position_value

        total = self._get_invested_value()
        if total == 0:
            return {}

        # Use invested value (not total_capital) so exposure reflects actual allocation
        return {s: (v / total) * 100 for s, v in sector_values.items()}

    def _get_portfolio_heat(self) -> float:
        """Calculate portfolio heat (total risk as % of capital).

        Uses actual stop_loss per position when available for accurate risk.
        Falls back to 2% estimate if stop_loss not tracked.

        Applies correlation adjustment: correlated positions compound risk.
        Uses sqrt(sum of correlated risk^2) instead of naive sum.
        """
        if not self.positions:
            return 0.0

        # Calculate per-position risk in INR
        position_risks = []
        for p in self.positions:
            risk_pct = p.risk_percent  # Uses actual stop if available, else 2%
            position_risks.append(p.position_value * (risk_pct / 100))

        # Group by sector for correlation adjustment
        sector_risk: dict[str, float] = {}
        for p, risk in zip(self.positions, position_risks):
            sector = p.sector or self._get_sector(p.symbol)
            sector_risk[sector] = sector_risk.get(sector, 0) + risk

        # Correlated sectors: use sqrt(sum of squares) which is more accurate
        # than naive sum (correlated positions don't fully compound)
        import math
        adjusted_risk = 0.0
        sectors = list(sector_risk.keys())
        for i, s1 in enumerate(sectors):
            for j, s2 in enumerate(sectors):
                corr = _get_sector_correlation(s1, s2)
                adjusted_risk += sector_risk[s1] * sector_risk[s2] * corr

        total_risk = math.sqrt(max(adjusted_risk, 0))

        return (total_risk / self.total_capital) * 100 if self.total_capital > 0 else 0

    def calculate_position_size(
        self,
        symbol: str,
        data: pd.DataFrame,
        is_long: bool = True,
        stop_atr_multiplier: float = 1.5,
    ) -> PositionSize:
        """Calculate position size using ATR-based risk management.

        Formula: Shares = (Capital * Risk%) / (ATR * Multiplier)

        This ensures each trade risks the same dollar amount regardless
        of the stock's volatility.

        Args:
            symbol: Stock symbol
            data: DataFrame with OHLCV data
            is_long: True for long, False for short
            stop_atr_multiplier: ATR multiplier for stop loss

        Returns:
            PositionSize with calculated shares and levels
        """
        atr_result = self.atr_calc.analyze(data)
        entry_price = float(data["close"].iloc[-1])

        risk_amount = self.total_capital * (self.risk_per_trade / 100)
        risk_per_share = atr_result.current_atr * stop_atr_multiplier

        if risk_per_share <= 0:
            return PositionSize(
                symbol=symbol,
                shares=0,
                entry_price=entry_price,
                position_value=0,
                stop_loss=entry_price,
                risk_per_share=0,
                total_risk=0,
                allocation_percent=0,
                reason="ATR is zero - cannot calculate position size",
            )

        shares = int(risk_amount / risk_per_share)

        # Enforce maximum position size
        max_shares_by_allocation = int(
            (self.total_capital * self.MAX_POSITION_SIZE / 100) / entry_price
        )
        shares = min(shares, max_shares_by_allocation)

        # Enforce available cash
        cash = self._get_cash_available()
        max_shares_by_cash = int(cash / entry_price)
        shares = min(shares, max_shares_by_cash)

        # Enforce liquidity cap — never take more than 5% of average daily volume
        MAX_PARTICIPATION = 0.05  # Max 5% of average daily volume
        if "volume" in data.columns:
            avg_volume = data["volume"].iloc[-20:].mean()
            if avg_volume > 0:
                max_shares_by_liquidity = int(avg_volume * MAX_PARTICIPATION)
                shares = min(shares, max_shares_by_liquidity)

        if shares <= 0:
            return PositionSize(
                symbol=symbol,
                shares=0,
                entry_price=entry_price,
                position_value=0,
                stop_loss=entry_price,
                risk_per_share=risk_per_share,
                total_risk=0,
                allocation_percent=0,
                reason="Insufficient capital or position too large",
            )

        position_value = shares * entry_price
        if is_long:
            stop_loss = entry_price - risk_per_share
        else:
            stop_loss = entry_price + risk_per_share

        total_risk = shares * risk_per_share
        allocation_pct = (position_value / self.total_capital) * 100

        return PositionSize(
            symbol=symbol,
            shares=shares,
            entry_price=entry_price,
            position_value=round(position_value, 2),
            stop_loss=round(stop_loss, 2),
            risk_per_share=round(risk_per_share, 2),
            total_risk=round(total_risk, 2),
            allocation_percent=round(allocation_pct, 2),
            reason=f"ATR-based sizing: {shares} shares, risking {self.risk_per_trade}% of capital",
        )

    def assess_trade(
        self,
        symbol: str,
        data: pd.DataFrame,
        is_long: bool = True,
    ) -> RiskAssessment:
        """Full risk assessment before entering a trade.

        Checks all risk limits and returns position size if approved.

        Args:
            symbol: Stock symbol
            data: DataFrame with OHLCV data
            is_long: True for long, False for short

        Returns:
            RiskAssessment with position size and warnings/blockers
        """
        warnings = []
        blockers = []

        # 1. Check number of positions
        if len(self.positions) >= self.MAX_POSITIONS:
            blockers.append(
                f"Maximum positions reached ({self.MAX_POSITIONS}). Close a position first."
            )

        # 2. Check if already holding this stock
        existing = [p for p in self.positions if p.symbol == symbol]
        if existing:
            warnings.append(f"Already holding {symbol} ({existing[0].shares} shares)")

        # 3. Check sector concentration
        sector = self._get_sector(symbol)
        sector_exposure = self._get_sector_exposure()
        current_sector_pct = sector_exposure.get(sector, 0)
        if current_sector_pct >= self.MAX_SECTOR_EXPOSURE:
            blockers.append(
                f"Sector '{sector}' exposure already at {current_sector_pct:.1f}% "
                f"(max {self.MAX_SECTOR_EXPOSURE}%)"
            )

        # 4. Check portfolio heat
        heat = self._get_portfolio_heat()
        if heat >= self.MAX_PORTFOLIO_HEAT:
            blockers.append(
                f"Portfolio heat at {heat:.1f}% (max {self.MAX_PORTFOLIO_HEAT}%). "
                "Reduce existing risk first."
            )
        elif heat >= self.MAX_PORTFOLIO_HEAT * 0.8:
            warnings.append(f"Portfolio heat elevated at {heat:.1f}%")

        # 5. Check available cash
        cash = self._get_cash_available()
        if cash <= 0:
            blockers.append("No cash available for new positions")

        # 6. Check for highly correlated positions (weighted sector correlation)
        highly_correlated = []
        for p in self.positions:
            existing_sector = p.sector or self._get_sector(p.symbol)
            corr = _get_sector_correlation(sector, existing_sector)
            if corr >= HIGH_CORRELATION_THRESHOLD:
                highly_correlated.append((p.symbol, existing_sector, corr))
        if len(highly_correlated) >= self.MAX_CORRELATED_POSITIONS:
            correlated_names = ", ".join(
                f"{sym} ({sec}, r={c:.2f})" for sym, sec, c in highly_correlated
            )
            blockers.append(
                f"Already holding {len(highly_correlated)} highly correlated stocks: "
                f"{correlated_names}. Correlation limit reached — close a position first."
            )
        elif highly_correlated:
            correlated_names = ", ".join(
                f"{sym} (r={c:.2f})" for sym, _, c in highly_correlated
            )
            warnings.append(
                f"Correlated with existing positions: {correlated_names}"
            )

        # Calculate position size
        position_size = self.calculate_position_size(symbol, data, is_long)

        if position_size.shares == 0:
            blockers.append(position_size.reason)

        can_trade = len(blockers) == 0

        return RiskAssessment(
            can_trade=can_trade,
            position_size=position_size if can_trade else None,
            warnings=warnings,
            blockers=blockers,
        )

    def get_portfolio_risk(self) -> PortfolioRisk:
        """Calculate portfolio-level risk metrics.

        Returns:
            PortfolioRisk with all portfolio metrics
        """
        total_invested = self._get_invested_value()
        cash = self._get_cash_available()
        total_pnl = sum(p.pnl for p in self.positions)
        total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0

        sector_exposure = self._get_sector_exposure()
        portfolio_heat = self._get_portfolio_heat()

        largest_pct = 0
        for p in self.positions:
            pct = (p.position_value / self.total_capital) * 100
            largest_pct = max(largest_pct, pct)

        warnings = []
        if portfolio_heat > self.MAX_PORTFOLIO_HEAT * 0.7:
            warnings.append(f"Portfolio heat elevated: {portfolio_heat:.1f}%")
        if largest_pct > self.MAX_POSITION_SIZE * 0.8:
            warnings.append(f"Largest position is {largest_pct:.1f}% of portfolio")
        for sector, pct in sector_exposure.items():
            if pct > self.MAX_SECTOR_EXPOSURE * 0.8:
                warnings.append(f"High {sector} exposure: {pct:.1f}%")
        if cash < self.total_capital * 0.1:
            warnings.append(f"Low cash reserve: {cash / self.total_capital * 100:.1f}%")

        return PortfolioRisk(
            total_value=self.total_capital,
            total_invested=round(total_invested, 2),
            cash_available=round(cash, 2),
            total_pnl=round(total_pnl, 2),
            total_pnl_percent=round(total_pnl_pct, 2),
            num_positions=len(self.positions),
            sector_exposure=sector_exposure,
            largest_position_percent=round(largest_pct, 2),
            portfolio_heat=round(portfolio_heat, 2),
            warnings=warnings,
        )

    def check_drawdown_circuit_breaker(
        self,
        peak_capital: float,
        max_drawdown_pct: float = 5.0,
        realized_pnl: float = 0.0,
    ) -> tuple[bool, str]:
        """Check if portfolio has hit maximum drawdown limit.

        A circuit breaker that stops all new trades when the portfolio
        has drawn down too much from its peak.

        Tracks peak portfolio value properly by including both unrealized P&L
        from active positions AND realized P&L from closed positions.

        Args:
            peak_capital: Highest portfolio value achieved
            max_drawdown_pct: Maximum allowed drawdown percentage
            realized_pnl: Cumulative realized P&L from closed positions

        Returns:
            Tuple of (should_stop_trading, message)
        """
        # Calculate actual current value including realized + unrealized P&L
        unrealized_pnl = sum(p.pnl for p in self.positions)
        current_value = self.total_capital + realized_pnl + unrealized_pnl
        # Use peak_capital as-is: the caller is responsible for tracking the peak.
        # Inflating peak with current_value here would mask drawdowns.
        drawdown = ((peak_capital - current_value) / peak_capital) * 100 if peak_capital > 0 else 0
        drawdown = max(0.0, drawdown)  # Drawdown can't be negative

        if drawdown >= max_drawdown_pct:
            return True, (
                f"CIRCUIT BREAKER: Portfolio drawdown is {drawdown:.1f}% "
                f"(limit: {max_drawdown_pct}%). No new trades until review."
            )

        if drawdown >= max_drawdown_pct * 0.7:
            return False, (
                f"WARNING: Portfolio drawdown is {drawdown:.1f}% "
                f"(approaching limit of {max_drawdown_pct}%). Reduce risk."
            )

        return False, f"Drawdown: {drawdown:.1f}% (within limits)"

    def update_trailing_stops(
        self,
        current_prices: dict[str, float],
        trail_percent: float = 5.0,
    ) -> list[str]:
        """Update trailing stops for positions in profit.

        For longs: moves stop UP (never down) as price increases.
        For shorts: moves stop DOWN (never up) as price decreases.

        Args:
            current_prices: Dictionary of symbol -> current price
            trail_percent: Trailing stop distance as % of current price

        Returns:
            List of messages about stop updates
        """
        updates = []
        for p in self.positions:
            if p.symbol not in current_prices:
                continue

            p.current_price = current_prices[p.symbol]

            # Only trail for positions in profit with an existing stop
            if p.stop_loss and p.pnl_percent > 0:
                # Detect direction: stop below entry = long, stop above entry = short
                is_long = p.stop_loss < p.entry_price

                if is_long:
                    new_stop = p.current_price * (1 - trail_percent / 100)
                    if new_stop > p.stop_loss:
                        old_stop = p.stop_loss
                        p.stop_loss = round(new_stop, 2)
                        updates.append(
                            f"{p.symbol}: Trailing stop raised "
                            f"{old_stop:.2f} -> {p.stop_loss:.2f}"
                        )
                else:
                    # Short position: trail stop downward
                    new_stop = p.current_price * (1 + trail_percent / 100)
                    if new_stop < p.stop_loss:
                        old_stop = p.stop_loss
                        p.stop_loss = round(new_stop, 2)
                        updates.append(
                            f"{p.symbol}: Trailing stop lowered "
                            f"{old_stop:.2f} -> {p.stop_loss:.2f}"
                        )

        return updates

    @staticmethod
    def calculate_rolling_correlation(
        data_a: pd.DataFrame,
        data_b: pd.DataFrame,
        window: int = 60,
    ) -> float:
        """Calculate rolling correlation between two stocks using actual returns.

        More accurate than static sector estimates — uses real price data.

        Args:
            data_a: DataFrame with 'close' column for stock A
            data_b: DataFrame with 'close' column for stock B
            window: Rolling window in trading days

        Returns:
            Current rolling correlation coefficient (-1 to 1)
        """
        if len(data_a) < window or len(data_b) < window:
            return 0.0

        returns_a = data_a["close"].pct_change().dropna()
        returns_b = data_b["close"].pct_change().dropna()

        # Proper date-based alignment using inner join
        if "timestamp" in data_a.columns and "timestamp" in data_b.columns:
            ra = pd.DataFrame({"ret_a": returns_a.values}, index=data_a["timestamp"].iloc[1:].values)
            rb = pd.DataFrame({"ret_b": returns_b.values}, index=data_b["timestamp"].iloc[1:].values)
            aligned = pd.concat([ra, rb], axis=1, join="inner").dropna()
            if len(aligned) < 20:
                return 0.0
            aligned = aligned.iloc[-window:]
            returns_a_vals = aligned["ret_a"].values
            returns_b_vals = aligned["ret_b"].values
        else:
            # Fallback: trim to same length from end
            min_len = min(len(returns_a), len(returns_b))
            returns_a_vals = returns_a.iloc[-min(window, min_len):].values
            returns_b_vals = returns_b.iloc[-min(window, min_len):].values

        if len(returns_a_vals) < 20:
            return 0.0

        corr = np.corrcoef(returns_a_vals, returns_b_vals)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0
