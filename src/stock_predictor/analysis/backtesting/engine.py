"""Backtesting engine for validating signal strategies against historical data.

Provides:
- Walk-forward backtesting (no lookahead bias)
- Trade simulation with realistic slippage and commission
- Performance metrics (win rate, Sharpe, max drawdown, profit factor)
- Signal-level tracking for debugging and optimization
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum

import numpy as np
import pandas as pd

from stock_predictor.analysis.indicators.atr import ATRCalculator
from stock_predictor.analysis.signals.signal_generator import SignalGenerator
from stock_predictor.core.entities.signal import SignalType, TradingStyle
from stock_predictor.infrastructure.config.constants import RISK_FREE_RATE

logger = logging.getLogger(__name__)

# Indian market transaction costs (one-side rates)
# STT for delivery: 0.1% on SELL side only; buy side = 0% (as of 2024 Budget)
_STT_DELIVERY_SELL = 0.001   # 0.1% on sell side
_STT_DELIVERY_BUY = 0.0      # 0% on buy side
_BROKERAGE = 0.0003          # 0.03% typical discount broker
_EXCHANGE_TXN = 0.0000345    # NSE transaction charges
_GST = 0.18                  # 18% GST on brokerage + exchange charges
_SEBI_CHARGES = 0.000001     # Rs 10 per crore
_STAMP_DUTY = 0.00015        # 0.015% on buy side


def calculate_transaction_cost(price: float, quantity: int, is_buy: bool) -> float:
    """Calculate total transaction cost for Indian market.

    Includes STT (sell-only for delivery), brokerage, exchange charges, GST,
    SEBI charges, and stamp duty (buy side only).

    Args:
        price: Price per share
        quantity: Number of shares
        is_buy: True for buy side, False for sell side

    Returns:
        Total transaction cost in INR
    """
    turnover = price * quantity
    stt = turnover * (_STT_DELIVERY_BUY if is_buy else _STT_DELIVERY_SELL)
    brokerage = turnover * _BROKERAGE
    exchange = turnover * _EXCHANGE_TXN
    gst = (brokerage + exchange) * _GST
    sebi = turnover * _SEBI_CHARGES
    stamp = turnover * _STAMP_DUTY if is_buy else 0
    return stt + brokerage + exchange + gst + sebi + stamp


class TradeOutcome(Enum):
    """Outcome of a completed trade."""

    TARGET_HIT = "target_hit"
    STOP_HIT = "stop_hit"
    TIME_EXIT = "time_exit"  # Exited due to max holding period
    SIGNAL_EXIT = "signal_exit"  # Opposite signal generated


@dataclass
class BacktestTrade:
    """A single trade in the backtest."""

    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: datetime | None = None
    exit_price: float | None = None
    direction: str = "long"  # "long" or "short"
    stop_loss: float = 0
    target: float = 0
    shares: int = 1
    pnl: float = 0
    pnl_percent: float = 0
    outcome: TradeOutcome | None = None
    signal_type: str = ""
    signal_confidence: float = 0
    holding_days: int = 0


@dataclass
class BacktestResult:
    """Complete backtest results."""

    symbol: str
    style: str
    period: str
    start_date: datetime
    end_date: datetime

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0
    avg_win: float = 0
    avg_loss: float = 0
    largest_win: float = 0
    largest_loss: float = 0
    profit_factor: float = 0  # Gross profit / Gross loss
    avg_holding_days: float = 0

    # Return metrics
    total_return_pct: float = 0
    annualized_return: float = 0
    buy_and_hold_return: float = 0  # Comparison: just holding the stock

    # Risk metrics
    max_drawdown_pct: float = 0
    sharpe_ratio: float = 0
    sortino_ratio: float = 0
    calmar_ratio: float = 0  # Annualized return / max drawdown

    # Equity curve
    equity_curve: list[float] = field(default_factory=list)
    trades: list[BacktestTrade] = field(default_factory=list)

    # Signal quality
    strong_buy_accuracy: float = 0
    buy_accuracy: float = 0
    sell_accuracy: float = 0

    def summary(self) -> str:
        """Return a formatted summary string."""
        return (
            f"=== Backtest: {self.symbol} ({self.style}) ===\n"
            f"Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}\n"
            f"Total trades: {self.total_trades}\n"
            f"Win rate: {self.win_rate:.1f}%\n"
            f"Avg win: {self.avg_win:+.2f}% | Avg loss: {self.avg_loss:+.2f}%\n"
            f"Profit factor: {self.profit_factor:.2f}\n"
            f"Total return: {self.total_return_pct:+.2f}%\n"
            f"Buy & hold: {self.buy_and_hold_return:+.2f}%\n"
            f"Max drawdown: {self.max_drawdown_pct:.2f}%\n"
            f"Sharpe ratio: {self.sharpe_ratio:.2f}\n"
            f"Avg holding: {self.avg_holding_days:.1f} days"
        )


class BacktestEngine:
    """Walk-forward backtesting engine.

    Simulates trading by generating signals on historical data and tracking
    outcomes. Uses walk-forward analysis to avoid lookahead bias.
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        slippage_pct: float = 0.10,  # Raised from 0.05 — more realistic for Indian markets
        max_holding_days: int = 30,
        risk_per_trade_pct: float = 1.0,
    ) -> None:
        """Initialize backtest engine.

        Args:
            initial_capital: Starting capital in INR
            slippage_pct: Slippage as % of price (market impact)
            max_holding_days: Maximum days to hold a position before forced exit
            risk_per_trade_pct: Risk per trade as % of capital
        """
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct / 100
        self.max_holding_days = max_holding_days
        self.risk_per_trade_pct = risk_per_trade_pct / 100
        self.signal_generator = SignalGenerator()
        self.atr_calc = ATRCalculator()

    def _apply_slippage(
        self, price: float, is_buy: bool,
        volume: float = 0, avg_volume: float = 0,
    ) -> float:
        """Apply price-tier aware slippage model.

        Buys get worse (higher) price, sells get worse (lower) price.
        Slippage scales by:
        - Price tier: penny stocks (< Rs 50) get 3x, small-cap (< Rs 200) 2x, mid 1.5x
        - Volume: low liquidity increases slippage further
        """
        slip = self.slippage_pct

        # Price-tier multiplier (penny/small-cap stocks have wider spreads)
        if price < 50:
            slip *= 3.0
        elif price < 200:
            slip *= 2.0
        elif price < 1000:
            slip *= 1.5

        # Volume-based multiplier
        if avg_volume > 0 and volume > 0:
            vol_ratio = volume / avg_volume
            if vol_ratio < 0.5:
                slip *= 2.0  # Very low liquidity
            elif vol_ratio < 1.0:
                slip *= 1.5  # Below average

        if is_buy:
            return price * (1 + slip)
        return price * (1 - slip)

    def run(
        self,
        data: pd.DataFrame,
        symbol: str,
        style: TradingStyle = TradingStyle.SWING,
        min_signal_score: float = 0.3,
        lookback_window: int = 200,
    ) -> BacktestResult:
        """Run backtest on historical data.

        Walk-forward: at each bar, use only past data to generate signals.

        Args:
            data: Full historical DataFrame with OHLCV data
            symbol: Stock symbol
            style: Trading style
            min_signal_score: Minimum signal confidence to trade
            lookback_window: Bars of history needed for indicators

        Returns:
            BacktestResult with full performance metrics
        """
        if len(data) < lookback_window + 50:
            logger.warning(f"{symbol}: Insufficient data for backtest ({len(data)} bars)")
            return BacktestResult(
                symbol=symbol,
                style=style.value,
                period=f"{len(data)} bars",
                start_date=data["timestamp"].iloc[0],
                end_date=data["timestamp"].iloc[-1],
            )

        # Validate OHLC data integrity
        if (data["high"] < data["low"]).any():
            logger.warning(f"{symbol}: Invalid OHLC data detected (high < low)")
        if (data["volume"] <= 0).any():
            logger.warning(f"{symbol}: Zero/negative volume bars detected")

        capital = self.initial_capital
        peak_capital = capital
        trades: list[BacktestTrade] = []
        equity_curve = [capital]
        current_trade: BacktestTrade | None = None
        pending_entry: dict | None = None  # Deferred entry to avoid lookahead bias

        # Walk forward through data
        for i in range(lookback_window, len(data)):
            current_bar = data.iloc[i]

            # Validate OHLC before using for fills
            if (
                current_bar["high"] < current_bar["low"]
                or current_bar["open"] > current_bar["high"]
                or current_bar["open"] < current_bar["low"]
            ):
                continue  # Skip invalid bar

            window = data.iloc[: i + 1]
            current_price = float(current_bar["close"])
            current_date = current_bar["timestamp"]
            bar_open = float(current_bar["open"])

            # Fill pending entry at this bar's open (deferred from previous bar)
            if pending_entry is not None and current_trade is None:
                is_buy = pending_entry["direction"] == "long"
                entry_price = self._apply_slippage(bar_open, is_buy)
                shares = pending_entry["shares"]
                if shares > 0 and entry_price * shares <= capital:
                    capital -= calculate_transaction_cost(entry_price, shares, is_buy=True)
                    current_trade = BacktestTrade(
                        symbol=symbol,
                        entry_date=current_date,
                        entry_price=entry_price,
                        direction=pending_entry["direction"],
                        stop_loss=pending_entry["stop"],
                        target=pending_entry["target"],
                        shares=shares,
                        signal_type=pending_entry["signal_type"],
                        signal_confidence=pending_entry["confidence"],
                    )
                pending_entry = None

            # Check existing trade
            if current_trade is not None:
                current_trade.holding_days += 1
                high = float(current_bar["high"])
                low = float(current_bar["low"])

                bar_open = float(current_bar["open"])

                # --- LONG exit checks ---
                if current_trade.direction == "long" and low <= current_trade.stop_loss:
                    if bar_open < current_trade.stop_loss:
                        exit_price = self._apply_slippage(bar_open, False)
                    else:
                        exit_price = self._apply_slippage(current_trade.stop_loss, False)
                    current_trade.exit_price = exit_price
                    current_trade.exit_date = current_date
                    current_trade.outcome = TradeOutcome.STOP_HIT
                    pnl = (exit_price - current_trade.entry_price) * current_trade.shares
                    pnl -= calculate_transaction_cost(exit_price, current_trade.shares, is_buy=False)
                    current_trade.pnl = pnl
                    current_trade.pnl_percent = (exit_price / current_trade.entry_price - 1) * 100
                    capital += pnl
                    trades.append(current_trade)
                    current_trade = None

                elif current_trade.direction == "long" and high >= current_trade.target:
                    if bar_open > current_trade.target:
                        exit_price = self._apply_slippage(bar_open, False)
                    else:
                        exit_price = self._apply_slippage(current_trade.target, False)
                    current_trade.exit_price = exit_price
                    current_trade.exit_date = current_date
                    current_trade.outcome = TradeOutcome.TARGET_HIT
                    pnl = (exit_price - current_trade.entry_price) * current_trade.shares
                    pnl -= calculate_transaction_cost(exit_price, current_trade.shares, is_buy=False)
                    current_trade.pnl = pnl
                    current_trade.pnl_percent = (exit_price / current_trade.entry_price - 1) * 100
                    capital += pnl
                    trades.append(current_trade)
                    current_trade = None

                # --- SHORT exit checks ---
                elif current_trade.direction == "short" and high >= current_trade.stop_loss:
                    if bar_open > current_trade.stop_loss:
                        exit_price = self._apply_slippage(bar_open, True)
                    else:
                        exit_price = self._apply_slippage(current_trade.stop_loss, True)
                    current_trade.exit_price = exit_price
                    current_trade.exit_date = current_date
                    current_trade.outcome = TradeOutcome.STOP_HIT
                    pnl = (current_trade.entry_price - exit_price) * current_trade.shares
                    pnl -= calculate_transaction_cost(exit_price, current_trade.shares, is_buy=True)
                    current_trade.pnl = pnl
                    current_trade.pnl_percent = (current_trade.entry_price / exit_price - 1) * 100
                    capital += pnl
                    trades.append(current_trade)
                    current_trade = None

                elif current_trade.direction == "short" and low <= current_trade.target:
                    if bar_open < current_trade.target:
                        exit_price = self._apply_slippage(bar_open, True)
                    else:
                        exit_price = self._apply_slippage(current_trade.target, True)
                    current_trade.exit_price = exit_price
                    current_trade.exit_date = current_date
                    current_trade.outcome = TradeOutcome.TARGET_HIT
                    pnl = (current_trade.entry_price - exit_price) * current_trade.shares
                    pnl -= calculate_transaction_cost(exit_price, current_trade.shares, is_buy=True)
                    current_trade.pnl = pnl
                    current_trade.pnl_percent = (current_trade.entry_price / exit_price - 1) * 100
                    capital += pnl
                    trades.append(current_trade)
                    current_trade = None

                # --- Time exit (direction-aware) ---
                elif current_trade.holding_days >= self.max_holding_days:
                    is_long = current_trade.direction == "long"
                    exit_price = self._apply_slippage(current_price, not is_long)
                    current_trade.exit_price = exit_price
                    current_trade.exit_date = current_date
                    current_trade.outcome = TradeOutcome.TIME_EXIT
                    if is_long:
                        pnl = (exit_price - current_trade.entry_price) * current_trade.shares
                    else:
                        pnl = (current_trade.entry_price - exit_price) * current_trade.shares
                    pnl -= calculate_transaction_cost(exit_price, current_trade.shares, is_buy=not is_long)
                    current_trade.pnl = pnl
                    current_trade.pnl_percent = (
                        (exit_price / current_trade.entry_price - 1) * 100 if is_long
                        else (current_trade.entry_price / exit_price - 1) * 100
                    )
                    capital += pnl
                    trades.append(current_trade)
                    current_trade = None

            # Generate signal if no position and no pending entry
            if current_trade is None and pending_entry is None and i >= lookback_window:
                try:
                    signal = self.signal_generator.generate(window, symbol, style)

                    is_buy = (
                        signal.signal_type in (SignalType.BUY, SignalType.STRONG_BUY)
                        and signal.confidence >= min_signal_score
                    )
                    is_sell = (
                        signal.signal_type in (SignalType.SELL, SignalType.STRONG_SELL)
                        and signal.confidence >= min_signal_score
                    )

                    if is_buy or is_sell:
                        direction = "long" if is_buy else "short"

                        # ATR-based position sizing
                        atr_result = self.atr_calc.analyze(window)
                        risk_per_share = atr_result.current_atr * 1.5
                        risk_amount = capital * self.risk_per_trade_pct

                        # Estimate entry at current close for sizing (actual fill at next bar open)
                        est_entry = current_price
                        if risk_per_share > 0:
                            shares = max(1, int(risk_amount / risk_per_share))
                            max_shares = int((capital * 0.15) / est_entry) if est_entry > 0 else 1
                            shares = min(shares, max_shares)
                        else:
                            shares = max(1, int((capital * 0.05) / est_entry)) if est_entry > 0 else 1

                        if shares > 0:
                            if is_buy:
                                stop = float(signal.stop_loss) if signal.stop_loss else est_entry - risk_per_share
                                target = float(signal.target_price) if signal.target_price else est_entry + (risk_per_share * 2)
                            else:
                                stop = float(signal.stop_loss) if signal.stop_loss else est_entry + risk_per_share
                                target = float(signal.target_price) if signal.target_price else est_entry - (risk_per_share * 2)

                            # Defer entry to next bar's open to avoid lookahead bias
                            pending_entry = {
                                "direction": direction,
                                "shares": shares,
                                "stop": stop,
                                "target": target,
                                "signal_type": signal.signal_type.value,
                                "confidence": signal.confidence,
                            }
                except Exception as e:
                    logger.debug(f"Signal generation failed at bar {i}: {e}")

            # Track equity
            if current_trade is not None:
                unrealized = (current_price - current_trade.entry_price) * current_trade.shares
                equity_curve.append(capital + unrealized)
            else:
                equity_curve.append(capital)

            peak_capital = max(peak_capital, equity_curve[-1])

        # Close any remaining position at end (direction-aware)
        if current_trade is not None:
            exit_price = float(data["close"].iloc[-1])
            is_long = current_trade.direction == "long"
            current_trade.exit_price = exit_price
            current_trade.exit_date = data["timestamp"].iloc[-1]
            current_trade.outcome = TradeOutcome.TIME_EXIT
            if is_long:
                pnl = (exit_price - current_trade.entry_price) * current_trade.shares
            else:
                pnl = (current_trade.entry_price - exit_price) * current_trade.shares
            pnl -= calculate_transaction_cost(exit_price, current_trade.shares, is_buy=not is_long)
            current_trade.pnl = pnl
            current_trade.pnl_percent = (
                (exit_price / current_trade.entry_price - 1) * 100 if is_long
                else (current_trade.entry_price / exit_price - 1) * 100
            )
            capital += pnl
            trades.append(current_trade)

        return self._compute_results(
            symbol, style, data, trades, equity_curve, capital
        )

    def _compute_results(
        self,
        symbol: str,
        style: TradingStyle,
        data: pd.DataFrame,
        trades: list[BacktestTrade],
        equity_curve: list[float],
        final_capital: float,
    ) -> BacktestResult:
        """Compute performance metrics from trades."""
        result = BacktestResult(
            symbol=symbol,
            style=style.value,
            period=f"{len(data)} bars",
            start_date=data["timestamp"].iloc[0],
            end_date=data["timestamp"].iloc[-1],
            equity_curve=equity_curve,
            trades=trades,
        )

        if not trades:
            # Buy and hold comparison: use lookback window start, not hardcoded 200
            bh_start_idx = min(200, len(data) - 1)
            start_price = float(data["close"].iloc[bh_start_idx])
            end_price = float(data["close"].iloc[-1])
            result.buy_and_hold_return = ((end_price / start_price) - 1) * 100 if start_price > 0 else 0
            return result

        # Trade statistics
        result.total_trades = len(trades)
        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]
        result.winning_trades = len(winners)
        result.losing_trades = len(losers)
        result.win_rate = (len(winners) / len(trades)) * 100

        if winners:
            result.avg_win = np.mean([t.pnl_percent for t in winners])
            result.largest_win = max(t.pnl_percent for t in winners)
        if losers:
            result.avg_loss = np.mean([t.pnl_percent for t in losers])
            result.largest_loss = min(t.pnl_percent for t in losers)

        gross_profit = sum(t.pnl for t in winners)
        gross_loss = abs(sum(t.pnl for t in losers))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 99.9

        result.avg_holding_days = np.mean([t.holding_days for t in trades])

        # Return metrics
        result.total_return_pct = ((final_capital / self.initial_capital) - 1) * 100

        days = (data["timestamp"].iloc[-1] - data["timestamp"].iloc[0]).days
        years = days / 365.25
        if years > 0:
            result.annualized_return = (
                (final_capital / self.initial_capital) ** (1 / years) - 1
            ) * 100

        # Buy-and-hold: use first trade's entry date as baseline
        if trades:
            first_trade_date = trades[0].entry_date
            bh_mask = data["timestamp"] >= first_trade_date
            if bh_mask.any():
                start_price = float(data.loc[bh_mask, "close"].iloc[0])
            else:
                start_price = float(data["close"].iloc[min(200, len(data) - 1)])
        else:
            start_price = float(data["close"].iloc[min(200, len(data) - 1)])
        end_price = float(data["close"].iloc[-1])
        result.buy_and_hold_return = ((end_price / start_price) - 1) * 100 if start_price > 0 else 0

        # Risk metrics
        eq = np.array(equity_curve)
        if len(eq) > 1:
            peak = np.maximum.accumulate(eq)
            drawdown = (peak - eq) / peak * 100
            result.max_drawdown_pct = float(np.max(drawdown))

            # Daily returns for Sharpe calculation
            returns = np.diff(eq) / eq[:-1]
            if len(returns) > 0:
                # Use India's risk-free rate (10-year govt bond yield ~7%)
                daily_rf = RISK_FREE_RATE / 252
                excess_returns = returns - daily_rf
                if np.std(excess_returns) > 0:
                    result.sharpe_ratio = float(
                        (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
                    )

                # Sortino: penalize downside deviation below 0% (MAR = 0%)
                # Using 0% MAR is standard for Sortino — losing money is bad, period.
                target_return = 0.0
                downside_returns = returns[returns < target_return] - target_return
                downside_std = float(np.sqrt((downside_returns ** 2).mean())) if len(downside_returns) > 0 else 0.0
                if downside_std > 0:
                    result.sortino_ratio = float(
                        ((returns.mean() - target_return) / downside_std) * np.sqrt(252)
                    )

            if result.max_drawdown_pct > 0:
                result.calmar_ratio = result.annualized_return / result.max_drawdown_pct

        # Signal accuracy by type
        strong_buys = [t for t in trades if t.signal_type == "strong_buy"]
        buys = [t for t in trades if t.signal_type == "buy"]

        if strong_buys:
            result.strong_buy_accuracy = (
                len([t for t in strong_buys if t.pnl > 0]) / len(strong_buys) * 100
            )
        if buys:
            result.buy_accuracy = (
                len([t for t in buys if t.pnl > 0]) / len(buys) * 100
            )

        return result

    def run_multi_symbol(
        self,
        data_dict: dict[str, pd.DataFrame],
        style: TradingStyle = TradingStyle.SWING,
    ) -> dict[str, BacktestResult]:
        """Run backtest across multiple symbols.

        Args:
            data_dict: Dictionary mapping symbol to DataFrame
            style: Trading style

        Returns:
            Dictionary mapping symbol to BacktestResult
        """
        results = {}
        for symbol, data in data_dict.items():
            try:
                results[symbol] = self.run(data, symbol, style)
            except Exception as e:
                logger.error(f"Backtest failed for {symbol}: {e}")
        return results

    def monte_carlo_test(
        self,
        backtest_result: BacktestResult,
        n_simulations: int = 1000,
        confidence_level: float = 0.95,
    ) -> dict:
        """Monte Carlo significance test on backtest trade outcomes.

        Shuffles the order of trade P&Ls to estimate the distribution of
        possible outcomes, testing whether the strategy's edge is statistically
        significant or just lucky sequencing.

        Args:
            backtest_result: Result from a completed backtest
            n_simulations: Number of random reshuffles
            confidence_level: Significance level (e.g. 0.95 = 95%)

        Returns:
            Dictionary with Monte Carlo statistics
        """
        trades = backtest_result.trades
        if len(trades) < 10:
            return {"error": "Need at least 10 trades for Monte Carlo analysis"}

        trade_pnls = [t.pnl for t in trades]
        actual_total = sum(trade_pnls)

        sim_totals = []
        rng = np.random.default_rng(42)
        for _ in range(n_simulations):
            shuffled = rng.permutation(trade_pnls)
            # Track max drawdown on the shuffled sequence
            cum = np.cumsum(shuffled)
            peak = np.maximum.accumulate(cum)
            dd = peak - cum
            sim_totals.append(float(cum[-1]))

        sim_totals = np.array(sim_totals)
        percentile_low = np.percentile(sim_totals, (1 - confidence_level) * 100)
        percentile_high = np.percentile(sim_totals, confidence_level * 100)
        median = np.median(sim_totals)

        is_significant = actual_total > percentile_high or actual_total < percentile_low

        return {
            "actual_pnl": actual_total,
            "median_simulated_pnl": float(median),
            "percentile_5": float(np.percentile(sim_totals, 5)),
            "percentile_95": float(np.percentile(sim_totals, 95)),
            "is_significant": bool(is_significant),
            "n_simulations": n_simulations,
            "n_trades": len(trades),
            "p_value_approx": float(np.mean(sim_totals >= actual_total))
            if actual_total > 0
            else float(np.mean(sim_totals <= actual_total)),
        }

    def run_out_of_sample(
        self,
        data: pd.DataFrame,
        symbol: str,
        style: TradingStyle = TradingStyle.SWING,
        train_ratio: float = 0.7,
    ) -> dict:
        """Run out-of-sample validation with train/test split.

        Runs backtest on the first train_ratio% of data, then on the remaining
        test portion. Compares performance to detect overfitting.

        Args:
            data: Full historical DataFrame
            symbol: Stock symbol
            style: Trading style
            train_ratio: Fraction of data for in-sample (default 70%)

        Returns:
            Dictionary with in-sample, out-of-sample results and degradation analysis
        """
        split_idx = int(len(data) * train_ratio)

        if split_idx < 300 or (len(data) - split_idx) < 100:
            return {"error": "Insufficient data for train/test split"}

        train_data = data.iloc[:split_idx].copy()
        # Warmup window is prepended but not used for trade generation.
        # The engine's walk-forward loop starts at lookback_window (200),
        # so only data after index 200 in test_data generates trades.
        warmup_start = max(0, split_idx - 200)
        test_data = data.iloc[warmup_start:].copy()  # Warmup for indicators only

        train_result = self.run(train_data, symbol, style)
        test_result = self.run(test_data, symbol, style)

        # Calculate degradation
        wr_degradation = train_result.win_rate - test_result.win_rate
        return_degradation = train_result.total_return_pct - test_result.total_return_pct

        is_overfit = wr_degradation > 15 or return_degradation > train_result.total_return_pct * 0.5

        return {
            "in_sample": {
                "win_rate": train_result.win_rate,
                "total_return": train_result.total_return_pct,
                "sharpe": train_result.sharpe_ratio,
                "trades": train_result.total_trades,
                "max_drawdown": train_result.max_drawdown_pct,
            },
            "out_of_sample": {
                "win_rate": test_result.win_rate,
                "total_return": test_result.total_return_pct,
                "sharpe": test_result.sharpe_ratio,
                "trades": test_result.total_trades,
                "max_drawdown": test_result.max_drawdown_pct,
            },
            "degradation": {
                "win_rate_drop": round(wr_degradation, 1),
                "return_drop": round(return_degradation, 2),
                "is_likely_overfit": is_overfit,
            },
        }

    def aggregate_results(self, results: dict[str, BacktestResult]) -> dict:
        """Aggregate results across multiple symbols.

        Args:
            results: Dictionary of BacktestResult per symbol

        Returns:
            Dictionary with aggregate metrics
        """
        if not results:
            return {"error": "No results to aggregate"}

        all_trades = []
        for r in results.values():
            all_trades.extend(r.trades)

        total_trades = len(all_trades)
        if total_trades == 0:
            return {"total_trades": 0, "message": "No trades generated"}

        winners = [t for t in all_trades if t.pnl > 0]
        avg_returns = [r.total_return_pct for r in results.values()]
        avg_sharpe = [r.sharpe_ratio for r in results.values() if r.sharpe_ratio != 0]
        max_drawdowns = [r.max_drawdown_pct for r in results.values()]

        return {
            "symbols_tested": len(results),
            "total_trades": total_trades,
            "overall_win_rate": len(winners) / total_trades * 100,
            "avg_return": np.mean(avg_returns),
            "median_return": np.median(avg_returns),
            "avg_sharpe": np.mean(avg_sharpe) if avg_sharpe else 0,
            "avg_max_drawdown": np.mean(max_drawdowns),
            "worst_drawdown": max(max_drawdowns),
            "profitable_symbols": len([r for r in results.values() if r.total_return_pct > 0]),
            "losing_symbols": len([r for r in results.values() if r.total_return_pct <= 0]),
        }
