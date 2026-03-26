"""Performance tracker for logging and evaluating signal outcomes over time.

Provides:
- Persistent logging of every signal generated
- Outcome tracking (did the signal hit target or stop?)
- Signal quality analytics (win rate by type, confidence calibration)
- Historical accuracy reporting
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TrackedSignal:
    """A signal that has been logged for tracking."""

    id: str
    symbol: str
    signal_type: str  # strong_buy, buy, hold, sell, strong_sell
    confidence: float
    entry_price: float
    target_price: float | None
    stop_loss: float | None
    style: str
    generated_at: str  # ISO format timestamp
    reasons: list[str]

    # Market regime at time of signal generation
    regime: str | None = None  # "trending", "sideways", "volatile", "unknown"

    # Outcome (filled in later)
    outcome: str | None = None  # "target_hit", "stop_hit", "expired", "pending"
    exit_price: float | None = None
    exit_date: str | None = None
    pnl_percent: float | None = None
    days_held: int | None = None


@dataclass
class SignalAccuracy:
    """Accuracy statistics for signals."""

    total_signals: int
    resolved_signals: int  # Signals that have reached target or stop
    pending_signals: int

    targets_hit: int
    stops_hit: int
    expired: int

    win_rate: float  # % of resolved signals that hit target
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float  # Total wins / Total losses

    # By signal type
    accuracy_by_type: dict[str, dict]
    # By confidence range
    accuracy_by_confidence: dict[str, dict]
    # Calibration warnings (e.g. "high confidence signals have low win rate")
    calibration_warnings: list[str] | None = None


class PerformanceTracker:
    """Tracks signal outcomes over time for accuracy measurement."""

    def __init__(self, data_dir: str | None = None) -> None:
        """Initialize performance tracker.

        Args:
            data_dir: Directory for storing signal logs
        """
        if data_dir is None:
            data_dir = os.path.join(
                os.environ.get("APP_DATA_DIR", "data"), "performance"
            )
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.signals_file = self.data_dir / "tracked_signals.json"
        self._signals: list[TrackedSignal] = self._load_signals()

    def _load_signals(self) -> list[TrackedSignal]:
        """Load tracked signals from disk."""
        if not self.signals_file.exists():
            return []

        try:
            with open(self.signals_file) as f:
                data = json.load(f)
            return [TrackedSignal(**s) for s in data]
        except Exception as e:
            logger.error(f"Error loading signals: {e}")
            return []

    def _save_signals(self) -> None:
        """Save tracked signals to disk."""
        try:
            data = [asdict(s) for s in self._signals]
            with open(self.signals_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving signals: {e}")

    def _is_duplicate(self, symbol: str, style: str, cooldown_minutes: int = 60) -> bool:
        """Check if a signal for this symbol+style was logged recently.

        Prevents logging the same signal multiple times when a user
        runs analyze repeatedly.

        Args:
            symbol: Stock symbol
            style: Trading style
            cooldown_minutes: Minimum minutes between signals for same symbol+style

        Returns:
            True if a recent signal exists (duplicate)
        """
        cutoff = datetime.now() - timedelta(minutes=cooldown_minutes)
        for s in reversed(self._signals):
            if s.symbol == symbol and s.style == style:
                try:
                    generated = datetime.fromisoformat(s.generated_at)
                    if generated >= cutoff:
                        return True
                except (ValueError, TypeError):
                    continue
                break  # Only check the most recent signal for this symbol+style
        return False

    def log_signal(
        self,
        symbol: str,
        signal_type: str,
        confidence: float,
        entry_price: float,
        target_price: float | None,
        stop_loss: float | None,
        style: str,
        reasons: list[str] | None = None,
        regime: str | None = None,
    ) -> str | None:
        """Log a new signal for tracking.

        Includes deduplication: won't log if the same symbol+style
        was logged within the last 60 minutes.

        Args:
            symbol: Stock symbol
            signal_type: Signal type (buy, strong_buy, etc.)
            confidence: Signal confidence 0-1
            entry_price: Price at signal generation
            target_price: Target price
            stop_loss: Stop loss price
            style: Trading style
            reasons: List of reasons for the signal
            regime: Market regime at time of signal ("trending", "sideways", "volatile")

        Returns:
            Signal ID for later reference, or None if deduplicated
        """
        # Deduplication: skip if same symbol+style logged recently
        if self._is_duplicate(symbol, style):
            logger.info(f"Signal for {symbol}/{style} already logged recently, skipping duplicate")
            return None

        signal_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        signal = TrackedSignal(
            id=signal_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            style=style,
            generated_at=datetime.now().isoformat(),
            reasons=reasons or [],
            regime=regime or "unknown",
            outcome="pending",
        )

        self._signals.append(signal)
        self._save_signals()
        return signal_id

    def update_outcome(
        self,
        signal_id: str,
        outcome: str,
        exit_price: float,
    ) -> None:
        """Update the outcome of a tracked signal.

        Args:
            signal_id: The signal ID returned by log_signal
            outcome: "target_hit", "stop_hit", or "expired"
            exit_price: Actual exit price
        """
        for signal in self._signals:
            if signal.id == signal_id:
                signal.outcome = outcome
                signal.exit_price = exit_price
                # Validate outcome is consistent with exit price
                if outcome == "target_hit" and signal.target_price:
                    if signal.signal_type in ("buy", "strong_buy") and exit_price < signal.target_price * 0.98:
                        logger.warning(f"Signal {signal_id}: target_hit but exit {exit_price} < target {signal.target_price}")
                    elif signal.signal_type in ("sell", "strong_sell") and exit_price > signal.target_price * 1.02:
                        logger.warning(f"Signal {signal_id}: target_hit but exit {exit_price} > target {signal.target_price}")
                if outcome == "stop_hit" and signal.stop_loss:
                    if signal.signal_type in ("buy", "strong_buy") and exit_price > signal.stop_loss * 1.02:
                        logger.warning(f"Signal {signal_id}: stop_hit but exit {exit_price} > stop {signal.stop_loss}")
                    elif signal.signal_type in ("sell", "strong_sell") and exit_price < signal.stop_loss * 0.98:
                        logger.warning(f"Signal {signal_id}: stop_hit but exit {exit_price} < stop {signal.stop_loss}")
                signal.exit_date = datetime.now().isoformat()
                # Deduct round-trip cost: STT 0.1% (sell only) + brokerage 0.06% + exchange/GST ~0.04%
                ROUND_TRIP_COST_PCT = 0.20
                gross_pnl = (exit_price / signal.entry_price - 1) * 100 if signal.entry_price > 0 else 0
                signal.pnl_percent = gross_pnl - ROUND_TRIP_COST_PCT
                generated = datetime.fromisoformat(signal.generated_at)
                signal.days_held = (datetime.now() - generated).days
                break

        self._save_signals()

    def check_pending_signals(
        self, current_prices: dict[str, float]
    ) -> list[tuple[str, str, float]]:
        """Check pending signals against current prices.

        Args:
            current_prices: Dictionary of symbol -> current price

        Returns:
            List of (signal_id, outcome, exit_price) for resolved signals
        """
        resolved = []

        for signal in self._signals:
            if signal.outcome != "pending":
                continue

            price = current_prices.get(signal.symbol)
            if price is None:
                continue

            # Check if signal has expired (use actual trading day calendar)
            generated = datetime.fromisoformat(signal.generated_at)
            from stock_predictor.infrastructure.config.market_holidays import trading_days_between
            trading_days = trading_days_between(generated.date(), datetime.now().date())
            max_trading_days = 3 if signal.style == "intraday" else 20
            if trading_days > max_trading_days:
                self.update_outcome(signal.id, "expired", price)
                resolved.append((signal.id, "expired", price))
                continue

            if signal.signal_type in ("buy", "strong_buy"):
                if signal.target_price and price >= signal.target_price:
                    self.update_outcome(signal.id, "target_hit", price)
                    resolved.append((signal.id, "target_hit", price))
                elif signal.stop_loss and price <= signal.stop_loss:
                    self.update_outcome(signal.id, "stop_hit", price)
                    resolved.append((signal.id, "stop_hit", price))

            elif signal.signal_type in ("sell", "strong_sell"):
                if signal.target_price and price <= signal.target_price:
                    self.update_outcome(signal.id, "target_hit", price)
                    resolved.append((signal.id, "target_hit", price))
                elif signal.stop_loss and price >= signal.stop_loss:
                    self.update_outcome(signal.id, "stop_hit", price)
                    resolved.append((signal.id, "stop_hit", price))

        return resolved

    def get_accuracy(self, symbol: str | None = None) -> SignalAccuracy:
        """Calculate signal accuracy statistics.

        Args:
            symbol: Optional filter by symbol

        Returns:
            SignalAccuracy with detailed statistics
        """
        signals = self._signals
        if symbol:
            signals = [s for s in signals if s.symbol == symbol]

        total = len(signals)
        resolved = [s for s in signals if s.outcome in ("target_hit", "stop_hit", "expired")]
        pending = [s for s in signals if s.outcome == "pending"]

        targets_hit = [s for s in resolved if s.outcome == "target_hit"]
        stops_hit = [s for s in resolved if s.outcome == "stop_hit"]
        expired = [s for s in resolved if s.outcome == "expired"]

        win_rate = len(targets_hit) / len(resolved) * 100 if resolved else 0

        wins_pct = [s.pnl_percent for s in targets_hit if s.pnl_percent is not None]
        losses_pct = [s.pnl_percent for s in stops_hit if s.pnl_percent is not None]

        avg_win = sum(wins_pct) / len(wins_pct) if wins_pct else 0
        avg_loss = sum(losses_pct) / len(losses_pct) if losses_pct else 0

        total_wins = sum(wins_pct) if wins_pct else 0
        total_losses = abs(sum(losses_pct)) if losses_pct else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 99.9

        # Accuracy by signal type
        accuracy_by_type: dict[str, dict] = {}
        for signal_type in ("strong_buy", "buy", "sell", "strong_sell"):
            type_signals = [s for s in resolved if s.signal_type == signal_type]
            if type_signals:
                type_wins = [s for s in type_signals if s.outcome == "target_hit"]
                accuracy_by_type[signal_type] = {
                    "total": len(type_signals),
                    "wins": len(type_wins),
                    "win_rate": len(type_wins) / len(type_signals) * 100,
                }

        # Accuracy by confidence range
        MIN_SAMPLES_FOR_CALIBRATION = 20  # Need at least 20 signals per bucket for reliable calibration
        accuracy_by_confidence: dict[str, dict] = {}
        for low, high, label in [
            (0, 0.3, "low (0-30%)"),
            (0.3, 0.5, "medium (30-50%)"),
            (0.5, 0.7, "high (50-70%)"),
            (0.7, 1.01, "very high (70%+)"),
        ]:
            range_signals = [
                s for s in resolved if low <= s.confidence < high
            ]
            if range_signals:
                range_wins = [s for s in range_signals if s.outcome == "target_hit"]
                bucket_stats: dict = {
                    "total": len(range_signals),
                    "wins": len(range_wins),
                    "win_rate": len(range_wins) / len(range_signals) * 100,
                }
                # Mark whether this bucket has enough samples for reliable calibration
                if bucket_stats["total"] < MIN_SAMPLES_FOR_CALIBRATION:
                    bucket_stats["calibrated"] = False
                    bucket_stats["warning"] = (
                        f"Only {bucket_stats['total']} samples, "
                        f"need {MIN_SAMPLES_FOR_CALIBRATION} for reliable calibration"
                    )
                else:
                    bucket_stats["calibrated"] = True
                accuracy_by_confidence[label] = bucket_stats

        # Confidence calibration check — warn if confidence levels are misleading
        calibration_warnings = []
        for label, stats in accuracy_by_confidence.items():
            if not stats.get("calibrated", False):
                calibration_warnings.append(
                    f"'{label}': {stats.get('warning', 'insufficient samples')}"
                )
            elif stats["total"] >= MIN_SAMPLES_FOR_CALIBRATION:
                if "very high" in label and stats["win_rate"] < 50:
                    calibration_warnings.append(
                        f"'{label}' signals have only {stats['win_rate']:.0f}% "
                        "win rate — confidence is miscalibrated"
                    )
                elif "low" in label and stats["win_rate"] > 70:
                    calibration_warnings.append(
                        f"'{label}' signals have {stats['win_rate']:.0f}% "
                        "win rate — these may be underconfident"
                    )

        # Monotonicity check: higher confidence should mean higher win rate.
        # If a lower bucket outperforms a higher bucket, calibration is broken.
        calibrated_buckets = [
            (label, stats) for label, stats in accuracy_by_confidence.items()
            if stats.get("calibrated", False)
        ]
        for i in range(len(calibrated_buckets) - 1):
            low_label, low_stats = calibrated_buckets[i]
            high_label, high_stats = calibrated_buckets[i + 1]
            if low_stats["win_rate"] > high_stats["win_rate"] + 5:
                calibration_warnings.append(
                    f"Non-monotonic: '{low_label}' ({low_stats['win_rate']:.0f}%) "
                    f"outperforms '{high_label}' ({high_stats['win_rate']:.0f}%) — "
                    "confidence scoring needs recalibration"
                )

        return SignalAccuracy(
            total_signals=total,
            resolved_signals=len(resolved),
            pending_signals=len(pending),
            targets_hit=len(targets_hit),
            stops_hit=len(stops_hit),
            expired=len(expired),
            win_rate=win_rate,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            profit_factor=profit_factor,
            accuracy_by_type=accuracy_by_type,
            accuracy_by_confidence=accuracy_by_confidence,
            calibration_warnings=calibration_warnings if calibration_warnings else None,
        )

    def get_recent_signals(self, days: int = 7, symbol: str | None = None) -> list[TrackedSignal]:
        """Get recently generated signals.

        Args:
            days: Look back N days
            symbol: Optional filter by symbol

        Returns:
            List of recent TrackedSignal objects
        """
        cutoff = datetime.now() - timedelta(days=days)
        signals = self._signals

        if symbol:
            signals = [s for s in signals if s.symbol == symbol]

        return [
            s for s in signals
            if datetime.fromisoformat(s.generated_at) >= cutoff
        ]

    def get_regime_performance(self) -> dict[str, dict]:
        """Get performance broken down by market regime.

        Analyzes signal outcomes grouped by the market regime that was active
        when each signal was generated. Useful for understanding which regimes
        produce the best signals.

        Returns:
            Dictionary mapping regime -> {signals, wins, win_rate, avg_return,
            avg_confidence}
        """
        regime_stats: dict[str, dict] = {}

        for signal in self._signals:
            regime = signal.regime or "unknown"
            if regime not in regime_stats:
                regime_stats[regime] = {
                    "signals": 0,
                    "wins": 0,
                    "total_return": 0.0,
                    "total_confidence": 0.0,
                }

            regime_stats[regime]["signals"] += 1

            if signal.pnl_percent is not None:
                if signal.pnl_percent > 0:
                    regime_stats[regime]["wins"] += 1
                regime_stats[regime]["total_return"] += signal.pnl_percent

            regime_stats[regime]["total_confidence"] += signal.confidence

        # Calculate averages
        for regime, stats in regime_stats.items():
            n = stats["signals"]
            stats["win_rate"] = stats["wins"] / n if n > 0 else 0
            stats["avg_return"] = stats["total_return"] / n if n > 0 else 0
            stats["avg_confidence"] = stats["total_confidence"] / n if n > 0 else 0

        return regime_stats
