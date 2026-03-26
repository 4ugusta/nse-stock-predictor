"""Price alert storage and management."""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Type of price alert."""

    ABOVE = "above"  # Trigger when price goes above target
    BELOW = "below"  # Trigger when price goes below target


class AlertStatus(Enum):
    """Status of an alert."""

    ACTIVE = "active"
    TRIGGERED = "triggered"
    EXPIRED = "expired"


@dataclass
class PriceAlert:
    """A price alert for a stock."""

    id: str
    symbol: str
    alert_type: AlertType
    target_price: float
    created_at: datetime
    status: AlertStatus = AlertStatus.ACTIVE
    note: str = ""
    triggered_at: Optional[datetime] = None
    triggered_price: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "alert_type": self.alert_type.value,
            "target_price": self.target_price,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "note": self.note,
            "triggered_at": self.triggered_at.isoformat() if self.triggered_at else None,
            "triggered_price": self.triggered_price,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PriceAlert":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            symbol=data["symbol"],
            alert_type=AlertType(data["alert_type"]),
            target_price=data["target_price"],
            created_at=datetime.fromisoformat(data["created_at"]),
            status=AlertStatus(data["status"]),
            note=data.get("note", ""),
            triggered_at=datetime.fromisoformat(data["triggered_at"]) if data.get("triggered_at") else None,
            triggered_price=data.get("triggered_price"),
        )

    def check(self, current_price: float) -> bool:
        """Check if alert should be triggered.

        Args:
            current_price: Current stock price

        Returns:
            True if alert is triggered
        """
        if self.status != AlertStatus.ACTIVE:
            return False

        if self.alert_type == AlertType.ABOVE:
            return current_price >= self.target_price
        else:  # BELOW
            return current_price <= self.target_price


class AlertManager:
    """Manages price alerts with file-based storage."""

    def __init__(self, storage_dir: Path | None = None) -> None:
        """Initialize alert manager.

        Args:
            storage_dir: Directory to store alerts (uses default data dir if not provided)
        """
        if storage_dir is None:
            from stock_predictor.infrastructure.config.settings import get_settings

            settings = get_settings()
            storage_dir = settings.data_dir / "alerts"

        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.alerts_file = self.storage_dir / "alerts.json"

    def _load_alerts(self) -> list[PriceAlert]:
        """Load all alerts from storage."""
        if not self.alerts_file.exists():
            return []

        try:
            data = json.loads(self.alerts_file.read_text())
            return [PriceAlert.from_dict(alert) for alert in data]
        except Exception as e:
            logger.error(f"Error loading alerts: {e}")
            return []

    def _save_alerts(self, alerts: list[PriceAlert]) -> None:
        """Save alerts to storage."""
        data = [alert.to_dict() for alert in alerts]
        self.alerts_file.write_text(json.dumps(data, indent=2))

    def _generate_id(self) -> str:
        """Generate a unique alert ID."""
        import uuid

        return str(uuid.uuid4())[:8]

    def create_alert(
        self,
        symbol: str,
        alert_type: AlertType,
        target_price: float,
        note: str = "",
    ) -> PriceAlert:
        """Create a new price alert.

        Args:
            symbol: Stock symbol
            alert_type: Type of alert (above/below)
            target_price: Target price to trigger alert
            note: Optional note for the alert

        Returns:
            Created PriceAlert
        """
        alert = PriceAlert(
            id=self._generate_id(),
            symbol=symbol.upper(),
            alert_type=alert_type,
            target_price=target_price,
            created_at=datetime.now(),
            note=note,
        )

        alerts = self._load_alerts()
        alerts.append(alert)
        self._save_alerts(alerts)

        return alert

    def get_alerts(
        self,
        symbol: str | None = None,
        status: AlertStatus | None = None,
        active_only: bool = False,
    ) -> list[PriceAlert]:
        """Get alerts with optional filtering.

        Args:
            symbol: Filter by symbol
            status: Filter by status
            active_only: Only return active alerts

        Returns:
            List of matching alerts
        """
        alerts = self._load_alerts()

        if symbol:
            alerts = [a for a in alerts if a.symbol == symbol.upper()]

        if status:
            alerts = [a for a in alerts if a.status == status]
        elif active_only:
            alerts = [a for a in alerts if a.status == AlertStatus.ACTIVE]

        return alerts

    def delete_alert(self, alert_id: str) -> bool:
        """Delete an alert by ID.

        Args:
            alert_id: Alert ID to delete

        Returns:
            True if alert was deleted
        """
        alerts = self._load_alerts()
        original_count = len(alerts)
        alerts = [a for a in alerts if a.id != alert_id]

        if len(alerts) < original_count:
            self._save_alerts(alerts)
            return True
        return False

    def clear_triggered(self) -> int:
        """Clear all triggered alerts.

        Returns:
            Number of alerts cleared
        """
        alerts = self._load_alerts()
        active_alerts = [a for a in alerts if a.status == AlertStatus.ACTIVE]
        cleared = len(alerts) - len(active_alerts)

        if cleared > 0:
            self._save_alerts(active_alerts)

        return cleared

    def check_alerts(self, prices: dict[str, float]) -> list[PriceAlert]:
        """Check alerts against current prices.

        Args:
            prices: Dictionary of symbol -> current price

        Returns:
            List of triggered alerts
        """
        alerts = self._load_alerts()
        triggered = []

        for alert in alerts:
            if alert.status != AlertStatus.ACTIVE:
                continue

            current_price = prices.get(alert.symbol)
            if current_price is None:
                continue

            if alert.check(current_price):
                alert.status = AlertStatus.TRIGGERED
                alert.triggered_at = datetime.now()
                alert.triggered_price = current_price
                triggered.append(alert)

        if triggered:
            self._save_alerts(alerts)

        return triggered

    def get_symbols_to_check(self) -> list[str]:
        """Get list of symbols with active alerts.

        Returns:
            List of unique symbols
        """
        alerts = self.get_alerts(active_only=True)
        return list(set(a.symbol for a in alerts))
