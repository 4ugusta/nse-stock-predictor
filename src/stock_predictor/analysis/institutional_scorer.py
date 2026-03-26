"""Converts institutional and promoter data into signal scores.

Takes FII/DII flows, promoter holding patterns, and bulk deal data
from NSEDataProvider and produces a normalized score (-1 to +1) that
can be fed into the signal generator as an ExternalScores.fundamental_score.

Scoring logic:
- FII net buying → bullish for the market (positive score)
- DII net buying → mildly bullish (institutions providing support)
- High promoter holding → management confidence (positive)
- High promoter pledge → leverage risk (negative)
- Pledge increasing quarter-over-quarter → warning (negative)
- Bulk/block buy deals in the stock → positive
- Bulk/block sell deals → negative
"""

import logging
from dataclasses import dataclass

from stock_predictor.analysis.signals.signal_generator import ExternalScores
from stock_predictor.infrastructure.data_providers.nse_provider import (
    NSEInstitutionalData,
)

logger = logging.getLogger(__name__)


@dataclass
class InstitutionalBreakdown:
    """Detailed breakdown of how the institutional score was calculated."""

    fii_dii_score: float  # -1 to 1
    promoter_score: float  # -1 to 1
    bulk_deal_score: float  # -1 to 1
    combined_score: float  # Weighted average -1 to 1
    reasons: list[str]


class InstitutionalScorer:
    """Converts institutional data into signal-compatible scores.

    The combined score is a weighted blend:
    - FII/DII flows:    40% weight (broad market sentiment)
    - Promoter quality:  40% weight (stock-specific confidence)
    - Bulk deals:        20% weight (smart money indicator)
    """

    FII_DII_WEIGHT = 0.40
    PROMOTER_WEIGHT = 0.40
    BULK_DEAL_WEIGHT = 0.20

    # FII/DII thresholds (in crores)
    FII_STRONG_BUY = 2000  # FII net buy > 2000 Cr = strong bullish
    FII_MODERATE_BUY = 500
    FII_STRONG_SELL = -2000
    FII_MODERATE_SELL = -500

    # Promoter holding thresholds
    PROMOTER_HIGH = 60  # > 60% = strong management confidence
    PROMOTER_GOOD = 45
    PROMOTER_LOW = 25  # < 25% = weak promoter backing

    # Pledge risk thresholds
    PLEDGE_CRITICAL = 50
    PLEDGE_HIGH = 30
    PLEDGE_MODERATE = 10

    def score(self, data: NSEInstitutionalData) -> InstitutionalBreakdown:
        """Calculate institutional score from NSE data.

        Args:
            data: NSEInstitutionalData from nse_provider.get_all_data()

        Returns:
            InstitutionalBreakdown with component scores and reasons
        """
        reasons: list[str] = []

        fii_dii_score = self._score_fii_dii(data, reasons)
        promoter_score = self._score_promoter(data, reasons)
        bulk_deal_score = self._score_bulk_deals(data, reasons)

        combined = (
            fii_dii_score * self.FII_DII_WEIGHT
            + promoter_score * self.PROMOTER_WEIGHT
            + bulk_deal_score * self.BULK_DEAL_WEIGHT
        )

        # Clamp to [-1, 1]
        combined = max(-1.0, min(1.0, combined))

        if data.fetch_errors:
            for err in data.fetch_errors:
                reasons.append(f"[Data gap] {err}")

        return InstitutionalBreakdown(
            fii_dii_score=round(fii_dii_score, 3),
            promoter_score=round(promoter_score, 3),
            bulk_deal_score=round(bulk_deal_score, 3),
            combined_score=round(combined, 3),
            reasons=reasons,
        )

    def to_external_scores(
        self, data: NSEInstitutionalData
    ) -> ExternalScores:
        """Convert institutional data directly to ExternalScores.

        This is the convenience method for integration with the signal
        generator. Returns ExternalScores with the institutional data
        mapped to fundamental_score.

        Args:
            data: NSEInstitutionalData from nse_provider.get_all_data()

        Returns:
            ExternalScores ready for signal_generator.generate()
        """
        breakdown = self.score(data)

        return ExternalScores(
            sentiment_score=None,  # Sentiment comes from a separate module
            fundamental_score=breakdown.combined_score,
            sentiment_weight=0.10,
            fundamental_weight=0.10,
        )

    def _score_fii_dii(
        self, data: NSEInstitutionalData, reasons: list[str]
    ) -> float:
        """Score based on FII/DII activity.

        FII buying = international money flowing in (bullish).
        DII buying = domestic institutions providing support.
        Both buying = very bullish. Both selling = very bearish.
        """
        if not data.fii_dii:
            return 0.0

        latest = data.fii_dii[0]
        fii_net = latest.fii_net_value
        dii_net = latest.dii_net_value

        score = 0.0

        # FII component (stronger signal)
        if fii_net >= self.FII_STRONG_BUY:
            score += 0.6
            reasons.append(f"FII strong buying: +₹{fii_net:,.0f} Cr")
        elif fii_net >= self.FII_MODERATE_BUY:
            score += 0.3
            reasons.append(f"FII moderate buying: +₹{fii_net:,.0f} Cr")
        elif fii_net <= self.FII_STRONG_SELL:
            score -= 0.6
            reasons.append(f"FII strong selling: ₹{fii_net:,.0f} Cr")
        elif fii_net <= self.FII_MODERATE_SELL:
            score -= 0.3
            reasons.append(f"FII moderate selling: ₹{fii_net:,.0f} Cr")

        # DII component (weaker signal, often counter-cyclical)
        if dii_net >= self.FII_MODERATE_BUY:
            score += 0.2
            reasons.append(f"DII buying support: +₹{dii_net:,.0f} Cr")
        elif dii_net <= self.FII_MODERATE_SELL:
            score -= 0.1
            reasons.append(f"DII selling: ₹{dii_net:,.0f} Cr")

        # Divergence: FII selling + DII buying = uncertainty
        if fii_net < -500 and dii_net > 500:
            score *= 0.5  # Dampen the signal
            reasons.append("FII-DII divergence (uncertainty)")

        return max(-1.0, min(1.0, score))

    def _score_promoter(
        self, data: NSEInstitutionalData, reasons: list[str]
    ) -> float:
        """Score based on promoter holding and pledge levels.

        High promoter holding = management has skin in the game.
        High pledge = promoter using shares as collateral (risky).
        """
        if data.promoter is None:
            return 0.0

        p = data.promoter
        score = 0.0

        # Promoter holding level
        if p.promoter_holding_pct >= self.PROMOTER_HIGH:
            score += 0.4
            reasons.append(f"Strong promoter holding: {p.promoter_holding_pct:.1f}%")
        elif p.promoter_holding_pct >= self.PROMOTER_GOOD:
            score += 0.2
            reasons.append(f"Good promoter holding: {p.promoter_holding_pct:.1f}%")
        elif p.promoter_holding_pct < self.PROMOTER_LOW:
            score -= 0.3
            reasons.append(f"Low promoter holding: {p.promoter_holding_pct:.1f}%")

        # Pledge risk (major negative signal)
        if p.promoter_pledge_pct >= self.PLEDGE_CRITICAL:
            score -= 0.8
            reasons.append(
                f"CRITICAL pledge level: {p.promoter_pledge_pct:.1f}% of promoter shares pledged"
            )
        elif p.promoter_pledge_pct >= self.PLEDGE_HIGH:
            score -= 0.5
            reasons.append(f"High pledge risk: {p.promoter_pledge_pct:.1f}%")
        elif p.promoter_pledge_pct >= self.PLEDGE_MODERATE:
            score -= 0.2
            reasons.append(f"Moderate pledge: {p.promoter_pledge_pct:.1f}%")
        elif p.promoter_pledge_pct == 0:
            score += 0.1
            reasons.append("Zero promoter pledge")

        return max(-1.0, min(1.0, score))

    def _score_bulk_deals(
        self, data: NSEInstitutionalData, reasons: list[str]
    ) -> float:
        """Score based on recent bulk/block deals.

        Large institutional buys = smart money accumulation.
        Large institutional sells = distribution.
        """
        if not data.bulk_deals:
            return 0.0

        net_buy_qty = 0
        net_buy_value = 0.0

        for deal in data.bulk_deals:
            value = deal.quantity * deal.price
            if deal.deal_type.upper() in ("BUY", "B"):
                net_buy_qty += deal.quantity
                net_buy_value += value
            elif deal.deal_type.upper() in ("SELL", "S"):
                net_buy_qty -= deal.quantity
                net_buy_value -= value

        if net_buy_qty == 0:
            return 0.0

        score = 0.0

        if net_buy_value > 0:
            score = min(0.5, net_buy_value / 1_000_000_000)  # Scale: 100Cr = 0.1
            reasons.append(
                f"Net bulk buying: {len(data.bulk_deals)} deals, "
                f"₹{abs(net_buy_value) / 10_000_000:,.1f} Cr net"
            )
        else:
            score = max(-0.5, net_buy_value / 1_000_000_000)
            reasons.append(
                f"Net bulk selling: {len(data.bulk_deals)} deals, "
                f"₹{abs(net_buy_value) / 10_000_000:,.1f} Cr net"
            )

        return score
