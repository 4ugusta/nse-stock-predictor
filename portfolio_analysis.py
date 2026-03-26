#!/usr/bin/env python3
"""Portfolio analysis script using all system capabilities.

Analyzes current portfolio, market conditions, and finds optimal allocation
for additional capital using signal generation, market intelligence,
high-confidence picker, and risk management.
"""

import json
import sys
import logging

# Suppress noisy logs
logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, "src")

from stock_predictor.analysis.high_confidence_picker import (
    ConfidenceLevel,
    HighConfidencePicker,
    PickerConfig,
)
from stock_predictor.analysis.market_intelligence import (
    fetch_india_vix,
    get_market_breadth,
    get_fii_dii_activity,
    get_fii_dii_note,
)
from stock_predictor.analysis.risk_manager import (
    PortfolioPosition,
    RiskManager,
)
from stock_predictor.analysis.signals.signal_generator import SignalGenerator
from stock_predictor.core.entities.signal import TradingStyle
from stock_predictor.infrastructure.config.constants import (
    NIFTY50_SYMBOLS,
    NIFTY_NEXT50_SYMBOLS,
    SECTOR_STOCKS,
)
from stock_predictor.infrastructure.data_providers.yahoo_provider import YahooDataProvider

# ─── User's Current Portfolio ───────────────────────────────────────────────
EXTRA_CASH = 40196.0

PORTFOLIO = [
    {"symbol": "BAJAJ-AUTO", "qty": 1, "avg_price": 9106.50},
    {"symbol": "INFY", "qty": 3, "avg_price": 1497.90},
    {"symbol": "IRFC", "qty": 25, "avg_price": 144.24},
    {"symbol": "ITC", "qty": 15, "avg_price": 358.50},
    {"symbol": "LT", "qty": 1, "avg_price": 3747.20},
    {"symbol": "LUPIN", "qty": 3, "avg_price": 2185.20},   # ₹6,555.60 / 3 shares
    {"symbol": "NTPC", "qty": 20, "avg_price": 338.49},
    {"symbol": "ONGC", "qty": 44, "avg_price": 242.83},    # ₹10,684.42 / 44 shares
    {"symbol": "SBIN", "qty": 13, "avg_price": 335.00},
    {"symbol": "TATASTEEL", "qty": 39, "avg_price": 176.78},
]


def main():
    provider = YahooDataProvider()

    print("=" * 70)
    print("  PORTFOLIO ANALYSIS & INVESTMENT RECOMMENDATION")
    print("  Using: Signals + Market Intelligence + Risk Management")
    print("=" * 70)

    # ─── Step 1: Market Intelligence ────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  STEP 1: MARKET INTELLIGENCE")
    print("─" * 70)

    # VIX
    vix = fetch_india_vix()
    print(f"\n  India VIX: {vix.value:.2f} ({vix.regime})")
    print(f"    Confidence adjustment: {vix.confidence_adjustment:.2f}")
    print(f"    {vix.description}")

    # Market Breadth
    breadth = get_market_breadth()
    print(f"\n  Market Breadth: {breadth.breadth_signal}")
    print(f"    Advances: {breadth.advances} | Declines: {breadth.declines} | Unchanged: {breadth.unchanged}")
    print(f"    A/D Ratio: {breadth.advance_decline_ratio:.2f}")
    print(f"    NIFTY Change: {breadth.nifty_change_pct:+.2f}%")
    print(f"    Confidence adjustment: {breadth.confidence_adjustment:.2f}")

    # FII/DII
    fii_dii = get_fii_dii_activity()
    print(f"\n  FII/DII Activity: {fii_dii.flow_signal}")
    print(f"    FII Net Contracts: {fii_dii.fii_net_contracts:+,}")
    print(f"    FII Index Futures Net: {fii_dii.fii_index_futures_net:+,}")
    print(f"    Confidence adjustment: {fii_dii.confidence_adjustment:.2f}")
    print(f"    {fii_dii.description}")

    # Overall market assessment
    market_score = vix.confidence_adjustment * breadth.confidence_adjustment * fii_dii.confidence_adjustment
    if market_score >= 1.0:
        market_outlook = "FAVORABLE"
    elif market_score >= 0.90:
        market_outlook = "NEUTRAL"
    else:
        market_outlook = "CAUTIOUS"
    print(f"\n  >>> OVERALL MARKET OUTLOOK: {market_outlook} (combined score: {market_score:.3f})")

    # ─── Step 2: Current Portfolio Analysis ─────────────────────────────────
    print("\n" + "─" * 70)
    print("  STEP 2: CURRENT PORTFOLIO HEALTH")
    print("─" * 70)

    positions = []
    portfolio_signals = {}

    generator = SignalGenerator()

    for holding in PORTFOLIO:
        sym = holding["symbol"]
        try:
            data = provider.get_historical(sym, period="1y", interval="1d")
            current_price = float(data["close"].iloc[-1])

            pos = PortfolioPosition(
                symbol=sym,
                shares=holding["qty"],
                entry_price=holding["avg_price"],
                current_price=current_price,
            )
            positions.append(pos)

            # Generate signal for each holding
            signal = generator.generate(data, sym, TradingStyle.SWING)
            portfolio_signals[sym] = signal

        except Exception as e:
            print(f"  WARNING: Could not fetch data for {sym}: {e}")
            positions.append(PortfolioPosition(
                symbol=sym,
                shares=holding["qty"],
                entry_price=holding["avg_price"],
                current_price=holding["avg_price"],
            ))

    # Portfolio risk assessment
    total_invested = sum(h["qty"] * h["avg_price"] for h in PORTFOLIO)
    total_capital = total_invested + EXTRA_CASH
    rm = RiskManager(total_capital=total_capital, risk_per_trade=1.0, existing_positions=positions)
    portfolio_risk = rm.get_portfolio_risk()

    print(f"\n  Total Capital: ₹{total_capital:,.0f}")
    print(f"  Invested: ₹{portfolio_risk.total_invested:,.0f}")
    print(f"  Cash Available: ₹{EXTRA_CASH:,.0f}")
    print(f"  P&L: ₹{portfolio_risk.total_pnl:,.0f} ({portfolio_risk.total_pnl_percent:+.1f}%)")
    print(f"  Positions: {portfolio_risk.num_positions}")
    print(f"  Portfolio Heat: {portfolio_risk.portfolio_heat:.1f}%")
    print(f"  Largest Position: {portfolio_risk.largest_position_percent:.1f}%")

    print(f"\n  Sector Exposure:")
    for sector, pct in sorted(portfolio_risk.sector_exposure.items(), key=lambda x: -x[1]):
        bar = "█" * int(pct / 2)
        print(f"    {sector:<12} {pct:5.1f}% {bar}")

    if portfolio_risk.warnings:
        print(f"\n  ⚠ Risk Warnings:")
        for w in portfolio_risk.warnings:
            print(f"    • {w}")

    # Current holdings signals
    print(f"\n  Current Holdings - Signal Analysis:")
    print(f"  {'Symbol':<12} {'Price':>10} {'P&L%':>8} {'Signal':>12} {'Confidence':>12}")
    print(f"  {'-'*12} {'-'*10} {'-'*8} {'-'*12} {'-'*12}")

    for pos in sorted(positions, key=lambda p: p.pnl_percent, reverse=True):
        sig = portfolio_signals.get(pos.symbol)
        sig_type = sig.signal_type.value if sig else "N/A"
        sig_conf = f"{sig.confidence:.0%}" if sig else "N/A"
        pnl_pct = pos.pnl_percent
        print(f"  {pos.symbol:<12} ₹{pos.current_price:>8,.0f} {pnl_pct:>+7.1f}% {sig_type:>12} {sig_conf:>12}")

    # ─── Step 3: High Confidence Scanning ───────────────────────────────────
    print("\n" + "─" * 70)
    print("  STEP 3: HIGH CONFIDENCE STOCK SCANNING")
    print("─" * 70)

    # Scan Nifty 50 + Next 50 for opportunities
    scan_universe = NIFTY50_SYMBOLS + NIFTY_NEXT50_SYMBOLS
    # Remove stocks already in portfolio
    portfolio_symbols = {h["symbol"] for h in PORTFOLIO}
    scan_universe = [s for s in scan_universe if s not in portfolio_symbols]

    print(f"\n  Scanning {len(scan_universe)} stocks (Nifty 50 + Next 50, excluding held)...")
    print("  Applying 6 filters: Trend + Volume + MA + Momentum + RSI + Whipsaw\n")

    picker = HighConfidencePicker(
        config=PickerConfig(
            min_risk_reward=1.5,
            min_adx=25.0,
            min_volume_ratio=1.0,
        )
    )

    both_sides = picker.scan_both_sides(scan_universe, min_confidence=ConfidenceLevel.MEDIUM)
    long_picks = both_sides["long"]
    short_picks = both_sides["short"]

    print(f"  Found {len(long_picks)} LONG opportunities, {len(short_picks)} SHORT opportunities\n")

    if long_picks:
        print(f"  TOP LONG PICKS:")
        print(f"  {'#':<3} {'Symbol':<14} {'Price':>10} {'Confidence':>12} {'R:R':>6} {'Stop':>10} {'Target':>10} {'Regime':<15}")
        print(f"  {'-'*3} {'-'*14} {'-'*10} {'-'*12} {'-'*6} {'-'*10} {'-'*10} {'-'*15}")

        for i, pick in enumerate(long_picks[:10], 1):
            print(
                f"  {i:<3} {pick.symbol:<14} ₹{pick.current_price:>8,.0f} "
                f"{pick.confidence_score:>10.0f}% "
                f"{pick.risk_reward_ratio:>5.1f}x "
                f"₹{pick.stop_loss:>8,.0f} "
                f"₹{pick.target_1:>8,.0f} "
                f"{pick.regime.value:<15}"
            )

        # Detailed analysis of top picks
        print(f"\n  DETAILED TOP PICKS:")
        for i, pick in enumerate(long_picks[:5], 1):
            print(f"\n  #{i} {pick.symbol} - {pick.recommendation}")
            print(f"     Current: ₹{pick.current_price:,.2f} | Entry: ₹{pick.entry_price:,.2f}")
            print(f"     Stop: ₹{pick.stop_loss:,.2f} ({pick.risk_percent:.1f}% risk)")
            print(f"     Target 1: ₹{pick.target_1:,.2f} ({pick.reward_percent:.1f}% reward)")
            print(f"     Target 2: ₹{pick.target_2:,.2f}")
            print(f"     R:R: 1:{pick.risk_reward_ratio:.1f} | ADX: {pick.adx_value:.0f} | RSI: {pick.rsi_value:.0f} | Vol: {pick.volume_ratio:.1f}x")

            passed_str = ", ".join(f.name for f in pick.filters_passed)
            failed_str = ", ".join(f.name for f in pick.filters_failed) if pick.filters_failed else "None"
            print(f"     ✓ Passed: {passed_str}")
            print(f"     ✗ Failed: {failed_str}")

            # Generate full signal for context
            try:
                data = provider.get_historical(pick.symbol, period="1y")
                sig = generator.generate(data, pick.symbol, TradingStyle.SWING)
                print(f"     Signal: {sig.signal_type.value} (confidence: {sig.confidence:.0%})")
                if sig.reasons:
                    for r in sig.reasons[:3]:
                        print(f"       • {r}")
            except Exception:
                pass

    # ─── Step 4: Risk-Managed Allocation ────────────────────────────────────
    print("\n" + "─" * 70)
    print("  STEP 4: RISK-MANAGED ALLOCATION FOR ₹{:,.0f}".format(EXTRA_CASH))
    print("─" * 70)

    if not long_picks:
        print("\n  No high-confidence LONG picks passed all filters.")
        print("  This indicates a WEAK/BEARISH market phase.")

    # Use risk manager to size positions from the best picks
    allocation_rm = RiskManager(
        total_capital=total_capital,
        risk_per_trade=1.0,
        existing_positions=positions,
    )

    allocations = []
    remaining_cash = EXTRA_CASH

    for pick in long_picks[:8]:
        if remaining_cash < 1000:
            break

        try:
            data = provider.get_historical(pick.symbol, period="1y")

            # Risk assessment (checks sector concentration, correlation, etc.)
            assessment = allocation_rm.assess_trade(pick.symbol, data, is_long=True)

            if assessment.can_trade and assessment.position_size and assessment.position_size.shares > 0:
                pos_size = assessment.position_size

                # Cap to remaining cash
                max_shares_by_cash = int(remaining_cash / pos_size.entry_price)
                actual_shares = min(pos_size.shares, max_shares_by_cash)

                if actual_shares > 0:
                    actual_cost = actual_shares * pos_size.entry_price
                    actual_risk = actual_shares * pos_size.risk_per_share

                    allocations.append({
                        "symbol": pick.symbol,
                        "shares": actual_shares,
                        "price": pos_size.entry_price,
                        "cost": actual_cost,
                        "stop_loss": pos_size.stop_loss,
                        "risk": actual_risk,
                        "target_1": float(pick.target_1),
                        "target_2": float(pick.target_2),
                        "confidence": pick.confidence_score,
                        "risk_reward": pick.risk_reward_ratio,
                        "regime": pick.regime.value,
                        "warnings": assessment.warnings,
                        "recommendation": pick.recommendation,
                    })

                    remaining_cash -= actual_cost

                    # Add to RM positions for next iteration's concentration check
                    allocation_rm.positions.append(PortfolioPosition(
                        symbol=pick.symbol,
                        shares=actual_shares,
                        entry_price=pos_size.entry_price,
                        current_price=pos_size.entry_price,
                    ))

            elif assessment.blockers:
                print(f"\n  {pick.symbol} BLOCKED:")
                for b in assessment.blockers:
                    print(f"    ✗ {b}")

        except Exception as e:
            print(f"  WARNING: Could not assess {pick.symbol}: {e}")

    # ─── Step 5: Final Recommendation ───────────────────────────────────────
    print("\n" + "═" * 70)
    print("  FINAL INVESTMENT RECOMMENDATION")
    print("═" * 70)

    print(f"\n  Market Outlook: {market_outlook}")
    print(f"  Budget: ₹{EXTRA_CASH:,.0f}")
    print(f"  Allocated: ₹{sum(a['cost'] for a in allocations):,.0f}")
    print(f"  Remaining Cash: ₹{remaining_cash:,.0f}")

    if allocations:
        print(f"\n  RECOMMENDED BUYS:")
        print(f"  {'Symbol':<14} {'Shares':>6} {'Price':>10} {'Cost':>12} {'Stop':>10} {'Target':>10} {'Max Loss':>10}")
        print(f"  {'-'*14} {'-'*6} {'-'*10} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")

        total_cost = 0
        total_risk = 0
        total_upside = 0

        for a in allocations:
            upside = a["shares"] * (a["target_1"] - a["price"])
            total_cost += a["cost"]
            total_risk += a["risk"]
            total_upside += upside

            print(
                f"  {a['symbol']:<14} {a['shares']:>6} ₹{a['price']:>8,.0f} ₹{a['cost']:>10,.0f} "
                f"₹{a['stop_loss']:>8,.0f} ₹{a['target_1']:>8,.0f} ₹{a['risk']:>8,.0f}"
            )

            if a["warnings"]:
                for w in a["warnings"]:
                    print(f"    ⚠ {w}")

        print(f"\n  SUMMARY:")
        print(f"    Total Investment: ₹{total_cost:,.0f}")
        print(f"    Total Max Risk (if all stops hit): ₹{total_risk:,.0f} ({total_risk/total_capital*100:.1f}% of capital)")
        print(f"    Total Potential Upside (Target 1): ₹{total_upside:,.0f}")
        print(f"    Cash Reserve After Investment: ₹{remaining_cash:,.0f}")

        # Portfolio after allocation
        print(f"\n  PORTFOLIO AFTER ALLOCATION:")
        print(f"    Total Positions: {len(positions) + len(allocations)}")

        # Calculate new sector exposure
        all_positions = list(positions)
        for a in allocations:
            all_positions.append(PortfolioPosition(
                symbol=a["symbol"],
                shares=a["shares"],
                entry_price=a["price"],
                current_price=a["price"],
            ))

        new_rm = RiskManager(total_capital=total_capital, existing_positions=all_positions)
        new_risk = new_rm.get_portfolio_risk()

        print(f"    Total Invested: ₹{new_risk.total_invested:,.0f}")
        print(f"    Cash Reserve: ₹{new_risk.cash_available:,.0f}")
        print(f"    Portfolio Heat: {new_risk.portfolio_heat:.1f}%")

        print(f"\n    New Sector Exposure:")
        for sector, pct in sorted(new_risk.sector_exposure.items(), key=lambda x: -x[1]):
            bar = "█" * int(pct / 2)
            print(f"      {sector:<12} {pct:5.1f}% {bar}")

        if new_risk.warnings:
            print(f"\n    ⚠ New Risk Warnings:")
            for w in new_risk.warnings:
                print(f"      • {w}")

    else:
        print("\n  No allocations recommended given current market conditions and risk limits.")
        print("  RECOMMENDATION: Keep ₹{:,.0f} in cash and wait for better setups.".format(EXTRA_CASH))

    # Display short picks if found
    if short_picks:
        print(f"\n  SHORT OPPORTUNITIES (for awareness - AVOID if you're a long-only investor):")
        for i, pick in enumerate(short_picks[:5], 1):
            print(
                f"    {i}. {pick.symbol} - {pick.recommendation} "
                f"(Conf: {pick.confidence_score:.0f}%, R:R 1:{pick.risk_reward_ratio:.1f})"
            )

    # Existing portfolio actions
    print(f"\n  ACTIONS ON EXISTING HOLDINGS:")
    for pos in positions:
        sig = portfolio_signals.get(pos.symbol)
        if sig:
            if sig.signal_type.value in ("STRONG_SELL", "SELL"):
                print(f"    ⚠ {pos.symbol}: Consider REDUCING - Signal: {sig.signal_type.value} ({sig.confidence:.0%})")
                if sig.stop_loss:
                    print(f"      Set stop loss at ₹{sig.stop_loss:.0f}")
            elif sig.signal_type.value == "NO_EDGE":
                print(f"    → {pos.symbol}: HOLD (no clear edge) - monitor for breakout/breakdown")
            elif sig.signal_type.value in ("STRONG_BUY", "BUY") and pos.pnl_percent > 0:
                print(f"    ✓ {pos.symbol}: Profitable (+{pos.pnl_percent:.1f}%) with {sig.signal_type.value} signal - HOLD/ADD")
            elif sig.signal_type.value in ("STRONG_BUY", "BUY") and pos.pnl_percent < -5:
                print(f"    → {pos.symbol}: In loss ({pos.pnl_percent:.1f}%) but {sig.signal_type.value} - HOLD/AVERAGE DOWN")
            else:
                print(f"    → {pos.symbol}: HOLD - Signal: {sig.signal_type.value} (P&L: {pos.pnl_percent:+.1f}%)")

    print("\n" + "═" * 70)
    print("  DISCLAIMER")
    print("═" * 70)
    print("  This analysis is for EDUCATIONAL purposes only.")
    print("  Indicator Agreement (%) measures how many indicators align,")
    print("  NOT the probability of profit. Always do your own research.")
    print("  The authors are NOT SEBI-registered advisors.")
    print("  Past performance does not guarantee future results.")
    print("=" * 70)


if __name__ == "__main__":
    main()
