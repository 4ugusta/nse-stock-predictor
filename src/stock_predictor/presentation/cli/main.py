"""CLI entry point for NSE Stock Predictor."""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from stock_predictor.analysis.screener.scanner import SCREENER_PRESETS, ScreenerFilters, StockScanner
from stock_predictor.analysis.sentiment.analyzer import SentimentAnalyzer
from stock_predictor.analysis.signals.signal_generator import ExternalScores, SignalGenerator
from stock_predictor.core.entities.signal import SignalType, TradingStyle
from stock_predictor.infrastructure.config.constants import (
    ALL_STOCKS,
    BANKNIFTY_SYMBOLS,
    FNO_STOCKS,
    HIGH_VOLATILITY_STOCKS,
    NIFTY50_SYMBOLS,
    NIFTY_MIDCAP_SYMBOLS,
    NIFTY_NEXT50_SYMBOLS,
    NIFTY_SMALLCAP_SYMBOLS,
    NSE_INDICES,
    SECTOR_STOCKS,
)
from stock_predictor.analysis.performance_tracker import PerformanceTracker
from stock_predictor.infrastructure.data_providers.yahoo_provider import YahooDataProvider
from stock_predictor.infrastructure.news_providers.aggregator import NewsAggregator

app = typer.Typer(
    name="nse-predict",
    help="Indian NSE Stock Predictor - Technical Analysis & Trading Signals",
    add_completion=False,
)
console = Console()


def get_signal_color(signal_type: SignalType) -> str:
    """Get color for signal type."""
    return {
        SignalType.STRONG_BUY: "bold green",
        SignalType.BUY: "green",
        SignalType.HOLD: "yellow",
        SignalType.NO_EDGE: "dim",
        SignalType.SELL: "red",
        SignalType.STRONG_SELL: "bold red",
    }.get(signal_type, "white")


def get_signal_label(signal_type: SignalType) -> str:
    """Get display label for signal type."""
    if signal_type == SignalType.NO_EDGE:
        return "NO EDGE - Insufficient signal clarity"
    return signal_type.value.upper()


def _build_external_scores(symbol: str, skip_institutional: bool = False) -> ExternalScores | None:
    """Fetch institutional + sentiment data and combine into ExternalScores.

    Returns None if all data sources fail (signal generator will use pure technicals).
    """
    sentiment_score = None
    fundamental_score = None

    # 1. Sentiment from news
    try:
        aggregator = NewsAggregator()
        analyzer = SentimentAnalyzer()
        articles = aggregator.fetch_for_symbol(symbol, limit_per_source=5)
        if articles:
            articles = analyzer.analyze_articles(articles)
            aggregate = analyzer.get_aggregate_sentiment(articles)
            sentiment_score = aggregate.overall_score  # Already -1 to 1
            console.print(f"[dim]Sentiment ({len(articles)} articles): {sentiment_score:+.2f}[/dim]")
    except Exception as e:
        console.print(f"[dim]Sentiment unavailable: {e}[/dim]")

    # 2. Institutional data from NSE
    if not skip_institutional:
        try:
            from stock_predictor.analysis.institutional_scorer import InstitutionalScorer
            from stock_predictor.infrastructure.data_providers.nse_provider import NSEDataProvider

            nse = NSEDataProvider()
            nse_data = nse.get_all_data(symbol)
            scorer = InstitutionalScorer()
            breakdown = scorer.score(nse_data)
            fundamental_score = breakdown.combined_score

            if breakdown.reasons:
                console.print("[dim]Institutional data:[/dim]")
                for reason in breakdown.reasons[:4]:
                    console.print(f"[dim]  • {reason}[/dim]")
        except Exception as e:
            console.print(f"[dim]NSE institutional data unavailable: {e}[/dim]")

    if sentiment_score is None and fundamental_score is None:
        return None

    return ExternalScores(
        sentiment_score=sentiment_score,
        fundamental_score=fundamental_score,
    )


@app.command()
def analyze(
    symbol: str = typer.Argument(..., help="NSE stock symbol (e.g., RELIANCE)"),
    style: str = typer.Option(
        "swing", "--style", "-s", help="Trading style: intraday, swing, positional"
    ),
    period: str = typer.Option("1y", "--period", "-p", help="Data period (1d, 5d, 1mo, 3mo, 6mo, 1y)"),
    show_indicators: bool = typer.Option(False, "--indicators", "-i", help="Show detailed indicators"),
    no_institutional: bool = typer.Option(False, "--no-institutional", help="Skip NSE institutional data fetch"),
) -> None:
    """Analyze a stock and generate trading signals."""
    symbol = symbol.upper()

    # Map style string to enum
    style_map = {
        "intraday": TradingStyle.INTRADAY,
        "swing": TradingStyle.SWING,
        "positional": TradingStyle.POSITIONAL,
    }
    trading_style = style_map.get(style.lower(), TradingStyle.SWING)

    # Intraday warning: daily data is unreliable for intraday signals
    if trading_style == TradingStyle.INTRADAY:
        console.print(
            "\n[yellow]NOTE: Intraday style uses daily data intervals. "
            "Targets and stops are based on daily ATR, which may be too wide for "
            "same-day trades. For genuine intraday signals, use 5m/15m data "
            "with a real-time data provider (Kite Connect).[/yellow]"
        )

    console.print(f"\n[bold]Analyzing {symbol}...[/bold]\n")

    try:
        # Fetch data
        provider = YahooDataProvider()
        data = provider.get_historical(symbol, period=period)

        # Get quote
        try:
            quote = provider.get_quote(symbol)
            price_text = f"[bold]₹{quote.last_price:.2f}[/bold]"
            change_color = "green" if quote.is_positive else "red"
            change_text = f"[{change_color}]{quote.change:+.2f} ({quote.change_percent:+.2f}%)[/{change_color}]"
        except Exception:
            price_text = f"[bold]₹{data['close'].iloc[-1]:.2f}[/bold]"
            change_text = ""

        console.print(f"{symbol}: {price_text} {change_text}\n")

        # Build external scores from institutional data + sentiment
        external_scores = _build_external_scores(symbol, no_institutional)

        # Generate signal
        generator = SignalGenerator()
        signal = generator.generate(data, symbol, trading_style, external_scores=external_scores)

        # Log signal for performance tracking (with regime and auto-resolution)
        try:
            tracker = PerformanceTracker()
            # Auto-resolve pending signals using current prices
            try:
                pending_symbols = {s.symbol for s in tracker._signals if s.outcome == "pending"}
                if pending_symbols:
                    current_prices = {}
                    for ps in pending_symbols:
                        try:
                            q = provider.get_quote(ps)
                            current_prices[ps] = float(q.last_price)
                        except Exception:
                            pass
                    if current_prices:
                        resolved = tracker.check_pending_signals(current_prices)
                        if resolved:
                            console.print(f"[dim]Auto-resolved {len(resolved)} pending signal(s)[/dim]")
            except Exception:
                pass

            # Get regime from diagnostics if available
            regime_str = "unknown"
            if hasattr(signal, 'diagnostics') and signal.diagnostics:
                regime_str = signal.diagnostics.market_regime.value

            tracker.log_signal(
                symbol=symbol,
                signal_type=signal.signal_type.value,
                confidence=signal.confidence,
                entry_price=signal.entry_price,
                target_price=signal.target_price,
                stop_loss=signal.stop_loss,
                style=trading_style.value,
                reasons=signal.reasons,
                regime=regime_str,
            )
        except Exception:
            pass  # Don't let tracking failures affect analysis

        # Display signal
        signal_color = get_signal_color(signal.signal_type)

        # Build indicator agreement detail
        total_indicators = len(signal.indicator_signals) if signal.indicator_signals else 0
        if total_indicators > 0:
            agreeing = sum(
                1 for ind in signal.indicator_signals
                if (ind.score > 0 and signal.signal_type in (SignalType.BUY, SignalType.STRONG_BUY))
                or (ind.score < 0 and signal.signal_type in (SignalType.SELL, SignalType.STRONG_SELL))
                or (abs(ind.score) < 0.1 and signal.signal_type == SignalType.HOLD)
            )
            agreement_detail = f"Indicator Agreement: {signal.confidence:.1%} ({agreeing}/{total_indicators} indicators agree)"
        else:
            agreement_detail = f"Indicator Agreement: {signal.confidence:.1%}"

        signal_label = get_signal_label(signal.signal_type)
        signal_panel = Panel(
            f"[{signal_color}]{signal_label}[/{signal_color}]\n"
            f"Strength: {signal.strength.value.title()}\n"
            f"{agreement_detail}",
            title=f"[bold]{trading_style.value.title()} Signal[/bold]",
            border_style=signal_color,
        )
        console.print(signal_panel)

        # Entry/Exit levels
        if signal.signal_type not in (SignalType.HOLD, SignalType.NO_EDGE):
            levels_table = Table(show_header=False, box=None)
            levels_table.add_column("Label", style="dim")
            levels_table.add_column("Value", style="bold")

            levels_table.add_row("Entry", f"₹{signal.entry_price:.2f}")
            if signal.target_price:
                levels_table.add_row("Target", f"₹{signal.target_price:.2f}")
            if signal.stop_loss:
                levels_table.add_row("Stop Loss", f"₹{signal.stop_loss:.2f}")
            if signal.risk_reward_ratio:
                levels_table.add_row("Risk:Reward", f"1:{signal.risk_reward_ratio:.1f}")

            console.print("\n[bold]Entry/Exit Levels:[/bold]")
            console.print(levels_table)

            # Position sizing suggestion (1% risk per trade)
            if signal.stop_loss:
                risk_per_share = abs(signal.entry_price - signal.stop_loss)
                if risk_per_share > 0:
                    console.print("\n[bold]Position Sizing (1% capital risk per trade):[/bold]")
                    sizing_table = Table(show_header=True, header_style="dim", box=None)
                    sizing_table.add_column("Capital", style="dim")
                    sizing_table.add_column("Shares", justify="right")
                    sizing_table.add_column("Investment", justify="right")
                    sizing_table.add_column("Max Loss", justify="right", style="red")
                    for capital in [100_000, 500_000, 1_000_000]:
                        risk_amount = capital * 0.01
                        shares = int(risk_amount / risk_per_share)
                        if shares > 0:
                            cost = shares * signal.entry_price
                            max_loss = shares * risk_per_share
                            sizing_table.add_row(
                                f"₹{capital / 100_000:.0f}L",
                                str(shares),
                                f"₹{cost:,.0f}",
                                f"₹{max_loss:,.0f}",
                            )
                    console.print(sizing_table)

        # Show data freshness
        if hasattr(data, 'index') and len(data) > 0:
            last_date = data.index[-1]
            if isinstance(last_date, pd.Timestamp):
                freshness = (pd.Timestamp.now() - last_date).days
                if freshness > 1:
                    console.print(f"  [yellow]Data is {freshness} days old (last: {last_date.strftime('%Y-%m-%d')})[/yellow]")

        # Disclaimer
        console.print(
            "\n[dim][bold red]DISCLAIMER:[/bold red] This tool provides technical analysis signals for "
            "educational purposes only. Indicator Agreement (shown as %) measures how many technical "
            "indicators point in the same direction — it is NOT a prediction of profit probability. "
            "A 80% agreement means 80% of indicators are aligned, not that there's an 80% chance of profit. "
            "Always do your own research. Past performance does not guarantee future results. "
            "The authors are not SEBI-registered advisors.[/dim]"
        )

        # Historical accuracy (if enough tracked data exists)
        try:
            accuracy = PerformanceTracker().get_accuracy(symbol=symbol)
            if accuracy.resolved_signals >= 5:
                win_color = "green" if accuracy.win_rate >= 50 else "red"
                console.print(f"\n[bold]Historical Signal Accuracy for {symbol}:[/bold]")
                console.print(
                    f"  Tracked: {accuracy.total_signals} signals "
                    f"({accuracy.resolved_signals} resolved, {accuracy.pending_signals} pending)"
                )
                console.print(
                    f"  Win rate: [{win_color}]{accuracy.win_rate:.1f}%[/{win_color}] "
                    f"({accuracy.targets_hit}W / {accuracy.stops_hit}L / {accuracy.expired}E)"
                )
                console.print(
                    f"  Avg win: [green]+{accuracy.avg_win_pct:.1f}%[/green] | "
                    f"Avg loss: [red]{accuracy.avg_loss_pct:.1f}%[/red]"
                )
                if accuracy.profit_factor != float("inf"):
                    pf_color = "green" if accuracy.profit_factor >= 1.0 else "red"
                    console.print(f"  Profit factor: [{pf_color}]{accuracy.profit_factor:.2f}[/{pf_color}]")
                if accuracy.calibration_warnings:
                    for w in accuracy.calibration_warnings:
                        console.print(f"  [yellow]⚠ {w}[/yellow]")
        except Exception:
            pass

        # Reasons
        if signal.reasons:
            console.print("\n[bold]Analysis:[/bold]")
            for reason in signal.reasons:
                console.print(f"  • {reason}")

        # Detailed indicators
        if show_indicators:
            console.print("\n[bold]Technical Indicators:[/bold]")
            summary = generator.get_indicator_summary(data)

            ind_table = Table(show_header=True, header_style="bold")
            ind_table.add_column("Indicator")
            ind_table.add_column("Value")

            ind_table.add_row("Price", f"₹{summary['price']:.2f}")
            ind_table.add_row("RSI (14)", f"{summary['rsi']['value']:.1f} ({summary['rsi']['zone']})")
            ind_table.add_row(
                "MACD",
                f"{summary['macd']['macd']:.2f} / {summary['macd']['signal']:.2f} ({summary['macd']['trend']})",
            )
            ind_table.add_row(
                "Bollinger %B",
                f"{summary['bollinger']['percent_b']:.2f}",
            )

            for ma_name, ma_value in summary["moving_averages"].items():
                if ma_value and not pd.isna(ma_value):
                    ind_table.add_row(ma_name, f"₹{ma_value:.2f}")

            if summary["support_resistance"]["nearest_support"]:
                ind_table.add_row(
                    "Nearest Support",
                    f"₹{summary['support_resistance']['nearest_support']:.2f}",
                )
            if summary["support_resistance"]["nearest_resistance"]:
                ind_table.add_row(
                    "Nearest Resistance",
                    f"₹{summary['support_resistance']['nearest_resistance']:.2f}",
                )

            console.print(ind_table)

    except Exception as e:
        console.print(f"[red]Error analyzing {symbol}: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def market() -> None:
    """Show market overview - indices and market status."""
    console.print("\n[bold]Market Overview[/bold]\n")

    provider = YahooDataProvider()

    # Market status
    is_open = provider.is_market_open()
    status_text = "[green]OPEN[/green]" if is_open else "[red]CLOSED[/red]"
    console.print(f"Market Status: {status_text}\n")

    # Indices
    console.print("[bold]Key Indices:[/bold]")
    indices_table = Table(show_header=True, header_style="bold")
    indices_table.add_column("Index")
    indices_table.add_column("Value", justify="right")
    indices_table.add_column("Change", justify="right")

    for index_key in ["NIFTY50", "BANKNIFTY"]:
        try:
            index_data = provider.get_index_data(index_key)
            change_color = "green" if index_data.change >= 0 else "red"
            indices_table.add_row(
                index_data.name,
                f"₹{index_data.value:,.2f}",
                f"[{change_color}]{index_data.change:+.2f} ({index_data.change_percent:+.2f}%)[/{change_color}]",
            )
        except Exception as e:
            indices_table.add_row(NSE_INDICES.get(index_key, index_key), "N/A", "N/A")

    console.print(indices_table)

    # Top movers from Nifty 50
    console.print("\n[bold]Top Movers (Nifty 50):[/bold]")

    try:
        # Get quotes for all Nifty 50 stocks (not just a sample)
        sample_symbols = NIFTY50_SYMBOLS
        quotes = provider.get_multiple_quotes(sample_symbols)

        if quotes:
            # Sort by change percent
            sorted_quotes = sorted(
                quotes.values(), key=lambda x: float(x.change_percent), reverse=True
            )

            gainers = sorted_quotes[:5]
            losers = sorted_quotes[-5:][::-1]

            movers_table = Table(show_header=True, header_style="bold")
            movers_table.add_column("Gainers")
            movers_table.add_column("Change", justify="right")
            movers_table.add_column("Losers")
            movers_table.add_column("Change", justify="right")

            for i in range(5):
                gainer = gainers[i] if i < len(gainers) else None
                loser = losers[i] if i < len(losers) else None

                gainer_text = gainer.symbol if gainer else ""
                gainer_change = f"[green]+{gainer.change_percent:.2f}%[/green]" if gainer else ""
                loser_text = loser.symbol if loser else ""
                loser_change = f"[red]{loser.change_percent:.2f}%[/red]" if loser else ""

                movers_table.add_row(gainer_text, gainer_change, loser_text, loser_change)

            console.print(movers_table)

    except Exception as e:
        console.print(f"[dim]Could not fetch top movers: {e}[/dim]")


@app.command()
def screen(
    preset: str = typer.Option(None, "--preset", "-p", help="Use preset screener"),
    rsi_oversold: bool = typer.Option(False, "--rsi-oversold", help="Find RSI < 30 stocks"),
    rsi_overbought: bool = typer.Option(False, "--rsi-overbought", help="Find RSI > 70 stocks"),
    above_200_sma: bool = typer.Option(False, "--above-200sma", help="Stocks above 200 SMA"),
    macd_bullish: bool = typer.Option(False, "--macd-bullish", help="Bullish MACD crossover"),
    squeeze: bool = typer.Option(False, "--squeeze", help="Bollinger Band squeeze"),
    universe: str = typer.Option(
        "fno",
        "--universe",
        "-u",
        help="Stock universe: nifty50, next50, midcap, smallcap, fno, volatile, all, or sector:name",
    ),
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum results"),
) -> None:
    """Screen stocks based on technical criteria.

    Stock Universes:
      - nifty50: Large-cap blue chips (low volatility)
      - next50: Large-cap growth stocks
      - midcap: Mid-cap stocks (medium volatility)
      - smallcap: Small-cap stocks (high volatility)
      - fno: F&O stocks with high liquidity
      - volatile: High-volatility stocks for aggressive trading
      - all: All ~300 stocks
      - sector:banking, sector:it, sector:pharma, etc.

    Examples:
      nse-predict screen -u midcap --preset momentum
      nse-predict screen -u volatile --rsi-oversold
      nse-predict screen -u sector:pharma --macd-bullish
    """
    # Determine symbols to scan based on universe
    universe_lower = universe.lower()

    # Universe mapping
    universes = {
        "nifty50": (NIFTY50_SYMBOLS, "Nifty 50 (Large-Cap)"),
        "next50": (NIFTY_NEXT50_SYMBOLS, "Nifty Next 50 (Large-Cap Growth)"),
        "midcap": (NIFTY_MIDCAP_SYMBOLS, "Nifty Midcap (Medium Volatility)"),
        "smallcap": (NIFTY_SMALLCAP_SYMBOLS, "Nifty Smallcap (High Volatility)"),
        "banknifty": (BANKNIFTY_SYMBOLS, "Bank Nifty"),
        "fno": (FNO_STOCKS, "F&O Stocks (Liquid)"),
        "volatile": (HIGH_VOLATILITY_STOCKS, "High Volatility Stocks"),
        "all": (ALL_STOCKS, "All Stocks (~300)"),
    }

    # Check for sector-specific universe
    if universe_lower.startswith("sector:"):
        sector_name = universe_lower.split(":")[1]
        if sector_name in SECTOR_STOCKS:
            symbols = SECTOR_STOCKS[sector_name]
            universe_name = f"Sector: {sector_name.title()}"
        else:
            console.print(f"[red]Unknown sector: {sector_name}[/red]")
            console.print("\nAvailable sectors:")
            for sector in SECTOR_STOCKS.keys():
                console.print(f"  • sector:{sector}")
            raise typer.Exit(1)
    elif universe_lower in universes:
        symbols, universe_name = universes[universe_lower]
    else:
        console.print(f"[red]Unknown universe: {universe}[/red]")
        console.print("\nAvailable universes:")
        for name, (_, desc) in universes.items():
            console.print(f"  • {name}: {desc}")
        console.print("\nSector universes:")
        for sector in SECTOR_STOCKS.keys():
            console.print(f"  • sector:{sector}")
        raise typer.Exit(1)

    # Use preset or build filters
    scanner = StockScanner()

    if preset:
        if preset not in SCREENER_PRESETS:
            console.print(f"[red]Unknown preset: {preset}[/red]")
            console.print("\nAvailable presets:")
            for name, desc in scanner.get_available_presets().items():
                console.print(f"  • {name}: {desc}")
            raise typer.Exit(1)

        console.print(f"\n[bold]Running '{preset}' screener on {universe_name}...[/bold]\n")
        results = scanner.scan_with_preset(symbols, preset, limit=limit)

    else:
        # Build custom filters
        filters = ScreenerFilters()

        if rsi_oversold:
            filters.rsi_max = 30
        if rsi_overbought:
            filters.rsi_min = 70
        if above_200_sma:
            filters.above_sma_200 = True
        if macd_bullish:
            filters.macd_bullish = True
        if squeeze:
            filters.bollinger_squeeze = True

        console.print(f"\n[bold]Screening {universe_name}...[/bold]\n")
        results = scanner.scan(symbols, filters, limit=limit)

    # Display results
    if results.filters_applied:
        console.print(f"[dim]Filters: {', '.join(results.filters_applied)}[/dim]")

    console.print(f"Scanned: {results.total_scanned} stocks\n")

    if not results.matches:
        console.print("[yellow]No stocks match the criteria.[/yellow]")
        return

    # Results table
    table = Table(show_header=True, header_style="bold")
    table.add_column("Symbol")
    table.add_column("Price", justify="right")
    table.add_column("Change", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Signal")
    table.add_column("Criteria")

    # Filter out NO_EDGE signals — insufficient clarity means no actionable info
    actionable_matches = [
        m for m in results.matches
        if not (m.signal and m.signal.signal_type == SignalType.NO_EDGE)
    ]
    hidden_count = len(results.matches) - len(actionable_matches)

    for match in actionable_matches:
        change_color = "green" if match.change_percent >= 0 else "red"
        signal_color = get_signal_color(match.signal.signal_type) if match.signal else "white"
        signal_text = get_signal_label(match.signal.signal_type) if match.signal else "N/A"

        table.add_row(
            match.symbol,
            f"₹{match.current_price:.2f}",
            f"[{change_color}]{match.change_percent:+.2f}%[/{change_color}]",
            f"{match.score:.1f}",
            f"[{signal_color}]{signal_text}[/{signal_color}]",
            ", ".join(match.matching_criteria[:3]),
        )

    console.print(table)

    if hidden_count > 0:
        console.print(f"\n[dim]{hidden_count} stocks hidden (NO_EDGE — insufficient indicator clarity)[/dim]")

    if results.errors:
        error_rate = len(results.errors) / max(results.total_scanned, 1) * 100
        if error_rate > 50:
            console.print(
                f"\n[bold red]WARNING: {len(results.errors)}/{results.total_scanned} stocks "
                f"failed ({error_rate:.0f}%). Results may be unreliable — "
                f"possible rate limiting or API issues.[/bold red]"
            )
        elif error_rate > 20:
            console.print(
                f"\n[yellow]Caution: {len(results.errors)} stocks skipped ({error_rate:.0f}% failure rate)[/yellow]"
            )
        else:
            console.print(f"\n[dim]Errors: {len(results.errors)} stocks skipped[/dim]")


@app.command()
def news(
    symbol: str = typer.Option(None, "--symbol", "-s", help="Stock symbol to filter news"),
    sentiment: bool = typer.Option(True, "--sentiment/--no-sentiment", help="Include sentiment analysis"),
    limit: int = typer.Option(15, "--limit", "-l", help="Maximum articles"),
) -> None:
    """Fetch and analyze market news."""
    aggregator = NewsAggregator()
    analyzer = SentimentAnalyzer()

    if symbol:
        symbol = symbol.upper()
        console.print(f"\n[bold]News for {symbol}:[/bold]\n")
        articles = aggregator.fetch_for_symbol(symbol, limit_per_source=limit // 2)
    else:
        console.print("\n[bold]Market News:[/bold]\n")
        articles = aggregator.fetch_market_news(limit_per_source=limit // 3)

    if not articles:
        console.print("[yellow]No news articles found.[/yellow]")
        return

    # Analyze sentiment if requested
    if sentiment:
        articles = analyzer.analyze_articles(articles)
        aggregate = analyzer.get_aggregate_sentiment(articles)

        # Display aggregate sentiment
        sentiment_color = {
            "very_bullish": "bold green",
            "bullish": "green",
            "neutral": "yellow",
            "bearish": "red",
            "very_bearish": "bold red",
        }.get(aggregate.category.value, "white")

        console.print(
            f"Overall Sentiment: [{sentiment_color}]{aggregate.category.value.replace('_', ' ').title()}[/{sentiment_color}] "
            f"(Score: {aggregate.overall_score:+.2f})"
        )
        console.print(f"Breakdown: {aggregate.sentiment_ratio}\n")

    # News table
    table = Table(show_header=True, header_style="bold", show_lines=True)
    table.add_column("Title", width=50)
    table.add_column("Source", width=15)
    table.add_column("Age")
    if sentiment:
        table.add_column("Sentiment")

    for article in articles[:limit]:
        age_text = f"{article.age_hours:.0f}h" if article.age_hours < 24 else f"{article.age_hours / 24:.0f}d"

        if sentiment and article.sentiment:
            sent_color = "green" if article.sentiment.compound > 0.05 else "red" if article.sentiment.compound < -0.05 else "yellow"
            sent_text = f"[{sent_color}]{article.sentiment.compound:+.2f}[/{sent_color}]"
            table.add_row(article.title[:50], article.source, age_text, sent_text)
        else:
            table.add_row(article.title[:50], article.source, age_text)

    console.print(table)


@app.command()
def watchlist(
    action: str = typer.Argument("list", help="Action: list, add, remove, show"),
    name: str = typer.Option("default", "--name", "-n", help="Watchlist name"),
    symbol: str = typer.Option(None, "--symbol", "-s", help="Symbol for add/remove"),
) -> None:
    """Manage watchlists (basic file-based storage)."""
    import json
    from pathlib import Path

    from stock_predictor.infrastructure.config.settings import get_settings

    settings = get_settings()
    watchlist_file = settings.watchlist_dir / f"{name}.json"

    def load_watchlist() -> list[str]:
        if watchlist_file.exists():
            return json.loads(watchlist_file.read_text())
        return []

    def save_watchlist(symbols: list[str]) -> None:
        watchlist_file.write_text(json.dumps(symbols, indent=2))

    if action == "list":
        # List all watchlists
        console.print("\n[bold]Watchlists:[/bold]\n")
        watchlist_files = list(settings.watchlist_dir.glob("*.json"))

        if not watchlist_files:
            console.print("[dim]No watchlists found. Create one with: watchlist add -n mylist -s RELIANCE[/dim]")
            return

        for wf in watchlist_files:
            symbols = json.loads(wf.read_text())
            console.print(f"  • {wf.stem}: {len(symbols)} symbols")

    elif action == "add":
        if not symbol:
            console.print("[red]Please specify a symbol with --symbol[/red]")
            raise typer.Exit(1)

        symbols = load_watchlist()
        symbol = symbol.upper()

        if symbol in symbols:
            console.print(f"[yellow]{symbol} is already in '{name}'[/yellow]")
        else:
            symbols.append(symbol)
            save_watchlist(symbols)
            console.print(f"[green]Added {symbol} to '{name}'[/green]")

    elif action == "remove":
        if not symbol:
            console.print("[red]Please specify a symbol with --symbol[/red]")
            raise typer.Exit(1)

        symbols = load_watchlist()
        symbol = symbol.upper()

        if symbol not in symbols:
            console.print(f"[yellow]{symbol} is not in '{name}'[/yellow]")
        else:
            symbols.remove(symbol)
            save_watchlist(symbols)
            console.print(f"[green]Removed {symbol} from '{name}'[/green]")

    elif action == "show":
        symbols = load_watchlist()

        if not symbols:
            console.print(f"[yellow]Watchlist '{name}' is empty[/yellow]")
            return

        console.print(f"\n[bold]Watchlist: {name}[/bold]\n")

        # Get quotes
        provider = YahooDataProvider()
        quotes = provider.get_multiple_quotes(symbols)

        table = Table(show_header=True, header_style="bold")
        table.add_column("Symbol")
        table.add_column("Price", justify="right")
        table.add_column("Change", justify="right")

        for sym in symbols:
            if sym in quotes:
                q = quotes[sym]
                change_color = "green" if q.is_positive else "red"
                table.add_row(
                    sym,
                    f"₹{q.last_price:.2f}",
                    f"[{change_color}]{q.change:+.2f} ({q.change_percent:+.2f}%)[/{change_color}]",
                )
            else:
                table.add_row(sym, "N/A", "N/A")

        console.print(table)

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Available actions: list, add, remove, show")


@app.command(name="presets")
def list_presets() -> None:
    """List available screener presets."""
    console.print("\n[bold]Available Screener Presets:[/bold]\n")

    for name, preset in SCREENER_PRESETS.items():
        console.print(f"[bold cyan]{name}[/bold cyan]")
        console.print(f"  {preset.description}\n")


@app.command(name="ai-analyze")
def ai_analyze(
    symbol: str = typer.Argument(..., help="Stock symbol to analyze"),
) -> None:
    """Get AI-powered deep analysis for a stock using Claude."""
    import os

    symbol = symbol.upper()

    if not os.getenv("ANTHROPIC_API_KEY"):
        console.print("[red]Error: ANTHROPIC_API_KEY environment variable not set.[/red]")
        console.print("\nTo use AI analysis, set your Anthropic API key:")
        console.print("  export ANTHROPIC_API_KEY='your-api-key'")
        console.print("\nGet an API key at: https://console.anthropic.com/")
        raise typer.Exit(1)

    console.print(f"\n[bold]AI Analysis for {symbol}[/bold]")
    console.print("[dim]Gathering technical, fundamental, and news data...[/dim]\n")

    try:
        from stock_predictor.analysis.ai_advisor import AIAdvisor

        advisor = AIAdvisor()

        with console.status("[bold green]Claude is analyzing..."):
            recommendation = advisor.analyze_stock(symbol)

        # Display recommendation
        action_colors = {
            "STRONG BUY": "bold green",
            "BUY": "green",
            "HOLD": "yellow",
            "SELL": "red",
            "AVOID": "bold red",
        }
        action_color = action_colors.get(recommendation.action, "white")

        # Main recommendation panel
        panel_content = f"""[{action_color}]{recommendation.action}[/{action_color}]
Confidence: {recommendation.confidence}%
Risk Level: {recommendation.risk_level}
Time Horizon: {recommendation.time_horizon}"""

        console.print(Panel(panel_content, title=f"[bold]AI Recommendation for {symbol}[/bold]", border_style=action_color))

        # Price targets
        if recommendation.target_price or recommendation.stop_loss:
            console.print("\n[bold]Price Targets:[/bold]")
            targets_table = Table(show_header=False, box=None)
            targets_table.add_column("Label", style="dim")
            targets_table.add_column("Value", style="bold")

            if recommendation.target_price:
                targets_table.add_row("Target Price", f"₹{recommendation.target_price:,.2f}")
            if recommendation.stop_loss:
                targets_table.add_row("Stop Loss", f"₹{recommendation.stop_loss:,.2f}")

            console.print(targets_table)

        # Key reasons
        console.print("\n[bold green]Why to consider:[/bold green]")
        for reason in recommendation.key_reasons:
            console.print(f"  ✓ {reason}")

        # Risks
        console.print("\n[bold red]Risks to watch:[/bold red]")
        for risk in recommendation.risks:
            console.print(f"  ⚠ {risk}")

        # Summary
        console.print(f"\n[bold]Summary:[/bold]\n{recommendation.summary}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command(name="smart-picks")
def smart_picks(
    risk: str = typer.Option("medium", "--risk", "-r", help="Risk tolerance: low, medium, high"),
    budget: float = typer.Option(None, "--budget", "-b", help="Investment budget in INR"),
) -> None:
    """Get AI-powered stock picks for today using Claude."""
    import os

    if not os.getenv("ANTHROPIC_API_KEY"):
        console.print("[red]Error: ANTHROPIC_API_KEY environment variable not set.[/red]")
        console.print("\nTo use AI picks, set your Anthropic API key:")
        console.print("  export ANTHROPIC_API_KEY='your-api-key'")
        console.print("\nGet an API key at: https://console.anthropic.com/")
        raise typer.Exit(1)

    console.print("\n[bold]AI-Powered Smart Picks for Today[/bold]")
    console.print(f"[dim]Risk Tolerance: {risk.title()}[/dim]")
    if budget:
        console.print(f"[dim]Budget: ₹{budget:,.0f}[/dim]")
    console.print()

    try:
        from stock_predictor.analysis.ai_advisor import AIAdvisor

        advisor = AIAdvisor()

        with console.status("[bold green]Claude is analyzing the market and finding opportunities..."):
            analysis = advisor.get_todays_picks(risk_tolerance=risk, budget=budget)

        # Market overview
        sentiment_colors = {
            "Bullish": "green",
            "Bearish": "red",
            "Neutral": "yellow",
        }
        sent_color = sentiment_colors.get(analysis.market_sentiment, "white")

        console.print(Panel(
            f"[bold]Sentiment:[/bold] [{sent_color}]{analysis.market_sentiment}[/{sent_color}]\n\n"
            f"{analysis.market_summary}\n\n"
            f"[bold]Strategy:[/bold] {analysis.strategy_advice}",
            title="[bold]Market Analysis[/bold]",
        ))

        # Sectors
        console.print("\n[bold]Sector Outlook:[/bold]")
        console.print(f"  [green]Recommended:[/green] {', '.join(analysis.recommended_sectors)}")
        if analysis.sectors_to_avoid:
            console.print(f"  [red]Avoid:[/red] {', '.join(analysis.sectors_to_avoid)}")

        # Top picks
        if not analysis.top_picks:
            console.print("\n[yellow]No picks today — AI recommends staying out.[/yellow]")
            console.print(f"[dim]Strategy: {analysis.strategy_advice}[/dim]")
            console.print("\n[dim]Disclaimer: AI-generated analysis, not financial advice. Always do your own research.[/dim]")
            return

        console.print("\n[bold]Today's Top Picks:[/bold]\n")

        for i, pick in enumerate(analysis.top_picks, 1):
            action_colors = {
                "STRONG BUY": "bold green",
                "BUY": "green",
                "HOLD": "yellow",
            }
            action_color = action_colors.get(pick.action, "white")

            console.print(f"[bold cyan]#{i} {pick.symbol}[/bold cyan] - [{action_color}]{pick.action}[/{action_color}] (Confidence: {pick.confidence}%)")

            picks_table = Table(show_header=False, box=None, padding=(0, 2))
            picks_table.add_column("Label", style="dim", width=12)
            picks_table.add_column("Value")

            if pick.target_price:
                picks_table.add_row("Target", f"₹{pick.target_price:,.2f}")
            if pick.stop_loss:
                picks_table.add_row("Stop Loss", f"₹{pick.stop_loss:,.2f}")
            picks_table.add_row("Time", pick.time_horizon)
            picks_table.add_row("Risk", pick.risk_level)

            console.print(picks_table)
            console.print(f"  [dim]{pick.summary}[/dim]")
            console.print()

        console.print("[dim]Disclaimer: These are AI-generated suggestions, not financial advice. Always do your own research.[/dim]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command(name="fundamentals")
def show_fundamentals(
    symbol: str = typer.Argument(..., help="Stock symbol"),
) -> None:
    """Show fundamental analysis for a stock."""
    symbol = symbol.upper()

    console.print(f"\n[bold]Fundamental Analysis: {symbol}[/bold]\n")

    try:
        from stock_predictor.analysis.fundamentals import FundamentalFetcher

        fetcher = FundamentalFetcher()
        fundamentals = fetcher.get_fundamentals(symbol)
        summary = fundamentals.to_summary_dict()

        # Company info
        console.print(f"[bold]{summary['name']}[/bold]")
        console.print(f"Sector: {summary['sector']} | Market Cap: {summary['market_cap_cr']}")
        console.print()

        # Valuation metrics
        val_table = Table(title="Valuation", show_header=True, header_style="bold")
        val_table.add_column("Metric")
        val_table.add_column("Value", justify="right")

        val_table.add_row("Current Price", summary["current_price"])
        val_table.add_row("P/E Ratio", summary["pe_ratio"])
        val_table.add_row("P/B Ratio", summary["pb_ratio"])
        val_table.add_row("52W High", summary["52w_high"])
        val_table.add_row("52W Low", summary["52w_low"])
        val_table.add_row("Analyst Target", summary["target_price"])

        console.print(val_table)

        # Quality metrics
        qual_table = Table(title="Quality & Health", show_header=True, header_style="bold")
        qual_table.add_column("Metric")
        qual_table.add_column("Value", justify="right")

        qual_table.add_row("ROE", summary["roe"])
        qual_table.add_row("Debt/Equity", summary["debt_to_equity"])
        qual_table.add_row("Dividend Yield", summary["dividend_yield"])
        qual_table.add_row("Analyst Rating", summary["analyst_rating"])

        console.print(qual_table)

        # Scores
        val_score = summary["valuation_score"]
        qual_score = summary["quality_score"]

        val_color = "green" if val_score >= 60 else "yellow" if val_score >= 40 else "red"
        qual_color = "green" if qual_score >= 60 else "yellow" if qual_score >= 40 else "red"

        console.print(f"\n[bold]Valuation Score:[/bold] [{val_color}]{val_score:.0f}/100[/{val_color}]")
        console.print(f"  {summary['valuation_reason']}")

        console.print(f"\n[bold]Quality Score:[/bold] [{qual_color}]{qual_score:.0f}/100[/{qual_color}]")
        console.print(f"  {summary['quality_reason']}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


# Import pandas for the analyze command
import pandas as pd


@app.command(name="alert")
def manage_alerts(
    action: str = typer.Argument("list", help="Action: add, list, check, delete, clear"),
    symbol: str = typer.Option(None, "--symbol", "-s", help="Stock symbol"),
    price: float = typer.Option(None, "--price", "-p", help="Target price"),
    alert_type: str = typer.Option("above", "--type", "-t", help="Alert type: above or below"),
    note: str = typer.Option("", "--note", "-n", help="Note for the alert"),
    alert_id: str = typer.Option(None, "--id", help="Alert ID for delete action"),
) -> None:
    """Manage price alerts for stocks.

    Examples:
        nse-predict alert add -s RELIANCE -p 2500 -t above -n "Breakout level"
        nse-predict alert add -s INFY -p 1400 -t below -n "Buy on dip"
        nse-predict alert list
        nse-predict alert check
        nse-predict alert delete --id abc123
    """
    from stock_predictor.infrastructure.storage.alerts import (
        AlertManager,
        AlertStatus,
        AlertType,
    )

    manager = AlertManager()

    if action == "add":
        if not symbol or not price:
            console.print("[red]Error: --symbol and --price are required for adding alerts[/red]")
            console.print("Example: nse-predict alert add -s RELIANCE -p 2500 -t above")
            raise typer.Exit(1)

        try:
            atype = AlertType.ABOVE if alert_type.lower() == "above" else AlertType.BELOW
            alert = manager.create_alert(
                symbol=symbol,
                alert_type=atype,
                target_price=price,
                note=note,
            )

            direction = "rises above" if atype == AlertType.ABOVE else "falls below"
            console.print(f"\n[green]Alert created![/green]")
            console.print(f"  ID: {alert.id}")
            console.print(f"  Alert when {alert.symbol} {direction} ₹{price:,.2f}")
            if note:
                console.print(f"  Note: {note}")

        except Exception as e:
            console.print(f"[red]Error creating alert: {e}[/red]")
            raise typer.Exit(1)

    elif action == "list":
        alerts = manager.get_alerts(symbol=symbol)

        if not alerts:
            console.print("\n[dim]No alerts found. Create one with:[/dim]")
            console.print("  nse-predict alert add -s RELIANCE -p 2500 -t above")
            return

        console.print("\n[bold]Price Alerts:[/bold]\n")

        # Group by status
        active = [a for a in alerts if a.status == AlertStatus.ACTIVE]
        triggered = [a for a in alerts if a.status == AlertStatus.TRIGGERED]

        if active:
            table = Table(title="Active Alerts", show_header=True, header_style="bold")
            table.add_column("ID")
            table.add_column("Symbol")
            table.add_column("Condition")
            table.add_column("Target", justify="right")
            table.add_column("Note")
            table.add_column("Created")

            for alert in active:
                direction = ">" if alert.alert_type == AlertType.ABOVE else "<"
                age = (datetime.now() - alert.created_at).days
                age_text = f"{age}d ago" if age > 0 else "today"

                table.add_row(
                    alert.id,
                    alert.symbol,
                    direction,
                    f"₹{alert.target_price:,.2f}",
                    alert.note[:20] if alert.note else "-",
                    age_text,
                )

            console.print(table)

        if triggered:
            console.print()
            trig_table = Table(title="Triggered Alerts", show_header=True, header_style="bold yellow")
            trig_table.add_column("ID")
            trig_table.add_column("Symbol")
            trig_table.add_column("Target", justify="right")
            trig_table.add_column("Triggered At", justify="right")
            trig_table.add_column("Price", justify="right")

            for alert in triggered:
                trig_table.add_row(
                    alert.id,
                    alert.symbol,
                    f"₹{alert.target_price:,.2f}",
                    f"₹{alert.triggered_price:,.2f}" if alert.triggered_price else "-",
                    alert.triggered_at.strftime("%Y-%m-%d %H:%M") if alert.triggered_at else "-",
                )

            console.print(trig_table)
            console.print("\n[dim]Clear triggered alerts with: nse-predict alert clear[/dim]")

    elif action == "check":
        active_alerts = manager.get_alerts(active_only=True)

        if not active_alerts:
            console.print("\n[dim]No active alerts to check.[/dim]")
            return

        console.print("\n[bold]Checking Price Alerts...[/bold]\n")

        # Get symbols to check
        symbols = manager.get_symbols_to_check()

        # Fetch current prices
        provider = YahooDataProvider()
        quotes = provider.get_multiple_quotes(symbols)
        prices = {sym: float(q.last_price) for sym, q in quotes.items()}

        # Check alerts
        triggered = manager.check_alerts(prices)

        # Show current status
        table = Table(show_header=True, header_style="bold")
        table.add_column("Symbol")
        table.add_column("Current", justify="right")
        table.add_column("Target", justify="right")
        table.add_column("Type")
        table.add_column("Distance", justify="right")
        table.add_column("Status")

        for alert in active_alerts:
            current = prices.get(alert.symbol)
            if current is None:
                continue

            direction = ">" if alert.alert_type == AlertType.ABOVE else "<"
            distance = ((alert.target_price - current) / current) * 100

            if alert in triggered:
                status = "[bold green]TRIGGERED![/bold green]"
            else:
                status = "[dim]Watching[/dim]"

            dist_color = "green" if abs(distance) < 2 else "yellow" if abs(distance) < 5 else "dim"

            table.add_row(
                alert.symbol,
                f"₹{current:,.2f}",
                f"₹{alert.target_price:,.2f}",
                direction,
                f"[{dist_color}]{distance:+.1f}%[/{dist_color}]",
                status,
            )

        console.print(table)

        if triggered:
            console.print(f"\n[bold green]🔔 {len(triggered)} alert(s) triggered![/bold green]")
            for alert in triggered:
                direction = "rose above" if alert.alert_type == AlertType.ABOVE else "fell below"
                console.print(f"  • {alert.symbol} {direction} ₹{alert.target_price:,.2f}")
                if alert.note:
                    console.print(f"    Note: {alert.note}")

    elif action == "delete":
        if not alert_id:
            console.print("[red]Error: --id is required for deleting alerts[/red]")
            console.print("Example: nse-predict alert delete --id abc123")
            raise typer.Exit(1)

        if manager.delete_alert(alert_id):
            console.print(f"[green]Alert {alert_id} deleted.[/green]")
        else:
            console.print(f"[yellow]Alert {alert_id} not found.[/yellow]")

    elif action == "clear":
        cleared = manager.clear_triggered()
        if cleared > 0:
            console.print(f"[green]Cleared {cleared} triggered alert(s).[/green]")
        else:
            console.print("[dim]No triggered alerts to clear.[/dim]")

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Available actions: add, list, check, delete, clear")


# Import datetime for alerts
from datetime import datetime


@app.command(name="high-confidence")
def high_confidence_picks(
    universe: str = typer.Option(
        "nifty50",
        "--universe",
        "-u",
        help="Stock universe: nifty50, next50, fno",
    ),
    budget: float = typer.Option(None, "--budget", "-b", help="Investment budget in INR"),
    limit: int = typer.Option(5, "--limit", "-l", help="Maximum results"),
) -> None:
    """Find high-confidence stock picks with strict multi-filter validation.

    This command uses a more rigorous stock selection process that requires
    multiple confirmations before recommending a stock:

    1. ADX Trend Strength - Only picks stocks in strong trends (ADX > 20)
    2. Volume Confirmation - Requires volume to support price action
    3. Moving Average Filter - Price must be above 200 & 50 SMA
    4. MACD Momentum - Requires bullish momentum
    5. RSI Filter - Must be in healthy range (40-70)

    This helps avoid false signals and premature stop-loss hits.

    Examples:
        nse-predict high-confidence -u nifty50 -b 10000
        nse-predict high-confidence -u fno -l 10
    """
    from stock_predictor.analysis.high_confidence_picker import (
        ConfidenceLevel,
        HighConfidencePicker,
    )

    # Universe mapping
    universes = {
        "nifty50": (NIFTY50_SYMBOLS, "Nifty 50"),
        "next50": (NIFTY_NEXT50_SYMBOLS, "Nifty Next 50"),
        "midcap": (NIFTY_MIDCAP_SYMBOLS, "Nifty Midcap"),
        "fno": (FNO_STOCKS, "F&O Stocks"),
    }

    universe_lower = universe.lower()
    if universe_lower not in universes:
        console.print(f"[red]Unknown universe: {universe}[/red]")
        console.print("Available: nifty50, next50, midcap, fno")
        raise typer.Exit(1)

    symbols, universe_name = universes[universe_lower]

    console.print(f"\n[bold]High-Confidence Stock Picker[/bold]")
    console.print(f"[dim]Scanning {universe_name} with strict multi-filter validation...[/dim]\n")

    if budget:
        console.print(f"[dim]Budget: ₹{budget:,.0f}[/dim]\n")

    picker = HighConfidencePicker()

    with console.status("[bold green]Analyzing stocks with 6 filters (long + short)..."):
        both_sides = picker.scan_both_sides(symbols, min_confidence=ConfidenceLevel.MEDIUM)
        long_picks = both_sides["long"]
        short_picks = both_sides["short"]

    picks = long_picks  # Primary display

    if not picks and not short_picks:
        console.print("[yellow]No stocks pass all filters right now.[/yellow]")
        console.print("[dim]This means the market may be in a weak/uncertain phase.[/dim]")
        console.print("[dim]Consider waiting for better setups or checking again later.[/dim]")
        return

    # Limit results
    picks = picks[:limit]

    # Display results
    total_found = len(long_picks) + len(short_picks)
    console.print(f"[bold]Found {len(long_picks)} long + {len(short_picks)} short picks:[/bold]\n")

    for i, pick in enumerate(picks, 1):
        # Confidence level colors
        conf_colors = {
            ConfidenceLevel.VERY_HIGH: "bold green",
            ConfidenceLevel.HIGH: "green",
            ConfidenceLevel.MEDIUM: "yellow",
            ConfidenceLevel.LOW: "red",
        }
        conf_color = conf_colors.get(pick.confidence_level, "white")

        # Header
        console.print(f"[bold cyan]#{i} {pick.symbol}[/bold cyan] - [{conf_color}]{pick.recommendation}[/{conf_color}]")
        console.print(f"    Confidence: [{conf_color}]{pick.confidence_score:.0f}%[/{conf_color}] ({pick.confidence_level.value.replace('_', ' ').title()})")

        # Key metrics
        metrics_table = Table(show_header=False, box=None, padding=(0, 2))
        metrics_table.add_column("Label", style="dim", width=14)
        metrics_table.add_column("Value")

        metrics_table.add_row("Current Price", f"₹{pick.current_price:,.2f}")
        metrics_table.add_row("Entry", f"₹{pick.entry_price:,.2f}")
        metrics_table.add_row("Stop Loss", f"₹{pick.stop_loss:,.2f} ({pick.risk_percent:.1f}% risk)")
        metrics_table.add_row("Target 1", f"₹{pick.target_1:,.2f} ({pick.reward_percent:.1f}% gain)")
        metrics_table.add_row("Target 2", f"₹{pick.target_2:,.2f}")
        metrics_table.add_row("Risk:Reward", f"1:{pick.risk_reward_ratio:.1f}")
        metrics_table.add_row("ADX", f"{pick.adx_value:.1f} ({pick.regime.value.replace('_', ' ').title()})")
        metrics_table.add_row("RSI", f"{pick.rsi_value:.1f}")
        metrics_table.add_row("Volume", f"{pick.volume_ratio:.1f}x avg")

        console.print(metrics_table)

        # Filters passed
        console.print(f"\n    [green]✓ Filters Passed ({len(pick.filters_passed)}/5):[/green]")
        for f in pick.filters_passed:
            console.print(f"      • {f.name}: {f.reason}")

        # Filters failed
        if pick.filters_failed:
            console.print(f"    [red]✗ Filters Failed ({len(pick.filters_failed)}/5):[/red]")
            for f in pick.filters_failed:
                console.print(f"      • {f.name}: {f.reason}")

        # Budget calculation
        if budget:
            shares = int(budget / float(pick.current_price))
            if shares > 0:
                total_cost = shares * float(pick.current_price)
                potential_profit = shares * float(pick.target_1 - pick.current_price)
                potential_loss = shares * float(pick.current_price - pick.stop_loss)
                console.print(f"\n    [bold]With ₹{budget:,.0f} budget:[/bold]")
                console.print(f"      Buy {shares} shares @ ₹{pick.current_price} = ₹{total_cost:,.0f}")
                console.print(f"      Potential profit: [green]₹{potential_profit:,.0f}[/green]")
                console.print(f"      Max loss: [red]₹{potential_loss:,.0f}[/red]")

        console.print()

    # Display short picks if any
    if short_picks:
        console.print(f"\n[bold red]Short Opportunities ({len(short_picks)}):[/bold red]\n")
        for i, pick in enumerate(short_picks[:limit], 1):
            conf_colors = {
                ConfidenceLevel.VERY_HIGH: "bold red",
                ConfidenceLevel.HIGH: "red",
                ConfidenceLevel.MEDIUM: "yellow",
            }
            conf_color = conf_colors.get(pick.confidence_level, "white")

            console.print(f"[bold cyan]#{i} {pick.symbol}[/bold cyan] - [{conf_color}]{pick.recommendation}[/{conf_color}]")
            console.print(f"    Confidence: [{conf_color}]{pick.confidence_score:.0f}%[/{conf_color}]")

            metrics_table = Table(show_header=False, box=None, padding=(0, 2))
            metrics_table.add_column("Label", style="dim", width=14)
            metrics_table.add_column("Value")
            metrics_table.add_row("Current Price", f"₹{pick.current_price:,.2f}")
            metrics_table.add_row("Entry (Short)", f"₹{pick.entry_price:,.2f}")
            metrics_table.add_row("Stop Loss", f"₹{pick.stop_loss:,.2f} ({pick.risk_percent:.1f}% risk)")
            metrics_table.add_row("Target 1", f"₹{pick.target_1:,.2f} ({pick.reward_percent:.1f}% gain)")
            metrics_table.add_row("Risk:Reward", f"1:{pick.risk_reward_ratio:.1f}")
            console.print(metrics_table)
            console.print()

    # Risk-based position sizing if budget provided
    if budget and picks:
        from stock_predictor.analysis.risk_manager import RiskManager
        console.print("\n[bold]ATR-Based Position Sizing:[/bold]")
        rm = RiskManager(total_capital=budget, risk_per_trade=1.0)
        for pick in picks[:3]:
            try:
                data = YahooDataProvider().get_historical(pick.symbol, period="1y")
                pos = rm.calculate_position_size(pick.symbol, data)
                if pos.shares > 0:
                    console.print(
                        f"  {pick.symbol}: {pos.shares} shares @ ₹{pos.entry_price:,.2f} "
                        f"= ₹{pos.position_value:,.0f} ({pos.allocation_percent:.1f}% of capital, "
                        f"max loss ₹{pos.total_risk:,.0f})"
                    )
            except Exception:
                pass

    console.print("\n[dim]Note: These picks have passed strict multi-filter validation.[/dim]")
    console.print("[dim]Always set a stop-loss and manage your position size.[/dim]")


@app.command(name="institutional")
def institutional_data(
    symbol: str = typer.Option(None, "--symbol", "-s", help="Stock symbol for promoter data"),
) -> None:
    """Show FII/DII flows and promoter holding data from NSE."""
    from stock_predictor.analysis.institutional_scorer import InstitutionalScorer
    from stock_predictor.infrastructure.data_providers.nse_provider import NSEDataProvider

    console.print("\n[bold]Institutional Data (from NSE)[/bold]\n")

    nse = NSEDataProvider()

    # FII/DII flows (market-wide)
    try:
        fii_dii = nse.get_fii_dii_activity()
        if fii_dii:
            latest = fii_dii[0]
            fii_table = Table(title="FII/DII Activity", show_header=True, header_style="bold")
            fii_table.add_column("Category")
            fii_table.add_column("Buy (Cr)", justify="right")
            fii_table.add_column("Sell (Cr)", justify="right")
            fii_table.add_column("Net (Cr)", justify="right")

            fii_color = "green" if latest.fii_net_value >= 0 else "red"
            dii_color = "green" if latest.dii_net_value >= 0 else "red"

            fii_table.add_row(
                "FII/FPI",
                f"₹{latest.fii_buy_value:,.0f}",
                f"₹{latest.fii_sell_value:,.0f}",
                f"[{fii_color}]₹{latest.fii_net_value:+,.0f}[/{fii_color}]",
            )
            fii_table.add_row(
                "DII",
                f"₹{latest.dii_buy_value:,.0f}",
                f"₹{latest.dii_sell_value:,.0f}",
                f"[{dii_color}]₹{latest.dii_net_value:+,.0f}[/{dii_color}]",
            )

            console.print(fii_table)
        else:
            console.print("[yellow]FII/DII data unavailable[/yellow]")
    except Exception as e:
        console.print(f"[red]Error fetching FII/DII data: {e}[/red]")

    # Promoter holding (per-stock)
    if symbol:
        symbol = symbol.upper()
        console.print()

        try:
            promoter = nse.get_promoter_holding(symbol)
            if promoter:
                prom_table = Table(
                    title=f"Shareholding Pattern: {symbol} ({promoter.quarter})",
                    show_header=True,
                    header_style="bold",
                )
                prom_table.add_column("Category")
                prom_table.add_column("Holding %", justify="right")

                prom_table.add_row("Promoter & Group", f"{promoter.promoter_holding_pct:.2f}%")
                pledge_color = (
                    "red" if promoter.pledge_risk in ("critical", "high")
                    else "yellow" if promoter.pledge_risk == "moderate"
                    else "green"
                )
                prom_table.add_row(
                    "  Pledged",
                    f"[{pledge_color}]{promoter.promoter_pledge_pct:.2f}%[/{pledge_color}]",
                )
                prom_table.add_row("Institutions", f"{promoter.institution_holding_pct:.2f}%")
                prom_table.add_row("Public", f"{promoter.public_holding_pct:.2f}%")

                console.print(prom_table)

                if promoter.pledge_risk in ("critical", "high"):
                    console.print(
                        f"\n[bold red]Warning: Promoter pledge level is {promoter.pledge_risk.upper()}[/bold red]"
                    )
            else:
                console.print(f"[yellow]No shareholding data for {symbol}[/yellow]")
        except Exception as e:
            console.print(f"[red]Error fetching promoter data: {e}[/red]")

    # Bulk deals
    try:
        deals = nse.get_bulk_deals(symbol)
        if deals:
            console.print()
            deals_table = Table(title="Recent Bulk/Block Deals", show_header=True, header_style="bold")
            deals_table.add_column("Symbol")
            deals_table.add_column("Client")
            deals_table.add_column("Type")
            deals_table.add_column("Qty", justify="right")
            deals_table.add_column("Price", justify="right")

            for deal in deals[:10]:
                deal_color = "green" if deal.deal_type.upper() in ("BUY", "B") else "red"
                deals_table.add_row(
                    deal.symbol,
                    deal.client_name[:25],
                    f"[{deal_color}]{deal.deal_type}[/{deal_color}]",
                    f"{deal.quantity:,}",
                    f"₹{deal.price:,.2f}",
                )

            console.print(deals_table)
    except Exception as e:
        console.print(f"[dim]Bulk deal data unavailable: {e}[/dim]")

    # Score summary
    if symbol:
        try:
            nse_data = nse.get_all_data(symbol)
            scorer = InstitutionalScorer()
            breakdown = scorer.score(nse_data)

            score_color = "green" if breakdown.combined_score > 0.1 else "red" if breakdown.combined_score < -0.1 else "yellow"
            console.print(f"\n[bold]Institutional Score:[/bold] [{score_color}]{breakdown.combined_score:+.3f}[/{score_color}]")
            console.print(f"  FII/DII: {breakdown.fii_dii_score:+.3f} | Promoter: {breakdown.promoter_score:+.3f} | Bulk Deals: {breakdown.bulk_deal_score:+.3f}")
        except Exception:
            pass


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
