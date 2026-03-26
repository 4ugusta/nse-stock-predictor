"""Streamlit web dashboard for NSE Stock Predictor."""

import streamlit as st

from stock_predictor.infrastructure.data_providers.yahoo_provider import YahooDataProvider

# Page configuration
st.set_page_config(
    page_title="NSE Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .positive { color: #00c853; }
    .negative { color: #ff1744; }
    </style>
    """,
    unsafe_allow_html=True,
)


def main() -> None:
    """Main dashboard entry point."""
    # Sidebar navigation
    st.sidebar.title("📈 NSE Predictor")

    page = st.sidebar.radio(
        "Navigation",
        ["Market Overview", "Stock Analysis", "Screener", "News & Sentiment", "Watchlist", "Portfolio Planner"],
        index=0,
    )

    # Market status
    provider = YahooDataProvider()
    is_open = provider.is_market_open()
    status = "🟢 Open" if is_open else "🔴 Closed"
    st.sidebar.markdown(f"**Market:** {status}")
    st.sidebar.markdown("---")

    # Route to pages
    if page == "Market Overview":
        show_market_overview()
    elif page == "Stock Analysis":
        show_stock_analysis()
    elif page == "Screener":
        show_screener()
    elif page == "News & Sentiment":
        show_news_sentiment()
    elif page == "Watchlist":
        show_watchlist()
    elif page == "Portfolio Planner":
        show_portfolio_planner()


def show_market_overview() -> None:
    """Display market overview page."""
    st.title("📊 Market Overview")

    provider = YahooDataProvider()

    # Key indices
    col1, col2, col3 = st.columns(3)

    with col1:
        try:
            nifty = provider.get_index_data("NIFTY50")
            delta_color = "normal" if nifty.change >= 0 else "inverse"
            st.metric(
                "NIFTY 50",
                f"₹{nifty.value:,.2f}",
                f"{nifty.change:+.2f} ({nifty.change_percent:+.2f}%)",
                delta_color=delta_color,
            )
        except Exception:
            st.metric("NIFTY 50", "N/A", "")

    with col2:
        try:
            banknifty = provider.get_index_data("BANKNIFTY")
            delta_color = "normal" if banknifty.change >= 0 else "inverse"
            st.metric(
                "BANK NIFTY",
                f"₹{banknifty.value:,.2f}",
                f"{banknifty.change:+.2f} ({banknifty.change_percent:+.2f}%)",
                delta_color=delta_color,
            )
        except Exception:
            st.metric("BANK NIFTY", "N/A", "")

    with col3:
        is_open = provider.is_market_open()
        st.metric("Market Status", "OPEN" if is_open else "CLOSED", "")

    st.markdown("---")

    # Top movers
    st.subheader("Top Movers - Nifty 50")

    from stock_predictor.infrastructure.config.constants import NIFTY50_SYMBOLS

    with st.spinner("Loading quotes..."):
        quotes = provider.get_multiple_quotes(NIFTY50_SYMBOLS)

    if quotes:
        sorted_quotes = sorted(
            quotes.values(), key=lambda x: float(x.change_percent), reverse=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🟢 Top Gainers**")
            gainers_data = []
            for q in sorted_quotes[:5]:
                gainers_data.append({
                    "Symbol": q.symbol,
                    "Price": f"₹{q.last_price:.2f}",
                    "Change": f"+{q.change_percent:.2f}%",
                })
            st.dataframe(gainers_data, hide_index=True, width="stretch")

        with col2:
            st.markdown("**🔴 Top Losers**")
            losers_data = []
            for q in sorted_quotes[-5:][::-1]:
                losers_data.append({
                    "Symbol": q.symbol,
                    "Price": f"₹{q.last_price:.2f}",
                    "Change": f"{q.change_percent:.2f}%",
                })
            st.dataframe(losers_data, hide_index=True, width="stretch")


def show_stock_analysis() -> None:
    """Display stock analysis page."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from stock_predictor.analysis.performance_tracker import PerformanceTracker
    from stock_predictor.analysis.signals.signal_generator import ExternalScores, SignalGenerator
    from stock_predictor.analysis.sentiment.analyzer import SentimentAnalyzer
    from stock_predictor.core.entities.signal import TradingStyle
    from stock_predictor.infrastructure.news_providers.aggregator import NewsAggregator

    st.title("📈 Stock Analysis")

    # Input controls
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        symbol = st.text_input("Stock Symbol", value="RELIANCE", key="analysis_symbol").upper()

    with col2:
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)

    with col3:
        style = st.selectbox("Trading Style", ["Swing", "Intraday", "Positional"], index=0)

    style_map = {
        "Intraday": TradingStyle.INTRADAY,
        "Swing": TradingStyle.SWING,
        "Positional": TradingStyle.POSITIONAL,
    }
    trading_style = style_map[style]

    if st.button("Analyze", type="primary"):
        with st.spinner(f"Analyzing {symbol}..."):
            try:
                provider = YahooDataProvider()
                data = provider.get_historical(symbol, period=period)
                generator = SignalGenerator()

                # Build external scores (sentiment + institutional) like CLI does
                external_scores = None
                try:
                    aggregator = NewsAggregator()
                    analyzer = SentimentAnalyzer()
                    articles = aggregator.fetch_for_symbol(symbol, limit_per_source=5)
                    if articles:
                        articles = analyzer.analyze_articles(articles)
                        aggregate = analyzer.get_aggregate_sentiment(articles)
                        external_scores = ExternalScores(
                            sentiment_score=aggregate.overall_score,
                        )
                except Exception:
                    pass  # Sentiment is optional

                signal = generator.generate(data, symbol, trading_style, external_scores=external_scores)
                summary = generator.get_indicator_summary(data)

                # Store in session state
                st.session_state["analysis_data"] = data
                st.session_state["analysis_signal"] = signal
                st.session_state["analysis_summary"] = summary
                st.session_state["analyzed_symbol"] = symbol

                # Log signal for performance tracking
                try:
                    tracker = PerformanceTracker()
                    tracker.log_signal(
                        symbol=symbol,
                        signal_type=signal.signal_type.value,
                        confidence=signal.confidence,
                        entry_price=signal.entry_price,
                        target_price=signal.target_price,
                        stop_loss=signal.stop_loss,
                        style=trading_style.value,
                        reasons=signal.reasons,
                    )
                except Exception:
                    pass

            except Exception as e:
                st.error(f"Error analyzing {symbol}: {e}")
                return

    # Display results if available
    if "analysis_data" in st.session_state:
        data = st.session_state["analysis_data"]
        signal = st.session_state["analysis_signal"]
        summary = st.session_state["analysis_summary"]
        symbol = st.session_state["analyzed_symbol"]

        # Signal display
        signal_colors = {
            "strong_buy": "#00c853",
            "buy": "#69f0ae",
            "hold": "#ffd54f",
            "no_edge": "#b0b0b0",
            "sell": "#ff8a80",
            "strong_sell": "#ff1744",
        }
        color = signal_colors.get(signal.signal_type.value, "#ffffff")

        # Display label
        signal_label = signal.signal_type.value.upper()
        if signal.signal_type.value == "no_edge":
            signal_label = "NO EDGE - Insufficient signal clarity"

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                f"""
                <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;">
                    <h2 style="color: black; margin: 0;">{signal_label}</h2>
                    <p style="color: black; margin: 5px 0;">Indicator Agreement: {signal.confidence:.1%}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.metric("Entry Price", f"₹{signal.entry_price:.2f}")

        with col3:
            if signal.target_price:
                st.metric("Target", f"₹{signal.target_price:.2f}")
            else:
                st.metric("Target", "N/A")

        with col4:
            if signal.stop_loss:
                st.metric("Stop Loss", f"₹{signal.stop_loss:.2f}")
            else:
                st.metric("Stop Loss", "N/A")

        # Position sizing suggestion
        if signal.stop_loss and signal.entry_price and signal.signal_type.value not in ("hold", "no_edge"):
            risk_per_share = abs(signal.entry_price - signal.stop_loss)
            if risk_per_share > 0:
                with st.expander("Position Sizing (1% capital risk per trade)"):
                    sizing_data = []
                    for capital in [100_000, 500_000, 1_000_000]:
                        risk_amount = capital * 0.01
                        shares = int(risk_amount / risk_per_share)
                        if shares > 0:
                            cost = shares * signal.entry_price
                            max_loss = shares * risk_per_share
                            sizing_data.append({
                                "Capital": f"₹{capital / 100_000:.0f}L",
                                "Shares": shares,
                                "Investment": f"₹{cost:,.0f}",
                                "Max Loss": f"₹{max_loss:,.0f}",
                            })
                    if sizing_data:
                        st.dataframe(sizing_data, hide_index=True, width="stretch")

        # Historical accuracy
        try:
            accuracy = PerformanceTracker().get_accuracy(symbol=symbol)
            if accuracy.resolved_signals >= 5:
                with st.expander(f"Historical Signal Accuracy for {symbol}"):
                    acc_col1, acc_col2, acc_col3 = st.columns(3)
                    with acc_col1:
                        st.metric("Win Rate", f"{accuracy.win_rate:.1f}%")
                    with acc_col2:
                        st.metric("Avg Win", f"+{accuracy.avg_win_pct:.1f}%")
                    with acc_col3:
                        st.metric("Avg Loss", f"{accuracy.avg_loss_pct:.1f}%")
                    st.caption(
                        f"Based on {accuracy.resolved_signals} resolved signals "
                        f"({accuracy.targets_hit}W / {accuracy.stops_hit}L / {accuracy.expired}E)"
                    )
                    if accuracy.calibration_warnings:
                        for w in accuracy.calibration_warnings:
                            st.warning(f"⚠ {w}")
        except Exception:
            pass

        # Disclaimer
        st.warning(
            "**DISCLAIMER:** Indicator Agreement (shown as %) measures how many technical "
            "indicators point in the same direction — it is NOT a prediction of profit probability. "
            "An 80% agreement means 80% of indicators are aligned, not that there's an 80% chance of profit. "
            "Actual results will differ due to slippage, gaps, and execution costs (STT ~0.35% round-trip). "
            "Always do your own research. Past performance does not guarantee future results. "
            "The authors are not SEBI-registered advisors."
        )

        # Analysis reasons
        if signal.reasons:
            with st.expander("Analysis Details", expanded=True):
                for reason in signal.reasons:
                    st.markdown(f"• {reason}")

        st.markdown("---")

        # Chart with indicators
        st.subheader("📊 Price Chart")

        # Indicator toggles
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            show_sma = st.checkbox("SMA (50, 200)", value=True)
        with col2:
            show_bb = st.checkbox("Bollinger Bands", value=True)
        with col3:
            show_volume = st.checkbox("Volume", value=True)
        with col4:
            show_rsi = st.checkbox("RSI", value=True)

        # Calculate number of rows
        num_rows = 1
        row_heights = [0.6]
        if show_volume:
            num_rows += 1
            row_heights.append(0.15)
        if show_rsi:
            num_rows += 1
            row_heights.append(0.15)

        # Create subplots
        fig = make_subplots(
            rows=num_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=row_heights,
        )

        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=data["timestamp"],
                open=data["open"],
                high=data["high"],
                low=data["low"],
                close=data["close"],
                name="OHLC",
            ),
            row=1,
            col=1,
        )

        # Moving averages
        if show_sma:
            from stock_predictor.analysis.indicators.moving_averages import MovingAverageCalculator
            ma_calc = MovingAverageCalculator()

            sma_50 = ma_calc.calculate_sma(data, 50)
            sma_200 = ma_calc.calculate_sma(data, 200)

            fig.add_trace(
                go.Scatter(x=data["timestamp"], y=sma_50, name="SMA 50", line=dict(color="blue", width=1)),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(x=data["timestamp"], y=sma_200, name="SMA 200", line=dict(color="red", width=1)),
                row=1, col=1,
            )

        # Bollinger Bands
        if show_bb:
            from stock_predictor.analysis.indicators.bollinger import BollingerBandsCalculator
            bb_calc = BollingerBandsCalculator()
            bb = bb_calc.calculate(data)

            fig.add_trace(
                go.Scatter(
                    x=data["timestamp"], y=bb.upper,
                    name="BB Upper", line=dict(color="gray", width=1, dash="dash"),
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=data["timestamp"], y=bb.lower,
                    name="BB Lower", line=dict(color="gray", width=1, dash="dash"),
                    fill="tonexty", fillcolor="rgba(128, 128, 128, 0.1)",
                ),
                row=1, col=1,
            )

        current_row = 2

        # Volume
        if show_volume:
            colors = ["green" if c >= o else "red" for c, o in zip(data["close"], data["open"])]
            fig.add_trace(
                go.Bar(x=data["timestamp"], y=data["volume"], name="Volume", marker_color=colors),
                row=current_row, col=1,
            )
            current_row += 1

        # RSI
        if show_rsi:
            from stock_predictor.analysis.indicators.rsi import RSICalculator
            rsi_calc = RSICalculator()
            rsi = rsi_calc.calculate(data)

            fig.add_trace(
                go.Scatter(x=data["timestamp"], y=rsi, name="RSI", line=dict(color="purple", width=1)),
                row=current_row, col=1,
            )
            # Overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)

        fig.update_layout(
            height=600,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            title=f"{symbol} - {style} Analysis",
        )

        st.plotly_chart(fig, width="stretch")

        # Indicator summary
        st.subheader("📋 Indicator Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**RSI**")
            rsi_val = summary["rsi"]["value"]
            rsi_zone = summary["rsi"]["zone"]
            st.write(f"Value: {rsi_val:.1f}")
            st.write(f"Zone: {rsi_zone}")

        with col2:
            st.markdown("**MACD**")
            st.write(f"MACD: {summary['macd']['macd']:.2f}")
            st.write(f"Signal: {summary['macd']['signal']:.2f}")
            st.write(f"Trend: {summary['macd']['trend']}")

        with col3:
            st.markdown("**Support/Resistance**")
            if summary["support_resistance"]["nearest_support"]:
                st.write(f"Support: ₹{summary['support_resistance']['nearest_support']:.2f}")
            if summary["support_resistance"]["nearest_resistance"]:
                st.write(f"Resistance: ₹{summary['support_resistance']['nearest_resistance']:.2f}")


def show_screener() -> None:
    """Display stock screener page."""
    from stock_predictor.analysis.screener.scanner import SCREENER_PRESETS, ScreenerFilters, StockScanner
    from stock_predictor.infrastructure.config.constants import (
        ALL_STOCKS, BANKNIFTY_SYMBOLS, FNO_STOCKS, HIGH_VOLATILITY_STOCKS,
        NIFTY50_SYMBOLS, NIFTY_MIDCAP_SYMBOLS, NIFTY_NEXT50_SYMBOLS,
        NIFTY_SMALLCAP_SYMBOLS,
    )

    st.title("🔍 Stock Screener")

    # Universe selection — now supports all universes like CLI
    UNIVERSE_MAP = {
        "Nifty 50": NIFTY50_SYMBOLS,
        "Nifty Next 50": NIFTY_NEXT50_SYMBOLS,
        "Nifty Midcap": NIFTY_MIDCAP_SYMBOLS,
        "Nifty Smallcap": NIFTY_SMALLCAP_SYMBOLS,
        "Bank Nifty": BANKNIFTY_SYMBOLS,
        "F&O Stocks": FNO_STOCKS,
        "High Volatility": HIGH_VOLATILITY_STOCKS,
        "All Stocks": ALL_STOCKS,
    }

    col1, col2 = st.columns(2)

    with col1:
        universe = st.selectbox("Stock Universe", list(UNIVERSE_MAP.keys()))
        symbols = UNIVERSE_MAP[universe]

    with col2:
        preset = st.selectbox(
            "Preset",
            ["Custom"] + list(SCREENER_PRESETS.keys()),
            format_func=lambda x: x if x == "Custom" else SCREENER_PRESETS[x].name,
        )

    # Custom filters
    if preset == "Custom":
        st.subheader("Filters")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**RSI**")
            rsi_min = st.number_input("RSI Min", min_value=0, max_value=100, value=0)
            rsi_max = st.number_input("RSI Max", min_value=0, max_value=100, value=100)

        with col2:
            st.markdown("**Moving Averages**")
            above_sma_200 = st.checkbox("Above 200 SMA")
            above_sma_50 = st.checkbox("Above 50 SMA")

        with col3:
            st.markdown("**Other**")
            macd_bullish = st.checkbox("MACD Bullish")
            bb_squeeze = st.checkbox("Bollinger Squeeze")

        filters = ScreenerFilters(
            rsi_min=rsi_min if rsi_min > 0 else None,
            rsi_max=rsi_max if rsi_max < 100 else None,
            above_sma_200=above_sma_200 if above_sma_200 else None,
            above_sma_50=above_sma_50 if above_sma_50 else None,
            macd_bullish=macd_bullish if macd_bullish else None,
            bollinger_squeeze=bb_squeeze if bb_squeeze else None,
        )

    if st.button("Run Screener", type="primary"):
        scanner = StockScanner()

        with st.spinner(f"Scanning {len(symbols)} stocks..."):
            if preset != "Custom":
                results = scanner.scan_with_preset(symbols, preset)
            else:
                results = scanner.scan(symbols, filters)

        st.success(f"Scanned {results.total_scanned} stocks")

        if results.matches:
            # Filter out NO_EDGE signals — insufficient clarity means no actionable info
            from stock_predictor.core.entities.signal import SignalType as _SignalType
            actionable_matches = [
                m for m in results.matches
                if not (m.signal and m.signal.signal_type == _SignalType.NO_EDGE)
            ]

            # Display results
            data = []
            for match in actionable_matches:
                signal_type = match.signal.signal_type.value if match.signal else "N/A"
                data.append({
                    "Symbol": match.symbol,
                    "Price": f"₹{match.current_price:.2f}",
                    "Change": f"{match.change_percent:+.2f}%",
                    "Score": f"{match.score:.1f}",
                    "Signal": signal_type.upper(),
                    "Criteria": ", ".join(match.matching_criteria[:2]),
                })

            st.dataframe(data, hide_index=True, width="stretch")
            if len(results.matches) > len(actionable_matches):
                st.caption(
                    f"{len(results.matches) - len(actionable_matches)} stocks hidden "
                    "(NO_EDGE — insufficient indicator clarity for a tradeable signal)"
                )
        else:
            st.warning("No stocks match the criteria")


def show_news_sentiment() -> None:
    """Display news and sentiment page."""
    from stock_predictor.analysis.sentiment.analyzer import SentimentAnalyzer
    from stock_predictor.infrastructure.news_providers.aggregator import NewsAggregator

    st.title("📰 News & Sentiment")

    # Filters
    col1, col2 = st.columns(2)

    with col1:
        symbol = st.text_input("Stock Symbol (optional)", key="news_symbol").upper()

    with col2:
        limit = st.slider("Max Articles", 5, 30, 15)

    if st.button("Fetch News", type="primary"):
        aggregator = NewsAggregator()
        analyzer = SentimentAnalyzer()

        with st.spinner("Fetching news..."):
            if symbol:
                articles = aggregator.fetch_for_symbol(symbol, limit_per_source=limit // 2)
            else:
                articles = aggregator.fetch_market_news(limit_per_source=limit // 3)

        if not articles:
            st.warning("No news articles found")
            return

        # Analyze sentiment
        articles = analyzer.analyze_articles(articles)
        aggregate = analyzer.get_aggregate_sentiment(articles)

        # Display aggregate sentiment
        col1, col2, col3, col4 = st.columns(4)

        sentiment_colors = {
            "very_bullish": "#00c853",
            "bullish": "#69f0ae",
            "neutral": "#ffd54f",
            "bearish": "#ff8a80",
            "very_bearish": "#ff1744",
        }
        color = sentiment_colors.get(aggregate.category.value, "#ffffff")

        with col1:
            st.markdown(
                f"""
                <div style="background-color: {color}; padding: 15px; border-radius: 10px; text-align: center;">
                    <h3 style="color: black; margin: 0;">{aggregate.category.value.replace('_', ' ').title()}</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.metric("Sentiment Score", f"{aggregate.overall_score:+.2f}")

        with col3:
            st.metric("Articles", str(aggregate.article_count))

        with col4:
            st.metric("Breakdown", aggregate.sentiment_ratio)

        st.markdown("---")

        # News articles
        st.subheader("Latest Articles")

        for article in articles[:limit]:
            sentiment_text = ""
            if article.sentiment:
                sent_color = "#00c853" if article.sentiment.compound > 0.05 else "#ff1744" if article.sentiment.compound < -0.05 else "#ffd54f"
                sentiment_text = f"<span style='color: {sent_color}'>({article.sentiment.compound:+.2f})</span>"

            age_text = f"{article.age_hours:.0f}h ago" if article.age_hours < 24 else f"{article.age_hours / 24:.0f}d ago"

            st.markdown(
                f"""
                **{article.title}** {sentiment_text}

                *{article.source} - {age_text}*

                [{article.url[:50]}...]({article.url})
                """,
                unsafe_allow_html=True,
            )
            st.markdown("---")


def show_watchlist() -> None:
    """Display watchlist management page."""
    import json

    from stock_predictor.infrastructure.config.settings import get_settings

    st.title("⭐ Watchlist")

    settings = get_settings()

    # List existing watchlists
    watchlist_files = list(settings.watchlist_dir.glob("*.json"))
    watchlist_names = [f.stem for f in watchlist_files]

    col1, col2 = st.columns([2, 1])

    with col1:
        if watchlist_names:
            selected_watchlist = st.selectbox("Select Watchlist", watchlist_names)
        else:
            st.info("No watchlists found. Create one below.")
            selected_watchlist = None

    with col2:
        new_name = st.text_input("Create New Watchlist")
        if st.button("Create") and new_name:
            new_file = settings.watchlist_dir / f"{new_name}.json"
            new_file.write_text("[]")
            st.success(f"Created watchlist: {new_name}")
            st.rerun()

    if selected_watchlist:
        watchlist_file = settings.watchlist_dir / f"{selected_watchlist}.json"
        symbols = json.loads(watchlist_file.read_text())

        st.markdown("---")

        # Add symbol
        col1, col2 = st.columns([3, 1])
        with col1:
            new_symbol = st.text_input("Add Symbol", key="add_symbol").upper()
        with col2:
            st.write("")
            st.write("")
            if st.button("Add") and new_symbol:
                if new_symbol not in symbols:
                    symbols.append(new_symbol)
                    watchlist_file.write_text(json.dumps(symbols, indent=2))
                    st.success(f"Added {new_symbol}")
                    st.rerun()

        # Display watchlist
        if symbols:
            provider = YahooDataProvider()

            with st.spinner("Loading quotes..."):
                quotes = provider.get_multiple_quotes(symbols)

            data = []
            for sym in symbols:
                if sym in quotes:
                    q = quotes[sym]
                    data.append({
                        "Symbol": sym,
                        "Price": f"₹{q.last_price:.2f}",
                        "Change": f"{q.change:+.2f}",
                        "Change %": f"{q.change_percent:+.2f}%",
                        "Volume": f"{q.volume:,}",
                    })
                else:
                    data.append({
                        "Symbol": sym,
                        "Price": "N/A",
                        "Change": "N/A",
                        "Change %": "N/A",
                        "Volume": "N/A",
                    })

            st.dataframe(data, hide_index=True, width="stretch")

            # Remove symbol
            remove_symbol = st.selectbox("Remove Symbol", [""] + symbols)
            if st.button("Remove") and remove_symbol:
                symbols.remove(remove_symbol)
                watchlist_file.write_text(json.dumps(symbols, indent=2))
                st.success(f"Removed {remove_symbol}")
                st.rerun()
        else:
            st.info("Watchlist is empty. Add some symbols above.")


def show_portfolio_planner() -> None:
    """Portfolio Planner - full analysis & allocation using all system capabilities."""
    import plotly.graph_objects as go

    from stock_predictor.analysis.high_confidence_picker import (
        ConfidenceLevel,
        HighConfidencePicker,
    )
    from stock_predictor.analysis.market_intelligence import (
        analyze_gap,
        fetch_india_vix,
        get_fii_dii_activity,
        get_market_breadth,
    )
    from stock_predictor.analysis.risk_manager import (
        PortfolioPosition,
        RiskManager,
    )
    from stock_predictor.analysis.signals.signal_generator import SignalGenerator
    from stock_predictor.core.entities.signal import SignalType, TradingStyle
    from stock_predictor.infrastructure.config.constants import (
        ALL_STOCKS,
        FNO_STOCKS,
        NIFTY50_SYMBOLS,
        NIFTY_MIDCAP_SYMBOLS,
        NIFTY_NEXT50_SYMBOLS,
        SECTOR_STOCKS,
    )

    UNIVERSE_MAP = {
        "Nifty 50": NIFTY50_SYMBOLS,
        "Nifty 50 + Next 50": NIFTY50_SYMBOLS + NIFTY_NEXT50_SYMBOLS,
        "Nifty Midcap": NIFTY_MIDCAP_SYMBOLS,
        "F&O Stocks": FNO_STOCKS,
        "All Stocks": ALL_STOCKS,
    }

    def _get_sector(symbol: str) -> str:
        for sector, syms in SECTOR_STOCKS.items():
            if symbol in syms:
                return sector
        return "unknown"

    def _holding_action(sig, pos):
        if sig.signal_type in (SignalType.STRONG_SELL, SignalType.SELL):
            return "EXIT", "Bearish signal - consider booking profits or cutting losses"
        if sig.signal_type == SignalType.STRONG_BUY and pos.pnl_percent > -5:
            return "ADD", "Strong bullish signal - consider adding to position"
        if sig.signal_type == SignalType.BUY:
            return "HOLD", "Bullish signal - continue holding"
        if sig.signal_type == SignalType.HOLD:
            return "HOLD", "No clear direction - maintain position"
        if sig.signal_type == SignalType.NO_EDGE:
            if pos.pnl_percent < -15:
                return "REVIEW", "No edge detected + significant loss - consider stop-loss"
            return "HOLD", "Indicators conflicted - keep with stop-loss in place"
        return "HOLD", ""

    st.title("Portfolio Planner")
    st.caption("Analyze your portfolio and get a risk-managed investment plan using all system capabilities")

    # ── Session state init ──────────────────────────────────────────────────
    if "pp_portfolio" not in st.session_state:
        st.session_state["pp_portfolio"] = [
            {"id": 0, "symbol": "", "qty": 0, "avg_price": 0.0},
        ]
    if "pp_row_counter" not in st.session_state:
        st.session_state["pp_row_counter"] = 1
    if "pp_results" not in st.session_state:
        st.session_state["pp_results"] = None

    # ── Portfolio Input Form ────────────────────────────────────────────────
    st.subheader("Your Current Portfolio")

    # Image upload option
    with st.expander("Upload portfolio screenshot (auto-extract holdings)", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload 1-3 screenshots of your portfolio", type=["png", "jpg", "jpeg"],
            accept_multiple_files=True, key="pp_upload",
        )
        if uploaded_files and st.button("Extract from Images", key="pp_extract"):
            import base64
            import json as _json
            import os

            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                st.error("Set ANTHROPIC_API_KEY environment variable to use image extraction.")
            else:
                with st.spinner("Extracting portfolio from images using Claude..."):
                    try:
                        import anthropic
                        client = anthropic.Anthropic(api_key=api_key)

                        # Build image content blocks
                        content_blocks = []
                        for f in uploaded_files[:3]:
                            img_data = base64.standard_b64encode(f.read()).decode("utf-8")
                            media = "image/png" if f.name.endswith(".png") else "image/jpeg"
                            content_blocks.append({
                                "type": "image",
                                "source": {"type": "base64", "media_type": media, "data": img_data},
                            })

                        content_blocks.append({
                            "type": "text",
                            "text": (
                                "Extract the portfolio holdings from these screenshots. "
                                "Return ONLY a valid JSON array with objects containing: "
                                '"symbol" (NSE stock symbol like RELIANCE, INFY, TATASTEEL etc.), '
                                '"qty" (integer quantity/shares), '
                                '"avg_price" (average buy price as float). '
                                "Do NOT include any markdown formatting, code blocks, or explanation. "
                                "Just the raw JSON array."
                            ),
                        })

                        resp = client.messages.create(
                            model="claude-sonnet-4-5-20250929",
                            max_tokens=1024,
                            messages=[{"role": "user", "content": content_blocks}],
                        )

                        raw = resp.content[0].text.strip()
                        # Strip markdown code fences if present
                        if raw.startswith("```"):
                            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                        if raw.endswith("```"):
                            raw = raw[:-3]
                        raw = raw.strip()

                        parsed = _json.loads(raw)
                        if isinstance(parsed, list) and len(parsed) > 0:
                            new_rows = []
                            counter = st.session_state["pp_row_counter"]
                            for item in parsed:
                                sym = str(item.get("symbol", "")).upper().strip()
                                qty = int(item.get("qty", 0))
                                avg = float(item.get("avg_price", 0.0))
                                if sym and qty > 0 and avg > 0:
                                    new_rows.append({"id": counter, "symbol": sym, "qty": qty, "avg_price": round(avg, 2)})
                                    counter += 1
                            if new_rows:
                                st.session_state["pp_portfolio"] = new_rows
                                st.session_state["pp_row_counter"] = counter
                                st.session_state["pp_results"] = None
                                st.success(f"Extracted {len(new_rows)} holdings from images!")
                                st.rerun()
                            else:
                                st.warning("Could not extract valid holdings. Check your screenshot and try again.")
                        else:
                            st.warning("Could not parse holdings from image. Try a clearer screenshot.")

                    except Exception as e:
                        st.error(f"Image extraction failed: {e}")

    rows = st.session_state["pp_portfolio"]

    # Render each portfolio row
    rows_to_delete = []
    for idx, row in enumerate(rows):
        c1, c2, c3, c4 = st.columns([3, 1.5, 2, 0.5])
        with c1:
            row["symbol"] = st.text_input(
                "Symbol", value=row["symbol"], key=f"pp_sym_{row['id']}", label_visibility="collapsed" if idx > 0 else "visible",
            ).upper().strip()
        with c2:
            row["qty"] = st.number_input(
                "Qty", value=row["qty"], min_value=0, step=1, key=f"pp_qty_{row['id']}", label_visibility="collapsed" if idx > 0 else "visible",
            )
        with c3:
            row["avg_price"] = st.number_input(
                "Avg Price (INR)", value=row["avg_price"], min_value=0.0, step=0.1, format="%.2f", key=f"pp_avg_{row['id']}",
                label_visibility="collapsed" if idx > 0 else "visible",
            )
        with c4:
            if idx == 1:
                st.markdown("<br>", unsafe_allow_html=True)
            if st.button("X", key=f"pp_del_{row['id']}"):
                rows_to_delete.append(idx)

    # Process deletions
    if rows_to_delete:
        for idx in sorted(rows_to_delete, reverse=True):
            rows.pop(idx)
        st.rerun()

    # Add row button
    if st.button("+ Add Row"):
        counter = st.session_state["pp_row_counter"]
        rows.append({"id": counter, "symbol": "", "qty": 0, "avg_price": 0.0})
        st.session_state["pp_row_counter"] = counter + 1
        st.rerun()

    st.markdown("---")

    # ── Config inputs ───────────────────────────────────────────────────────
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        extra_cash = st.number_input("Extra Cash to Invest (INR)", min_value=0.0, value=0.0, step=1000.0, key="pp_cash")
    with cc2:
        universe_name = st.selectbox("Scan Universe", list(UNIVERSE_MAP.keys()), index=0, key="pp_universe")
    with cc3:
        risk_pct = st.slider("Risk Per Trade (%)", min_value=0.5, max_value=3.0, value=1.0, step=0.25, key="pp_risk")

    # ── Analyze button ──────────────────────────────────────────────────────
    if st.button("Analyze & Plan", type="primary", width="stretch"):
        # Validate
        valid_rows = [r for r in rows if r["symbol"] and r["qty"] > 0 and r["avg_price"] > 0]
        invested_value = sum(r["qty"] * r["avg_price"] for r in valid_rows)
        total_capital = invested_value + extra_cash

        if total_capital <= 0 and not valid_rows:
            st.error("Add at least one holding or provide extra cash to invest.")
            return

        provider = YahooDataProvider()

        try:
            # ── Step 1: Market Intelligence ──────────────────────────────────
            with st.spinner("Fetching market intelligence (VIX, breadth, FII/DII)..."):
                vix = fetch_india_vix()
                breadth = get_market_breadth()
                fii_dii = get_fii_dii_activity()

            # ── Step 2: Current prices ───────────────────────────────────────
            positions = []
            price_warnings = []

            if valid_rows:
                with st.spinner("Fetching current prices..."):
                    holding_symbols = [r["symbol"] for r in valid_rows]
                    quotes = provider.get_multiple_quotes(holding_symbols)

                for r in valid_rows:
                    sym = r["symbol"]
                    if sym in quotes:
                        cur_price = float(quotes[sym].last_price)
                    else:
                        cur_price = r["avg_price"]
                        price_warnings.append(sym)

                    positions.append(PortfolioPosition(
                        symbol=sym,
                        shares=r["qty"],
                        entry_price=r["avg_price"],
                        current_price=cur_price,
                        sector=_get_sector(sym),
                    ))

            # ── Step 3: Portfolio risk ───────────────────────────────────────
            total_capital = sum(p.position_value for p in positions) + extra_cash if positions else extra_cash
            rm = RiskManager(total_capital=total_capital, risk_per_trade=risk_pct, existing_positions=positions)
            portfolio_risk = rm.get_portfolio_risk()

            # ── Step 4: Signals + gap for each holding ───────────────────────
            generator = SignalGenerator()
            holding_signals = {}
            holding_gaps = {}

            if positions:
                with st.spinner("Analyzing current holdings..."):
                    for pos in positions:
                        try:
                            data = provider.get_historical(pos.symbol, period="1y", interval="1d")
                            sig = generator.generate(data, pos.symbol, TradingStyle.SWING)
                            gap = analyze_gap(data)
                            holding_signals[pos.symbol] = sig
                            holding_gaps[pos.symbol] = gap
                        except Exception:
                            holding_signals[pos.symbol] = None
                            holding_gaps[pos.symbol] = None

            # ── Step 5: High confidence scan ─────────────────────────────────
            scan_symbols = UNIVERSE_MAP.get(universe_name, NIFTY50_SYMBOLS)
            held_set = {r["symbol"] for r in valid_rows}
            scan_symbols = [s for s in scan_symbols if s not in held_set]

            long_picks = []
            short_picks = []
            pick_gaps = {}

            if extra_cash > 0 and scan_symbols:
                with st.spinner(f"Scanning {len(scan_symbols)} stocks for opportunities..."):
                    picker = HighConfidencePicker()
                    both = picker.scan_both_sides(scan_symbols, min_confidence=ConfidenceLevel.MEDIUM)
                    long_picks = both["long"]
                    short_picks = both["short"]

                # Gap analysis for top picks
                with st.spinner("Running gap analysis on top picks..."):
                    for pick in (long_picks[:8] + short_picks[:5]):
                        try:
                            data = provider.get_historical(pick.symbol, period="6mo")
                            pick_gaps[pick.symbol] = analyze_gap(data)
                        except Exception:
                            pick_gaps[pick.symbol] = None

            # ── Step 6: Risk-managed allocation ──────────────────────────────
            allocations = []
            blocked = []

            if long_picks and extra_cash > 0:
                with st.spinner("Computing risk-managed allocations..."):
                    alloc_rm = RiskManager(total_capital=total_capital, risk_per_trade=risk_pct, existing_positions=list(positions))
                    remaining = extra_cash

                    for pick in long_picks[:8]:
                        if remaining < 500:
                            break
                        try:
                            data = provider.get_historical(pick.symbol, period="1y")
                            assessment = alloc_rm.assess_trade(pick.symbol, data, is_long=True)
                            if assessment.can_trade and assessment.position_size and assessment.position_size.shares > 0:
                                ps = assessment.position_size
                                max_by_cash = int(remaining / ps.entry_price)
                                actual_shares = min(ps.shares, max_by_cash)
                                if actual_shares > 0:
                                    cost = actual_shares * ps.entry_price
                                    risk = actual_shares * ps.risk_per_share
                                    allocations.append({
                                        "symbol": pick.symbol,
                                        "shares": actual_shares,
                                        "price": ps.entry_price,
                                        "cost": cost,
                                        "stop_loss": ps.stop_loss,
                                        "risk": risk,
                                        "target_1": float(pick.target_1),
                                        "target_2": float(pick.target_2),
                                        "confidence": pick.confidence_score,
                                        "risk_reward": pick.risk_reward_ratio,
                                        "regime": pick.regime.value,
                                        "warnings": assessment.warnings,
                                        "recommendation": pick.recommendation,
                                    })
                                    remaining -= cost
                                    alloc_rm.positions.append(PortfolioPosition(
                                        symbol=pick.symbol, shares=actual_shares,
                                        entry_price=ps.entry_price, current_price=ps.entry_price,
                                        sector=_get_sector(pick.symbol),
                                    ))
                            elif assessment.blockers:
                                blocked.append({"symbol": pick.symbol, "blockers": assessment.blockers, "warnings": assessment.warnings})
                        except Exception:
                            pass

            # ── Step 7: Projected portfolio ───────────────────────────────────
            proj_positions = list(positions)
            for a in allocations:
                proj_positions.append(PortfolioPosition(
                    symbol=a["symbol"], shares=a["shares"],
                    entry_price=a["price"], current_price=a["price"],
                    sector=_get_sector(a["symbol"]),
                ))
            proj_rm = RiskManager(total_capital=total_capital, existing_positions=proj_positions)
            projected_risk = proj_rm.get_portfolio_risk()

            # ── Store results ────────────────────────────────────────────────
            st.session_state["pp_results"] = {
                "vix": vix, "breadth": breadth, "fii_dii": fii_dii,
                "positions": positions, "portfolio_risk": portfolio_risk,
                "holding_signals": holding_signals, "holding_gaps": holding_gaps,
                "long_picks": long_picks, "short_picks": short_picks,
                "pick_gaps": pick_gaps, "allocations": allocations,
                "blocked": blocked, "projected_risk": projected_risk,
                "total_capital": total_capital, "extra_cash": extra_cash,
                "price_warnings": price_warnings,
            }

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.caption("Check your inputs and try again. Market data APIs may be temporarily unavailable.")
            return

    # ── Display Results ─────────────────────────────────────────────────────
    results = st.session_state.get("pp_results")
    if results is None:
        st.info("Enter your portfolio above and click **Analyze & Plan** to get started.")
        return

    vix = results["vix"]
    breadth = results["breadth"]
    fii_dii = results["fii_dii"]
    positions = results["positions"]
    portfolio_risk = results["portfolio_risk"]
    holding_signals = results["holding_signals"]
    holding_gaps = results["holding_gaps"]
    long_picks = results["long_picks"]
    short_picks = results["short_picks"]
    pick_gaps = results["pick_gaps"]
    allocations = results["allocations"]
    blocked = results["blocked"]
    projected_risk = results["projected_risk"]
    total_capital = results["total_capital"]
    extra_cash_result = results["extra_cash"]

    for w in results.get("price_warnings", []):
        st.warning(f"Could not fetch live price for {w} - using avg price instead")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION A: Market Intelligence
    # ═══════════════════════════════════════════════════════════════════════
    st.subheader("Market Intelligence")

    mi1, mi2, mi3, mi4 = st.columns(4)

    vix_color = {"low": "normal", "normal": "normal", "elevated": "off", "high": "inverse", "extreme": "inverse"}.get(vix.regime, "normal")
    mi1.metric("India VIX", f"{vix.value:.1f}", delta=vix.regime.upper(), delta_color=vix_color)
    mi1.caption(vix.description)

    if breadth.advance_decline_ratio is not None:
        breadth_delta = f"{breadth.advances} adv / {breadth.declines} dec"
        mi2.metric("Breadth A/D", f"{breadth.advance_decline_ratio:.2f}", delta=breadth_delta,
                    delta_color="normal" if breadth.advance_decline_ratio >= 1.0 else "inverse")
    else:
        mi2.metric("Breadth", "Unavailable")
    mi2.caption(f"Signal: {breadth.breadth_signal}")

    fii_label = fii_dii.flow_signal.replace("_", " ").title()
    mi3.metric("FII Flow", fii_label,
               delta=f"Net {fii_dii.fii_index_futures_net:+,.0f} idx futures" if fii_dii.flow_signal != "unavailable" else None,
               delta_color="normal" if fii_dii.fii_index_futures_net >= 0 else "inverse")
    mi3.caption(fii_dii.description[:80] if len(fii_dii.description) > 80 else fii_dii.description)

    # Overall outlook
    combined = vix.confidence_adjustment * breadth.confidence_adjustment * fii_dii.confidence_adjustment
    if combined >= 1.0:
        outlook, outlook_color = "FAVORABLE", "green"
    elif combined >= 0.90:
        outlook, outlook_color = "NEUTRAL", "orange"
    else:
        outlook, outlook_color = "CAUTIOUS", "red"
    mi4.markdown(f"### Overall")
    mi4.markdown(f"<h2 style='color:{outlook_color}'>{outlook}</h2>", unsafe_allow_html=True)
    mi4.caption(f"Combined score: {combined:.3f}")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION B: Portfolio Health
    # ═══════════════════════════════════════════════════════════════════════
    if positions:
        st.subheader("Portfolio Health")

        h1, h2, h3, h4, h5 = st.columns(5)
        h1.metric("Total Capital", f"₹{total_capital:,.0f}")
        h2.metric("Invested", f"₹{portfolio_risk.total_invested:,.0f}")
        pnl_delta = f"{portfolio_risk.total_pnl_percent:+.1f}%"
        h3.metric("P&L", f"₹{portfolio_risk.total_pnl:,.0f}", delta=pnl_delta)
        h4.metric("Cash Available", f"₹{extra_cash_result:,.0f}")
        heat_color = "normal" if portfolio_risk.portfolio_heat < 5 else ("off" if portfolio_risk.portfolio_heat < 8 else "inverse")
        h5.metric("Portfolio Heat", f"{portfolio_risk.portfolio_heat:.1f}%", delta_color=heat_color)

        # Holdings table
        import pandas as pd
        holdings_data = []
        for pos in sorted(positions, key=lambda p: p.pnl_percent, reverse=True):
            holdings_data.append({
                "Symbol": pos.symbol,
                "Qty": pos.shares,
                "Avg Price": f"₹{pos.entry_price:,.2f}",
                "Current": f"₹{pos.current_price:,.2f}",
                "P&L": f"₹{pos.pnl:,.0f}",
                "P&L %": f"{pos.pnl_percent:+.1f}%",
                "Sector": pos.sector or _get_sector(pos.symbol),
            })
        st.dataframe(pd.DataFrame(holdings_data), hide_index=True, width="stretch")

        # Sector exposure chart + risk warnings
        sec_col, warn_col = st.columns([2, 1])

        with sec_col:
            if portfolio_risk.sector_exposure:
                labels = list(portfolio_risk.sector_exposure.keys())
                values = list(portfolio_risk.sector_exposure.values())
                fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.4,
                                       textinfo="label+percent", textposition="outside"))
                fig.update_layout(title="Sector Exposure", height=350, margin=dict(t=40, b=20, l=20, r=20))
                st.plotly_chart(fig, width="stretch")

        with warn_col:
            if portfolio_risk.warnings:
                for w in portfolio_risk.warnings:
                    st.warning(w)
            else:
                st.success("No risk warnings")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION C: Holdings Signals + Gap Analysis
    # ═══════════════════════════════════════════════════════════════════════
    if holding_signals:
        st.subheader("Signals for Your Holdings")

        for pos in sorted(positions, key=lambda p: p.pnl_percent, reverse=True):
            sig = holding_signals.get(pos.symbol)
            gap = holding_gaps.get(pos.symbol)
            if sig is None:
                continue

            sig_label = sig.signal_type.value.upper()
            sig_colors = {
                "STRONG_BUY": "#00c853", "BUY": "#69f0ae", "HOLD": "#ffd54f",
                "NO_EDGE": "#b0b0b0", "SELL": "#ff8a80", "STRONG_SELL": "#ff1744",
            }
            sig_color = sig_colors.get(sig.signal_type.value, "#b0b0b0")
            action, action_reason = _holding_action(sig, pos)
            action_colors = {"EXIT": "#ff1744", "ADD": "#00c853", "HOLD": "#ffd54f", "REVIEW": "#ff9100"}

            header = f"{pos.symbol} — {sig_label} ({sig.confidence:.0%}) — P&L: {pos.pnl_percent:+.1f}%"

            with st.expander(header, expanded=False):
                # Signal + levels
                s1, s2, s3, s4 = st.columns(4)
                s1.markdown(
                    f"<div style='background:{sig_color}; color:white; padding:8px 16px; "
                    f"border-radius:6px; text-align:center; font-weight:bold'>{sig_label}</div>",
                    unsafe_allow_html=True,
                )
                s2.metric("Entry", f"₹{sig.entry_price:,.2f}")
                s3.metric("Target", f"₹{sig.target_price:,.2f}" if sig.target_price else "N/A")
                s4.metric("Stop Loss", f"₹{sig.stop_loss:,.2f}" if sig.stop_loss else "N/A")

                # Gap analysis
                if gap and gap.has_gap:
                    direction = "Up" if gap.gap_percent > 0 else "Down"
                    gap_bg = "#fff3e0" if gap.gap_type == "medium" else ("#ffebee" if gap.gap_type == "large" else "#e3f2fd")
                    gap_border = "#ff9800" if gap.gap_type == "medium" else ("#f44336" if gap.gap_type == "large" else "#2196f3")
                    st.markdown(
                        f'<div style="background:{gap_bg}; padding:10px; border-radius:6px; '
                        f'border-left:4px solid {gap_border}; margin:8px 0">'
                        f'Gap {direction} <b>{abs(gap.gap_percent):.1f}%</b> ({gap.gap_type}) '
                        f'| Fill probability: <b>{gap.fill_probability:.0%}</b> '
                        f'| Entry suggestion: <b>{gap.entry_adjustment.replace("_", " ")}</b>'
                        f'</div>', unsafe_allow_html=True,
                    )

                # Action
                ac = action_colors.get(action, "#ffd54f")
                st.markdown(
                    f'<div style="background:{ac}20; padding:8px 12px; border-radius:6px; '
                    f'border-left:4px solid {ac}; margin:8px 0">'
                    f'<b>Recommended: {action}</b> — {action_reason}</div>',
                    unsafe_allow_html=True,
                )

                # Reasons
                if sig.reasons:
                    st.markdown("**Analysis:**")
                    for r in sig.reasons[:5]:
                        st.markdown(f"- {r}")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION D: New Investment Picks
    # ═══════════════════════════════════════════════════════════════════════
    st.subheader("New Investment Opportunities")

    if not long_picks and not short_picks:
        if extra_cash_result > 0:
            st.info("No stocks pass the strict multi-filter criteria right now. The market may be in a weak/uncertain phase. Consider keeping cash.")
        else:
            st.info("Set extra cash above 0 to scan for new investment opportunities.")
    else:
        tab_long, tab_short = st.tabs([f"Long Picks ({len(long_picks)})", f"Short Picks ({len(short_picks)})"])

        with tab_long:
            if not long_picks:
                st.info("No long picks pass all filters. Market may be weak/bearish.")
            for i, pick in enumerate(long_picks[:10]):
                gap = pick_gaps.get(pick.symbol)
                conf_colors = {
                    ConfidenceLevel.VERY_HIGH: "#00c853", ConfidenceLevel.HIGH: "#69f0ae",
                    ConfidenceLevel.MEDIUM: "#ffd54f", ConfidenceLevel.LOW: "#ff8a80",
                }
                conf_c = conf_colors.get(pick.confidence_level, "#b0b0b0")

                with st.expander(f"#{i+1} {pick.symbol} — {pick.recommendation} ({pick.confidence_score:.0f}%)"):
                    p1, p2, p3, p4 = st.columns(4)
                    p1.metric("Price", f"₹{pick.current_price:,.2f}")
                    p2.metric("Target 1", f"₹{pick.target_1:,.2f}")
                    p3.metric("Stop Loss", f"₹{pick.stop_loss:,.2f}")
                    p4.metric("Risk:Reward", f"1:{pick.risk_reward_ratio:.1f}")

                    # Gap
                    if gap and gap.has_gap:
                        direction = "Up" if gap.gap_percent > 0 else "Down"
                        gap_bg = "#fff3e0" if gap.gap_type == "medium" else ("#ffebee" if gap.gap_type == "large" else "#e3f2fd")
                        gap_border = "#ff9800" if gap.gap_type == "medium" else ("#f44336" if gap.gap_type == "large" else "#2196f3")
                        st.markdown(
                            f'<div style="background:{gap_bg}; padding:10px; border-radius:6px; '
                            f'border-left:4px solid {gap_border}; margin:8px 0">'
                            f'Gap {direction} <b>{abs(gap.gap_percent):.1f}%</b> ({gap.gap_type}) '
                            f'| Fill prob: <b>{gap.fill_probability:.0%}</b> '
                            f'| Suggestion: <b>{gap.entry_adjustment.replace("_", " ")}</b>'
                            f'</div>', unsafe_allow_html=True,
                        )
                        if gap.entry_adjustment == "skip" and pick.confidence_score > 70:
                            st.warning("Large gap detected on a high-confidence pick. Consider waiting for gap fill before entering.")

                    # Filters
                    fc1, fc2 = st.columns(2)
                    with fc1:
                        st.markdown(f"**Passed ({len(pick.filters_passed)}/6):**")
                        for f in pick.filters_passed:
                            st.markdown(f"- :green[{f.name}]: {f.reason}")
                    with fc2:
                        if pick.filters_failed:
                            st.markdown(f"**Failed ({len(pick.filters_failed)}/6):**")
                            for f in pick.filters_failed:
                                st.markdown(f"- :red[{f.name}]: {f.reason}")

                    st.caption(f"ADX: {pick.adx_value:.1f} | RSI: {pick.rsi_value:.1f} | Volume: {pick.volume_ratio:.1f}x | Regime: {pick.regime.value}")

        with tab_short:
            if not short_picks:
                st.info("No short opportunities found.")
            for i, pick in enumerate(short_picks[:5]):
                gap = pick_gaps.get(pick.symbol)
                with st.expander(f"#{i+1} {pick.symbol} — {pick.recommendation} ({pick.confidence_score:.0f}%)"):
                    p1, p2, p3, p4 = st.columns(4)
                    p1.metric("Price", f"₹{pick.current_price:,.2f}")
                    p2.metric("Target 1", f"₹{pick.target_1:,.2f}")
                    p3.metric("Stop Loss", f"₹{pick.stop_loss:,.2f}")
                    p4.metric("Risk:Reward", f"1:{pick.risk_reward_ratio:.1f}")

                    if gap and gap.has_gap:
                        direction = "Up" if gap.gap_percent > 0 else "Down"
                        st.info(f"Gap {direction} {abs(gap.gap_percent):.1f}% ({gap.gap_type}) | Fill prob: {gap.fill_probability:.0%}")

                    st.caption(f"ADX: {pick.adx_value:.1f} | RSI: {pick.rsi_value:.1f} | Volume: {pick.volume_ratio:.1f}x | Regime: {pick.regime.value}")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION E: Allocation Plan
    # ═══════════════════════════════════════════════════════════════════════
    if extra_cash_result > 0:
        st.subheader("Allocation Plan")

        if allocations:
            total_deployed = sum(a["cost"] for a in allocations)
            total_risk_inr = sum(a["risk"] for a in allocations)
            remaining_after = extra_cash_result - total_deployed

            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Deployable Cash", f"₹{extra_cash_result:,.0f}")
            a2.metric("Allocated", f"₹{total_deployed:,.0f}")
            a3.metric("Max Risk", f"₹{total_risk_inr:,.0f}", delta=f"{total_risk_inr/total_capital*100:.1f}% of capital")
            a4.metric("Remaining Cash", f"₹{remaining_after:,.0f}")

            # Allocation table
            import pandas as pd
            alloc_data = []
            for a in allocations:
                upside = a["shares"] * (a["target_1"] - a["price"])
                alloc_data.append({
                    "Symbol": a["symbol"],
                    "Rec.": a["recommendation"],
                    "Shares": a["shares"],
                    "Entry": f"₹{a['price']:,.2f}",
                    "Stop Loss": f"₹{a['stop_loss']:,.2f}",
                    "Target": f"₹{a['target_1']:,.2f}",
                    "Cost": f"₹{a['cost']:,.0f}",
                    "Max Loss": f"₹{a['risk']:,.0f}",
                    "Upside": f"₹{upside:,.0f}",
                    "R:R": f"1:{a['risk_reward']:.1f}",
                    "Confidence": f"{a['confidence']:.0f}%",
                })
            st.dataframe(pd.DataFrame(alloc_data), hide_index=True, width="stretch")

            for a in allocations:
                if a["warnings"]:
                    for w in a["warnings"]:
                        st.warning(f"{a['symbol']}: {w}")

            # Projected sector exposure
            if projected_risk.sector_exposure:
                st.markdown("**Projected Sector Exposure (after allocation):**")
                sec1, sec2 = st.columns(2)
                with sec1:
                    labels = list(projected_risk.sector_exposure.keys())
                    values = list(projected_risk.sector_exposure.values())
                    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.4,
                                           textinfo="label+percent", textposition="outside"))
                    fig.update_layout(height=300, margin=dict(t=20, b=20, l=20, r=20))
                    st.plotly_chart(fig, width="stretch")
                with sec2:
                    st.metric("Projected Heat", f"{projected_risk.portfolio_heat:.1f}%")
                    st.metric("Total Positions", f"{projected_risk.num_positions}")
                    if projected_risk.warnings:
                        for w in projected_risk.warnings:
                            st.warning(w)
                    else:
                        st.success("Portfolio within all risk limits")
        else:
            st.info("No allocations recommended. The market conditions or risk limits don't favor new positions right now. Consider keeping cash.")

        # Blocked picks
        if blocked:
            with st.expander("Blocked Picks (failed risk checks)"):
                for b in blocked:
                    st.error(f"**{b['symbol']}**: {'; '.join(b['blockers'])}")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION F: Action Summary
    # ═══════════════════════════════════════════════════════════════════════
    if positions:
        st.subheader("Action Summary")

        import pandas as pd
        actions_data = []
        for pos in positions:
            sig = holding_signals.get(pos.symbol)
            if sig:
                action, reason = _holding_action(sig, pos)
                actions_data.append({
                    "Symbol": pos.symbol,
                    "P&L %": f"{pos.pnl_percent:+.1f}%",
                    "Signal": sig.signal_type.value.upper(),
                    "Confidence": f"{sig.confidence:.0%}",
                    "Action": action,
                    "Reason": reason,
                })

        if actions_data:
            st.markdown("**Current Holdings:**")
            st.dataframe(pd.DataFrame(actions_data), hide_index=True, width="stretch")

        if allocations:
            new_data = []
            for a in allocations:
                new_data.append({
                    "Symbol": a["symbol"],
                    "Side": "LONG",
                    "Shares": a["shares"],
                    "Cost": f"₹{a['cost']:,.0f}",
                    "Max Risk": f"₹{a['risk']:,.0f}",
                    "Confidence": f"{a['confidence']:.0f}%",
                })
            st.markdown("**New Positions to Open:**")
            st.dataframe(pd.DataFrame(new_data), hide_index=True, width="stretch")

    # ── Disclaimer ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.warning(
        "**DISCLAIMER:** This portfolio plan is for educational purposes only. "
        "Indicator Agreement (shown as %) measures how many technical indicators "
        "point in the same direction -- it is NOT a prediction of profit probability. "
        "Position sizes are calculated using ATR-based risk management, but actual "
        "slippage, gaps, and execution costs may differ. Always do your own research. "
        "The authors are not SEBI-registered advisors."
    )


if __name__ == "__main__":
    main()
