"""Application constants and NSE data."""

# NSE Market Indices
NSE_INDICES = {
    "NIFTY50": "NIFTY 50",
    "BANKNIFTY": "NIFTY BANK",
    "NIFTYIT": "NIFTY IT",
    "NIFTYPHARMA": "NIFTY PHARMA",
    "NIFTYAUTO": "NIFTY AUTO",
    "NIFTYFMCG": "NIFTY FMCG",
    "NIFTYMETAL": "NIFTY METAL",
    "NIFTYREALTY": "NIFTY REALTY",
    "NIFTYENERGY": "NIFTY ENERGY",
    "NIFTYPSU": "NIFTY PSU BANK",
    "NIFTYMIDCAP": "NIFTY MIDCAP 50",
    "NIFTYSMALL": "NIFTY SMALLCAP 50",
}

# Yahoo Finance index symbols
YAHOO_INDEX_SYMBOLS = {
    "NIFTY50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "NIFTYIT": "^CNXIT",
    "NIFTYPHARMA": "^CNXPHARMA",
    "NIFTYMIDCAP": "^NSEI",  # Approximation — yfinance doesn't have all NSE indices
}

# Sectors
SECTORS = [
    "Banking",
    "IT",
    "Pharma",
    "Auto",
    "FMCG",
    "Metal",
    "Energy",
    "Realty",
    "Infrastructure",
    "Telecom",
    "Media",
    "Cement",
    "Chemical",
]

# Trading intervals
TRADING_INTERVALS = {
    "1m": "1 minute",
    "5m": "5 minutes",
    "15m": "15 minutes",
    "30m": "30 minutes",
    "1h": "1 hour",
    "1d": "1 day",
    "1wk": "1 week",
    "1mo": "1 month",
}

# Valid periods for historical data
VALID_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]

# Nifty 50 constituents (updated Feb 2026)
NIFTY50_SYMBOLS = [
    "ADANIPORTS",
    "APOLLOHOSP",
    "ASIANPAINT",
    "AXISBANK",
    "BAJAJ-AUTO",
    "BAJFINANCE",
    "BAJAJFINSV",
    "BPCL",
    "BHARTIARTL",
    "BRITANNIA",
    "CIPLA",
    "COALINDIA",
    "DIVISLAB",
    "DRREDDY",
    "EICHERMOT",
    "GRASIM",
    "HCLTECH",
    "HDFCBANK",
    "HDFCLIFE",
    "HEROMOTOCO",
    "HINDALCO",
    "HINDUNILVR",
    "ICICIBANK",
    "INDUSINDBK",
    "INFY",
    "ITC",
    "JSWSTEEL",
    "KOTAKBANK",
    "LT",
    "M&M",
    "MARUTI",
    "NESTLEIND",
    "NTPC",
    "ONGC",
    "POWERGRID",
    "RELIANCE",
    "SBILIFE",
    "SBIN",
    "SUNPHARMA",
    "TATACONSUM",
    "TMPV",
    "TATASTEEL",
    "TCS",
    "TECHM",
    "TITAN",
    "ULTRACEMCO",
    "UPL",
    "WIPRO",
    "SHRIRAMFIN",
    "LTIM",
]

# Bank Nifty constituents
BANKNIFTY_SYMBOLS = [
    "HDFCBANK",
    "ICICIBANK",
    "KOTAKBANK",
    "AXISBANK",
    "SBIN",
    "INDUSINDBK",
    "BANKBARODA",
    "AUBANK",
    "FEDERALBNK",
    "IDFCFIRSTB",
    "PNB",
    "BANDHANBNK",
]

# Nifty Next 50 - Large caps with more growth potential
NIFTY_NEXT50_SYMBOLS = [
    "ABB",
    "ADANIENSOL",
    "ADANIGREEN",
    "ADANIPOWER",
    "AMBUJACEM",
    "ATGL",
    "BAJAJHLDNG",
    "BHEL",
    "BOSCHLTD",
    "CANBK",
    "CHOLAFIN",
    "COLPAL",
    "DABUR",
    "DLF",
    "GAIL",
    "GODREJCP",
    "HAVELLS",
    "ICICIPRULI",
    "INDIGO",
    "IOC",
    "IRCTC",
    "JINDALSTEL",
    "JIOFIN",
    "LICI",
    "LODHA",
    "MARICO",
    "MOTHERSON",
    "NAUKRI",
    "NHPC",
    "PIDILITIND",
    "PFC",
    "RECLTD",
    "SAIL",
    "SIEMENS",
    "SRF",
    "TATAPOWER",
    "TRENT",
    "UNIONBANK",
    "VBL",
    "VEDL",
    "ETERNAL",
    "ZYDUSLIFE",
]

# Nifty Midcap 100 - Higher volatility, better for swing trading
NIFTY_MIDCAP_SYMBOLS = [
    "AARTIIND",
    "ABCAPITAL",
    "ACC",
    "ALKEM",
    "ASTRAL",
    "AUROPHARMA",
    "BALKRISIND",
    "BEL",
    "BHARATFORG",
    "BHEL",
    "BIOCON",
    "CANFINHOME",
    "CENTRALBK",
    "CGPOWER",
    "COFORGE",
    "CONCOR",
    "CROMPTON",
    "CUMMINSIND",
    "DEEPAKNTR",
    "DEVYANI",
    "DIXON",
    "ESCORTS",
    "EXIDEIND",
    "FEDERALBNK",
    "FORTIS",
    "GMRAIRPORT",
    "GODREJPROP",
    "GUJGASLTD",
    "HAL",
    "HINDPETRO",
    "IDFCFIRSTB",
    "IEX",
    "INDHOTEL",
    "INDUSTOWER",
    "IRFC",
    "JUBLFOOD",
    "KPITTECH",
    "LTF",
    "LAURUSLABS",
    "LICHSGFIN",
    "LUPIN",
    "M&MFIN",
    "MANAPPURAM",
    "MFSL",
    "MGL",
    "MPHASIS",
    "MUTHOOTFIN",
    "NAM-INDIA",
    "NATIONALUM",
    "NIACL",
    "NMDC",
    "OBEROIRLTY",
    "OIL",
    "PAGEIND",
    "PERSISTENT",
    "PETRONET",
    "POLYCAB",
    "PRESTIGE",
    "PVRINOX",
    "RAIN",
    "RAMCOCEM",
    "RBLBANK",
    "RELAXO",
    "SBICARD",
    "SHREECEM",
    "SONACOMS",
    "STARHEALTH",
    "SUNDARMFIN",
    "SUNTV",
    "SUPREMEIND",
    "TATACHEM",
    "TATACOMM",
    "TATAELXSI",
    "TORNTPHARM",
    "TVSMOTOR",
    "UBL",
    "VOLTAS",
    "WHIRLPOOL",
    "ZEEL",
]

# Nifty Smallcap - Highest volatility, aggressive trading
NIFTY_SMALLCAP_SYMBOLS = [
    "AEGISLOG",
    "ANGELONE",
    "APTUS",
    "ASTRAZEN",
    "BATAINDIA",
    "BAYERCROP",
    "BDL",
    "BLUEDART",
    "BSOFT",
    "CAMPUS",
    "CDSL",
    "ABREL",
    "CERA",
    "COCHINSHIP",
    "CYIENT",
    "DCMSHRIRAM",
    "ELGIEQUIP",
    "EMAMILTD",
    "FACT",
    "FINEORG",
    "FSL",
    "GLENMARK",
    "GNFC",
    "GRINDWELL",
    "GSFC",
    "HATSUN",
    "HEG",
    "HUDCO",
    "IIFL",
    "INDIAMART",
    "INTELLECT",
    "IPCALAB",
    "IRB",
    "ITI",
    "JKCEMENT",
    "JSWENERGY",
    "KALYANKJIL",
    "KANSAINER",
    "KARURVYSYA",
    "KEC",
    "KEI",
    "KIRLOSENG",
    "KNRCON",
    "LATENTVIEW",
    "LMW",
    "LINDEINDIA",
    "LLOYDSME",
    "MAZDOCK",
    "METROPOLIS",
    "MOTILALOFS",
    "MRPL",
    "NATCOPHARM",
    "NAVINFLUOR",
    "NBCC",
    "NCC",
    "NLCINDIA",
    "PNBHOUSING",
    "POLYPLEX",
    "PPLPHARMA",
    "PRINCEPIPE",
    "RADICO",
    "RAJESHEXPO",
    "RATNAMANI",
    "RAYMOND",
    "REDINGTON",
    "RITES",
    "ROUTE",
    "RVNL",
    "SANOFI",
    "SAPPHIRE",
    "SCHAEFFLER",
    "SJVN",
    "SOBHA",
    "SUMICHEM",
    "SUNDRMFAST",
    "SYNGENE",
    "TANLA",
    "TATAINVEST",
    "THERMAX",
    "TIMKEN",
    "TRITURBINE",
    "TRIDENT",
    "WELCORP",
    "WELSPUNLIV",
    "ZENSARTECH",
]

# F&O Stocks - High liquidity, good for both intraday and swing
FNO_STOCKS = [
    # Nifty 50 (liquid)
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "BHARTIARTL",
    "HINDUNILVR", "ITC", "KOTAKBANK", "LT", "AXISBANK", "BAJFINANCE", "MARUTI",
    "ASIANPAINT", "TITAN", "SUNPHARMA", "WIPRO", "HCLTECH", "TMPV",
    # High-volatility F&O
    "ADANIENT", "ADANIPORTS", "BAJAJ-AUTO", "BAJAJFINSV", "BPCL", "BRITANNIA",
    "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY", "EICHERMOT", "GRASIM",
    "HEROMOTOCO", "HINDALCO", "INDUSINDBK", "JSWSTEEL", "M&M", "NESTLEIND",
    "NTPC", "ONGC", "POWERGRID", "SBILIFE", "TATASTEEL", "TECHM", "ULTRACEMCO",
    # Popular mid-cap F&O (higher volatility)
    "AARTIIND", "ABCAPITAL", "ACC", "ALKEM", "AMBUJACEM", "APOLLOHOSP",
    "AUROPHARMA", "BALKRISIND", "BANDHANBNK", "BANKBARODA", "BATAINDIA",
    "BEL", "BERGEPAINT", "BHARATFORG", "BHEL", "BIOCON", "CANBK", "CHOLAFIN",
    "COFORGE", "COLPAL", "CONCOR", "CROMPTON", "CUB", "CUMMINSIND", "DABUR",
    "DALBHARAT", "DEEPAKNTR", "DELTACORP", "DIXON", "DLF", "ESCORTS", "EXIDEIND",
    "FEDERALBNK", "GAIL", "GLENMARK", "GMRAIRPORT", "GNFC", "GODREJCP", "GODREJPROP",
    "GRANULES", "GUJGASLTD", "HAL", "HAVELLS", "HINDPETRO", "SAMMAANCAP",
    "ICICIPRULI", "IDEA", "IDFCFIRSTB", "IEX", "INDHOTEL", "INDIGO", "INDUSTOWER",
    "INTELLECT", "IOC", "IPCALAB", "IRCTC", "ITC", "JINDALSTEL", "JUBLFOOD",
    "LTF", "LALPATHLAB", "LAURUSLABS", "LICHSGFIN", "LUPIN", "M&MFIN", "MANAPPURAM",
    "MARICO", "UNITDSPR", "MCX", "METROPOLIS", "MFSL", "MGL", "MOTHERSON",
    "MPHASIS", "MRF", "MUTHOOTFIN", "NAM-INDIA", "NATIONALUM", "NAUKRI", "NAVINFLUOR",
    "NMDC", "OBEROIRLTY", "OFSS", "PAGEIND", "PERSISTENT", "PETRONET", "PFC",
    "PIDILITIND", "PIIND", "PNB", "POLYCAB", "PVRINOX", "RAMCOCEM", "RBLBANK",
    "RECLTD", "SAIL", "SBICARD", "SHREECEM", "SIEMENS", "SRF", "SHRIRAMFIN",
    "STARHEALTH", "SUNTV", "TATACHEM", "TATACOMM", "TATAELXSI", "TATAPOWER",
    "TORNTPHARM", "TRENT", "TVSMOTOR", "UBL", "UNIONBANK", "UPL", "VEDL",
    "VOLTAS", "WHIRLPOOL", "ZEEL", "ZYDUSLIFE",
]

# High Volatility Stocks - Best for short-term trading
HIGH_VOLATILITY_STOCKS = [
    # Adani group (news-driven volatility)
    "ADANIENT", "ADANIPORTS", "ADANIGREEN", "ADANIPOWER", "ATGL", "AWL",
    # PSU stocks (policy-driven)
    "BHEL", "COALINDIA", "GAIL", "HINDPETRO", "IOC", "NHPC", "NMDC", "NTPC",
    "ONGC", "PFC", "POWERGRID", "RECLTD", "SAIL", "SJVN",
    # Metal (commodity-driven)
    "HINDALCO", "JINDALSTEL", "JSWSTEEL", "NATIONALUM", "NMDC", "SAIL",
    "TATASTEEL", "VEDL",
    # Infra & Realty (high beta)
    "DLF", "GMRAIRPORT", "GODREJPROP", "IRB", "LTF", "NBCC", "NCC",
    "OBEROIRLTY", "PRESTIGE", "SOBHA",
    # Small finance & NBFC (volatile)
    "BANDHANBNK", "IDFCFIRSTB", "MANAPPURAM", "MUTHOOTFIN", "PNB", "RBLBANK",
    # Pharma mid-caps
    "AUROPHARMA", "BIOCON", "GLENMARK", "GRANULES", "IPCALAB", "LAURUSLABS",
    "LUPIN", "NATCOPHARM",
    # IT mid-caps
    "COFORGE", "CYIENT", "KPITTECH", "LTTS", "MPHASIS", "PERSISTENT", "TATAELXSI",
    # Others with high beta
    "DELTACORP", "IDEA", "INDHOTEL", "IEX", "IRCTC", "PVRINOX", "TATAPOWER",
    "TRENT", "ZEEL", "ETERNAL",
]

# Sector-wise stocks for targeted screening
SECTOR_STOCKS = {
    "banking": [
        "HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN", "INDUSINDBK",
        "BANKBARODA", "PNB", "CANBK", "UNIONBANK", "FEDERALBNK", "IDFCFIRSTB",
        "BANDHANBNK", "AUBANK", "RBLBANK", "CUB", "KARURVYSYA",
    ],
    "it": [
        "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM", "COFORGE",
        "MPHASIS", "PERSISTENT", "TATAELXSI", "KPITTECH", "CYIENT", "LTTS",
        "ZENSARTECH",
    ],
    "pharma": [
        "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP", "LUPIN",
        "AUROPHARMA", "BIOCON", "TORNTPHARM", "ALKEM", "GLENMARK", "IPCALAB",
        "LAURUSLABS", "NATCOPHARM", "GRANULES", "ZYDUSLIFE",
    ],
    "auto": [
        "MARUTI", "TMPV", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT",
        "TVSMOTOR", "ASHOKLEY", "BHARATFORG", "BALKRISIND", "MOTHERSON",
        "EXIDEIND", "APOLLOTYRE", "MRF", "BOSCHLTD",
    ],
    "fmcg": [
        "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "MARICO",
        "GODREJCP", "COLPAL", "TATACONSUM", "VBL", "EMAMILTD", "RADICO",
        "UNITDSPR", "UBL", "PGHH",
    ],
    "metal": [
        "TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "JINDALSTEL", "SAIL",
        "NMDC", "NATIONALUM", "COALINDIA", "MOIL", "WELCORP", "RATNAMANI",
    ],
    "energy": [
        "RELIANCE", "ONGC", "BPCL", "IOC", "GAIL", "HINDPETRO", "PETRONET",
        "NTPC", "POWERGRID", "TATAPOWER", "ADANIGREEN", "NHPC", "SJVN",
        "PFC", "RECLTD", "TORNTPOWER",
    ],
    "realty": [
        "DLF", "GODREJPROP", "OBEROIRLTY", "PRESTIGE", "LODHA", "SOBHA",
        "PHOENIXLTD", "BRIGADE", "MAHLIFE",
    ],
    "infra": [
        "LT", "ADANIPORTS", "GMRAIRPORT", "IRB", "NCC", "NBCC", "KEC",
        "KNRCON", "HCC", "RVNL", "IRCON", "ENGINERSIN",
    ],
    "defence": [
        "HAL", "BEL", "BHEL", "BDL", "COCHINSHIP", "MAZDOCK", "GRSE",
    ],
    "chemicals": [
        "PIDILITIND", "SRF", "AARTIIND", "DEEPAKNTR", "NAVINFLUOR", "PIIND",
        "FINEORG", "CLEAN", "ATUL", "SUMICHEM", "GNFC", "GSFC",
    ],
}

# All tradeable stocks combined (deduplicated, including FNO and sector stocks)
_sector_symbols = [sym for syms in SECTOR_STOCKS.values() for sym in syms]
ALL_STOCKS = sorted(set(
    NIFTY50_SYMBOLS +
    NIFTY_NEXT50_SYMBOLS +
    NIFTY_MIDCAP_SYMBOLS +
    NIFTY_SMALLCAP_SYMBOLS +
    FNO_STOCKS +
    HIGH_VOLATILITY_STOCKS +
    _sector_symbols
))

# Technical indicator default parameters
INDICATOR_DEFAULTS = {
    "sma_periods": [9, 21, 50, 200],
    "ema_periods": [9, 21, 50, 200],
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std_dev": 2.0,
}

# Signal thresholds (default, used for trending markets)
SIGNAL_THRESHOLDS = {
    "strong_buy": 0.7,
    "buy": 0.3,
    "hold_upper": 0.3,
    "hold_lower": -0.3,
    "sell": -0.3,
    "strong_sell": -0.7,
}

# Regime-adaptive thresholds: tighter in sideways/volatile to reduce false signals
REGIME_SIGNAL_THRESHOLDS = {
    "trending": {
        "strong_buy": 0.7, "buy": 0.3,
        "hold_upper": 0.3, "hold_lower": -0.3,
        "sell": -0.3, "strong_sell": -0.7,
    },
    "sideways": {
        "strong_buy": 0.9, "buy": 0.6,
        "hold_upper": 0.6, "hold_lower": -0.6,
        "sell": -0.6, "strong_sell": -0.9,
    },
    "volatile": {
        "strong_buy": 0.85, "buy": 0.55,
        "hold_upper": 0.55, "hold_lower": -0.55,
        "sell": -0.55, "strong_sell": -0.85,
    },
}

# Minimum bars of data required before generating signals (indicator warm-up)
# Must be >= longest indicator warmup (200-MA needs 210 bars)
# Using 50 as MINIMUM but signal generator should also check per-indicator warmup
MIN_DATA_WARMUP = 50  # Signal-level minimum; full warmup handled per-indicator

# Securities Transaction Tax: 0.1% charged on SELL side only (buy side = 0%) for delivery
STT_RATE_SELL = 0.001
STT_RATE_BUY = 0.0  # No STT on buy side for delivery trades

# Minimum average daily volume to generate signals (filters illiquid stocks)
MIN_LIQUIDITY_VOLUME = 50_000

# Notional (INR) minimum daily turnover to consider a stock liquid
MIN_LIQUIDITY_TURNOVER_INR = 5_000_000  # Rs 50 lakh minimum daily turnover

# Transaction cost buffer added to targets so displayed R:R accounts for real costs
# Includes STT (0.1% sell), brokerage (0.03% x2), exchange + GST (~0.01%)
TRANSACTION_COST_BUFFER_PCT = 0.35  # ~0.35% round-trip cost

# Per-indicator minimum warmup bars (indicators need different history lengths)
INDICATOR_WARMUP = {
    "moving_average_200": 210,
    "moving_average_50": 60,
    "macd": 50,
    "rsi": 30,
    "bollinger": 30,
    "adx": 30,
    "atr": 20,
    "support_resistance": 30,
    "volume": 25,
}

# Market timing (IST)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30

# Risk-free rate (approximate 10-year Indian govt bond yield)
RISK_FREE_RATE = 0.07

# Maximum participation rate: max 5% of a stock's daily volume to avoid market impact
MAX_PARTICIPATION_RATE = 0.05

# NSE circuit breaker limits (10%, 15%, 20% from previous close)
CIRCUIT_BREAKER_LIMITS = [0.10, 0.15, 0.20]

# News sources
NEWS_SOURCES = {
    "google": "Google News",
    "moneycontrol": "MoneyControl",
    "et": "Economic Times",
}
