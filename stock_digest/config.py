"""Configuration and API keys."""
import os
from dotenv import load_dotenv

load_dotenv()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY", "")

DEFAULT_TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM", "V", "UNH"]

# Scoring weights (must sum to 1.0)
DEFAULT_WEIGHTS = {
    "fundamentals": 0.25,
    "valuation": 0.20,
    "momentum": 0.20,
    "financial_health": 0.20,
    "sentiment": 0.15,
}

# Sector peer benchmarks for industry comparison
SECTOR_PEERS = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AVGO", "ADBE", "CRM"],
    "Financial Services": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "AXP"],
    "Healthcare": ["UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT"],
    "Consumer Cyclical": ["AMZN", "TSLA", "HD", "NKE", "MCD", "LOW", "BKNG", "SBUX"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "VZ", "CMCSA", "T", "TMUS"],
    "Industrials": ["GE", "HON", "UPS", "BA", "CAT", "RTX", "LMT", "DE"],
    "Consumer Defensive": ["WMT", "PG", "KO", "PEP", "COST", "MDLZ", "PM", "EL"],
    "Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "VLO", "PSX"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "PEG"],
    "Real Estate": ["AMT", "PLD", "CCI", "EQIX", "PSA", "O", "SPG", "DLR"],
    "Basic Materials": ["LIN", "APD", "SHW", "NEM", "ECL", "FCX", "DOW", "NUE"],
}
