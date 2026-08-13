"""
Fetch stock data from Yahoo Finance and Finnhub.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import finnhub
import pandas as pd
import requests
import yfinance as yf

from stock_digest.config import FINNHUB_API_KEY, NEWSAPI_API_KEY, SECTOR_PEERS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFetcher:
    def __init__(self):
        self.finnhub = finnhub.Client(api_key=FINNHUB_API_KEY) if FINNHUB_API_KEY else None

    def get_ticker(self, ticker: str) -> yf.Ticker:
        return yf.Ticker(ticker.upper())

    def get_info(self, ticker: str) -> Dict:
        try:
            t = self.get_ticker(ticker)
            return t.info or {}
        except Exception as e:
            logger.warning("Failed to fetch info for %s: %s", ticker, e)
            return {}

    def get_prices(self, ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
        import pandas as pd
        try:
            return self.get_ticker(ticker).history(period=period)
        except Exception as e:
            logger.warning("Failed to fetch prices for %s: %s", ticker, e)
            return None

    def get_fundamentals(self, ticker: str) -> Dict:
        """Fetch key financial statements."""
        try:
            t = self.get_ticker(ticker)
            return {
                "income": t.quarterly_income_stmt,
                "balance": t.quarterly_balance_sheet,
                "cashflow": t.quarterly_cash_flow,
                "info": t.info or {},
            }
        except Exception as e:
            logger.warning("Failed to fetch fundamentals for %s: %s", ticker, e)
            return {"income": None, "balance": None, "cashflow": None, "info": {}}

    def get_finnhub_quote(self, ticker: str) -> Optional[Dict]:
        if not self.finnhub:
            return None
        try:
            q = self.finnhub.quote(ticker.upper())
            return {
                "price": float(q.get("c", 0)),
                "change": float(q.get("d", 0)),
                "change_pct": float(q.get("dp", 0)),
                "high": float(q.get("h", 0)),
                "low": float(q.get("l", 0)),
                "open": float(q.get("o", 0)),
                "prev_close": float(q.get("pc", 0)),
            }
        except Exception as e:
            logger.warning("Finnhub quote failed for %s: %s", ticker, e)
            return None

    def get_peers(self, ticker: str, sector: str = "") -> List[str]:
        """Get peer tickers for industry comparison."""
        if self.finnhub:
            try:
                peers = self.finnhub.company_peers(ticker.upper())
                if peers:
                    return [p for p in peers if p and p != ticker.upper()][:10]
            except Exception as e:
                logger.warning("Finnhub peers failed for %s: %s", ticker, e)
        # Fallback to sector peer map
        return SECTOR_PEERS.get(sector, SECTOR_PEERS.get("Technology", []))

    def get_news(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Fetch news from Finnhub and NewsAPI."""
        headlines = []
        seen = set()

        if self.finnhub:
            try:
                end = datetime.now()
                start = end - timedelta(days=7)
                items = self.finnhub.company_news(
                    ticker.upper(),
                    _from=start.strftime("%Y-%m-%d"),
                    to=end.strftime("%Y-%m-%d"),
                )
                for item in (items or [])[:limit]:
                    title = item.get("headline", "")
                    if title and title.lower() not in seen:
                        seen.add(title.lower())
                        headlines.append({
                            "title": title,
                            "source": item.get("source", "Finnhub"),
                            "date": datetime.fromtimestamp(item.get("datetime", 0)).isoformat(),
                            "url": item.get("url", ""),
                        })
            except Exception as e:
                logger.warning("Finnhub news failed for %s: %s", ticker, e)

        if NEWSAPI_API_KEY:
            try:
                url = "https://newsapi.org/v2/everything"
                params = {
                    "q": f"{ticker.upper()} stock",
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": limit,
                    "apiKey": NEWSAPI_API_KEY,
                }
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                for item in resp.json().get("articles", [])[:limit]:
                    title = item.get("title", "")
                    if title and title.lower() not in seen:
                        seen.add(title.lower())
                        headlines.append({
                            "title": title,
                            "source": item.get("source", {}).get("name", "NewsAPI"),
                            "date": item.get("publishedAt", ""),
                            "url": item.get("url", ""),
                        })
            except Exception as e:
                logger.warning("NewsAPI failed for %s: %s", ticker, e)

        return headlines[:limit]
