"""
News scraping utilities for financial headlines.
Falls back to yfinance built-in news if scraping fails.
"""
import re
import time
import logging
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def clean_headline(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def scrape_finviz_news(ticker: str, limit: int = 10) -> list:
    """Scrape latest headlines from Finviz news table."""
    url = f"https://finviz.com/news.ashx?v=3&t={quote_plus(ticker.upper())}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", class_="news-table")
        if not table:
            return []
        headlines = []
        for row in table.find_all("tr")[:limit]:
            link = row.find("a")
            if link and link.text:
                headlines.append(clean_headline(link.text))
        return headlines
    except Exception as e:
        logger.warning("Finviz scrape failed for %s: %s", ticker, e)
        return []


def scrape_bing_news(ticker: str, limit: int = 10) -> list:
    """Scrape headlines from Bing News search."""
    query = quote_plus(f"{ticker.upper()} stock news")
    url = f"https://www.bing.com/news/search?q={query}&form=QBNH"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        headlines = []
        for card in soup.select(".news-card")[:limit]:
            title = card.select_one(".title, a[aria-label]")
            if title and title.text:
                headlines.append(clean_headline(title.text))
        if not headlines:
            for a in soup.select("a[href]"):
                text = a.get_text(strip=True)
                if text and len(text) > 20 and ticker.upper() in text.upper():
                    headlines.append(clean_headline(text))
        return list(dict.fromkeys(headlines))[:limit]
    except Exception as e:
        logger.warning("Bing scrape failed for %s: %s", ticker, e)
        return []


def yfinance_news(ticker: str, limit: int = 10) -> list:
    """Fetch news titles from yfinance."""
    try:
        t = yf.Ticker(ticker.upper())
        news = t.news or []
        return [clean_headline(item.get("title", "")) for item in news[:limit] if item.get("title")]
    except Exception as e:
        logger.warning("yfinance news failed for %s: %s", ticker, e)
        return []


def get_headlines(ticker: str, limit: int = 10, use_yfinance: bool = True) -> list:
    """Try multiple news sources and merge unique headlines."""
    seen = set()
    all_headlines = []

    for source_fn in [scrape_finviz_news, scrape_bing_news]:
        try:
            for h in source_fn(ticker, limit=limit):
                if h and h not in seen:
                    seen.add(h)
                    all_headlines.append(h)
            if len(all_headlines) >= limit:
                break
        except Exception as e:
            logger.warning("Source failed: %s", e)
        time.sleep(0.3)

    if use_yfinance and len(all_headlines) < limit:
        for h in yfinance_news(ticker, limit=limit):
            if h and h not in seen:
                seen.add(h)
                all_headlines.append(h)

    return all_headlines[:limit]
