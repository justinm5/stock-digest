"""
Stock recommender: combine news sentiment with yfinance price data.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf
from stock_digest.gru.scraper import get_headlines
from stock_digest.gru.model import GRUSentimentModel


def fetch_stock_info(ticker: str):
    try:
        t = yf.Ticker(ticker.upper())
        hist = t.history(period="2d")
        info = t.info or {}
        if len(hist) >= 2:
            prev_close = float(hist["Close"].iloc[-2])
            last_close = float(hist["Close"].iloc[-1])
            change_pct = round((last_close - prev_close) / prev_close * 100, 2)
        elif len(hist) == 1:
            last_close = float(hist["Close"].iloc[-1])
            change_pct = 0.0
            prev_close = last_close
        else:
            last_close = info.get("regularMarketPrice") or info.get("previousClose") or 0.0
            prev_close = info.get("previousClose") or last_close
            change_pct = 0.0
        return {
            "ticker": ticker.upper(),
            "name": info.get("shortName") or info.get("longName") or ticker.upper(),
            "price": round(last_close, 2) if last_close else None,
            "change_pct": change_pct,
        }
    except Exception as e:
        return {
            "ticker": ticker.upper(),
            "name": ticker.upper(),
            "price": None,
            "change_pct": 0.0,
            "error": str(e),
        }


def analyze_ticker(ticker: str, model: GRUSentimentModel, headlines_limit: int = 10):
    headlines = get_headlines(ticker, limit=headlines_limit)
    sentiment = model.aggregate(headlines)
    stock = fetch_stock_info(ticker)
    return {
        **stock,
        "headlines": headlines,
        "sentiment": sentiment,
        "recommendation": _recommendation(sentiment["score"], stock["change_pct"]),
        "rank_score": _rank_score(sentiment["score"], stock["change_pct"]),
    }


def _recommendation(sentiment_score: float, change_pct: float) -> str:
    combined = sentiment_score * 0.7 + (change_pct / 10.0) * 0.3
    if combined > 0.35:
        return "Buy"
    if combined < -0.35:
        return "Sell"
    return "Hold"


def _rank_score(sentiment_score: float, change_pct: float) -> float:
    return round(sentiment_score * 0.7 + (change_pct / 10.0) * 0.3, 3)


def rank_tickers(tickers: list, model: GRUSentimentModel, max_workers: int = 4, headlines_limit: int = 10) -> list:
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(analyze_ticker, t, model, headlines_limit): t for t in tickers
        }
        for future in as_completed(future_to_ticker):
            try:
                results.append(future.result())
            except Exception as e:
                ticker = future_to_ticker[future]
                results.append({
                    "ticker": ticker.upper(),
                    "name": ticker.upper(),
                    "error": str(e),
                    "rank_score": -999,
                })
    results.sort(key=lambda x: x.get("rank_score", -999), reverse=True)
    return results
