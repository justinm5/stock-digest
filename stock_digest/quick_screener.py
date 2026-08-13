"""
Fast sentiment + momentum screener.
No fundamentals needed — great for a quick daily buy/watch/avoid check.
"""
from concurrent.futures import ThreadPoolExecutor
from typing import List

from stock_digest.data_fetcher import DataFetcher
from stock_digest.sentiment_engine import analyze as analyze_sentiment


def _signal(sentiment_label: str, change_pct: float, confidence: float) -> str:
    if sentiment_label == "positive" and change_pct >= 0 and confidence >= 0.5:
        return "BUY NOW"
    if sentiment_label == "negative" and change_pct < 0 and confidence >= 0.5:
        return "AVOID"
    if sentiment_label == "positive" and change_pct >= -0.5:
        return "WATCH / BUY"
    if sentiment_label == "negative":
        return "AVOID / WATCH"
    return "HOLD"


def analyze_ticker(ticker: str, data: DataFetcher, news_limit: int = 8, sentiment_mode: str = "vader") -> dict:
    quote = data.get_quote(ticker)
    news = data.get_news(ticker, limit=news_limit)
    sent = analyze_sentiment(news, mode=sentiment_mode)
    change_pct = quote.get("change_pct") or 0.0
    sig = _signal(sent["label"], change_pct, sent["confidence"])

    momentum = change_pct / 10.0
    rank_score = round(sent["compound"] * 0.7 + momentum * 0.3, 3)
    top_headline = news[0]["title"] if news else "No recent headlines"

    return {
        "ticker": ticker.upper(),
        "name": quote.get("name", ticker.upper()),
        "price": quote.get("price"),
        "change_pct": change_pct,
        "sentiment": sent["label"],
        "sentiment_compound": sent["compound"],
        "confidence": sent["confidence"],
        "headlines_count": sent["headlines_scored"],
        "top_headline": top_headline,
        "signal": sig,
        "rank_score": rank_score,
        "sources": list(set([n["source"] for n in news])),
    }


def run_screener(tickers: List[str], news_limit: int = 8, max_workers: int = 4, sentiment_mode: str = "vader") -> List[dict]:
    data = DataFetcher()
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(analyze_ticker, t, data, news_limit, sentiment_mode): t for t in tickers}
        for future in future_to_ticker:
            try:
                results.append(future.result())
            except Exception as e:
                ticker = future_to_ticker[future]
                results.append({
                    "ticker": ticker.upper(),
                    "error": str(e),
                    "rank_score": -999,
                })
    results.sort(key=lambda x: x.get("rank_score", -999), reverse=True)
    return results
