"""
Orchestrate data fetching, scoring, peer comparison, and bullet generation.
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from stock_digest.config import DEFAULT_WEIGHTS
from stock_digest.data_fetcher import DataFetcher
from stock_digest.metrics import (
    extract_fundamentals,
    extract_momentum,
    extract_valuation,
)
from stock_digest.scoring import (
    score_financial_health,
    score_fundamentals,
    score_momentum,
    score_sentiment,
    score_valuation,
)
from stock_digest import sentiment_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockAnalyzer:
    def __init__(self, weights: Dict = None):
        self.fetcher = DataFetcher()
        self.weights = weights or DEFAULT_WEIGHTS.copy()

    def analyze(self, ticker: str, include_peers: bool = True, sentiment_mode: str = "vader") -> Dict:
        ticker = ticker.upper()
        info = self.fetcher.get_info(ticker)
        fundamentals_raw = self.fetcher.get_fundamentals(ticker)
        fundamentals_raw["info"] = info
        prices = self.fetcher.get_prices(ticker, period="2y")
        finnhub_quote = self.fetcher.get_finnhub_quote(ticker)
        news = self.fetcher.get_news(ticker, limit=10)

        # Quote priority: Finnhub real-time, then Yahoo, then info
        price = None
        if finnhub_quote:
            price = finnhub_quote.get("price")
        if not price and prices is not None and not prices.empty:
            price = float(prices["Close"].iloc[-1])
        if not price:
            price = info.get("currentPrice") or info.get("regularMarketPrice")

        change_pct = 0.0
        if finnhub_quote:
            change_pct = finnhub_quote.get("change_pct", 0.0)
        elif prices is not None and len(prices) >= 2:
            change_pct = float(prices["Close"].iloc[-1] / prices["Close"].iloc[-2] - 1) * 100

        shares = info.get("sharesOutstanding")
        fcf_metrics = extract_fundamentals(fundamentals_raw)
        valuation = extract_valuation(info, fcf_metrics.get("fcf", 0), price=price, shares=shares)
        momentum = extract_momentum(prices)
        sentiment = sentiment_engine.analyze(news, mode=sentiment_mode)

        # Peer comparison
        sector = info.get("sector", "")
        industry = info.get("industry", "")
        peer_tickers = []
        peer_fundamentals = []
        peer_valuations = []
        if include_peers:
            peer_tickers = self.fetcher.get_peers(ticker, sector=sector)[:8]
            peer_tickers = [p for p in peer_tickers if p != ticker]
            for peer in peer_tickers:
                try:
                    p_info = self.fetcher.get_info(peer)
                    p_fund = self.fetcher.get_fundamentals(peer)
                    p_fund["info"] = p_info
                    p_metrics = extract_fundamentals(p_fund)
                    p_val = extract_valuation(p_info, p_metrics.get("fcf", 0), shares=p_info.get("sharesOutstanding"))
                    peer_fundamentals.append(p_metrics)
                    peer_valuations.append(p_val)
                except Exception as e:
                    logger.debug("Peer %s failed: %s", peer, e)

        fund_score = score_fundamentals(fcf_metrics, peer_fundamentals)
        val_score = score_valuation(valuation, peer_valuations)
        health_score = score_financial_health(fcf_metrics, peer_fundamentals)
        mom_score = score_momentum(momentum)
        sent_score = score_sentiment(sentiment)

        composite = round(
            fund_score["score"] * self.weights["fundamentals"]
            + val_score["score"] * self.weights["valuation"]
            + mom_score["score"] * self.weights["momentum"]
            + health_score["score"] * self.weights["financial_health"]
            + sent_score["score"] * self.weights["sentiment"],
            1,
        )

        color = "green" if composite >= 70 else "yellow" if composite >= 40 else "red"

        analysis = {
            "ticker": ticker,
            "name": info.get("shortName") or info.get("longName") or ticker,
            "sector": sector,
            "industry": industry,
            "price": price,
            "change_pct": change_pct,
            "market_cap": info.get("marketCap"),
            "score": composite,
            "color": color,
            "category_scores": {
                "fundamentals": fund_score,
                "valuation": val_score,
                "momentum": mom_score,
                "financial_health": health_score,
                "sentiment": sent_score,
            },
            "metrics": {
                "fundamentals": fcf_metrics,
                "valuation": valuation,
                "momentum": momentum,
            },
            "sentiment": sentiment,
            "news": news,
            "peers": peer_tickers,
            "why_buy": [],
            "why_not": [],
            "watch": [],
        }
        self._add_bullets(analysis)
        return analysis

    def _add_bullets(self, analysis: Dict):
        """Generate green/yellow/red bullets."""
        cats = analysis["category_scores"]
        metrics = analysis["metrics"]

        # Green bullets: strong category scores and standout metrics
        if cats["fundamentals"]["score"] >= 75:
            analysis["why_buy"].append("Strong fundamentals vs peers")
        if cats["valuation"]["score"] >= 75:
            analysis["why_buy"].append("Attractive valuation relative to industry")
        if cats["momentum"]["score"] >= 75:
            analysis["why_buy"].append("Positive price momentum")
        if cats["financial_health"]["score"] >= 75:
            analysis["why_buy"].append("Solid balance sheet and low financial risk")
        if cats["sentiment"]["score"] >= 75:
            analysis["why_buy"].append("Recent news sentiment is positive")

        f = metrics["fundamentals"]
        if f.get("revenue_growth", 0) > 0.20:
            analysis["why_buy"].append("Revenue growing faster than 20%")
        if f.get("eps_growth", 0) > 0.20:
            analysis["why_buy"].append("EPS growing faster than 20%")
        if f.get("roic", 0) > 0.15:
            analysis["why_buy"].append("High ROIC (capital efficiency)")
        if f.get("fcf_margin", 0) > 0.15:
            analysis["why_buy"].append("Strong free cash flow margin")

        v = metrics["valuation"]
        if v.get("fcf_yield", 0) > 0.05:
            analysis["why_buy"].append("FCF yield above 5%")
        if 0 < v.get("peg", 999) < 1.2:
            analysis["why_buy"].append("PEG ratio below 1.2 (growth at a reasonable price)")

        # Red bullets: weak areas
        if cats["fundamentals"]["score"] < 40:
            analysis["why_not"].append("Weak fundamentals vs peers")
        if cats["valuation"]["score"] < 40:
            analysis["why_not"].append("Expensive valuation relative to industry")
        if cats["momentum"]["score"] < 40:
            analysis["why_not"].append("Negative price momentum")
        if cats["financial_health"]["score"] < 40:
            analysis["why_not"].append("Balance sheet risk (high debt or weak coverage)")
        if cats["sentiment"]["score"] < 40:
            analysis["why_not"].append("Recent news sentiment is negative")

        if f.get("revenue_growth", 0) < 0:
            analysis["why_not"].append("Revenue declining")
        if f.get("eps_growth", 0) < 0:
            analysis["why_not"].append("EPS declining")
        if f.get("debt_equity", 0) > 1.5:
            analysis["why_not"].append("High debt-to-equity")
        if f.get("share_dilution", 0) > 0.05:
            analysis["why_not"].append("Significant share dilution")

        # Yellow bullets: caution / mixed signals
        if 40 <= cats["fundamentals"]["score"] < 70:
            analysis["watch"].append("Fundamentals are mixed vs peers")
        if 40 <= cats["valuation"]["score"] < 70:
            analysis["watch"].append("Valuation is neutral")
        if f.get("earnings_consistency", 0) > 0.5:
            analysis["watch"].append("Earnings have been inconsistent")

    def analyze_tickers(self, tickers: List[str], max_workers: int = 4, sentiment_mode: str = "vader") -> List[Dict]:
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(self.analyze, t, True, sentiment_mode): t for t in tickers}
            for future in as_completed(future_to_ticker):
                try:
                    results.append(future.result())
                except Exception as e:
                    ticker = future_to_ticker[future]
                    logger.error("Analysis failed for %s: %s", ticker, e)
                    results.append({
                        "ticker": ticker,
                        "error": str(e),
                        "score": 0,
                        "color": "red",
                    })
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results
