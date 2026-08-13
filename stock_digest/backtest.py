"""
Backtest the Stock Digest scoring system on historical prices.
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from stock_digest.analyzer import StockAnalyzer
from stock_digest.data_fetcher import DataFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_backtest(
    tickers: List[str],
    start: str,
    end: str,
    rebalance_freq: str = "MS",  # month start
    top_n: int = 5,
    initial_cash: float = 10000.0,
    weights: Dict = None,
    use_fundamentals: bool = False,  # if True, re-analyzes fundamentals at each rebalance (slower, look-ahead warning)
) -> Dict:
    """
    Simple event-based backtest.
    - Rebalances at start of each period.
    - Buys equal-weight top-N scored tickers.
    - Sells previous holdings, then buys new ones at close price.
    - Returns portfolio NAV curve, returns, drawdown, and benchmark comparison.
    """
    fetcher = DataFetcher()
    analyzer = StockAnalyzer(weights=weights)

    # Fetch historical prices for all tickers + SPY benchmark
    all_tickers = list(set(tickers + ["SPY"]))
    price_data = {}
    for t in all_tickers:
        df = fetcher.get_prices(t, period="max")
        if df is not None and not df.empty:
            df = df[(df.index >= start) & (df.index <= end)]
            if not df.empty:
                price_data[t] = df

    if not price_data:
        return {"error": "No price data available for backtest."}

    dates = price_data[list(price_data.keys())[0]].resample(rebalance_freq).last().dropna().index
    dates = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]

    cash = initial_cash
    holdings = {}  # ticker -> shares
    nav_history = []
    trades = []

    for i, date in enumerate(dates):
        if use_fundamentals:
            # Recompute scores (note: this uses latest fundamentals, not point-in-time)
            try:
                analyses = analyzer.analyze_tickers(tickers)
            except Exception as e:
                logger.warning("Analysis at %s failed: %s", date, e)
                analyses = []
            ranked = [a["ticker"] for a in analyses if "error" not in a][:top_n]
        else:
            # Use price-based momentum ranking for historical signal to avoid fundamental look-ahead bias
            ranked = _rank_by_momentum(tickers, price_data, date)[:top_n]

        # Sell previous holdings at current close
        if holdings:
            proceeds = 0.0
            for t, shares in holdings.items():
                price = _get_price(price_data, t, date)
                if price:
                    proceeds += shares * price
            cash += proceeds
            trades.append({"date": date, "action": "SELL", "tickers": list(holdings.keys()), "cash": cash})
            holdings = {}

        # Buy new top N equally weighted
        if ranked and cash > 0:
            allocation = cash / len(ranked)
            new_holdings = {}
            for t in ranked:
                price = _get_price(price_data, t, date)
                if price and price > 0:
                    shares = allocation / price
                    new_holdings[t] = shares
                    cash -= shares * price
            holdings = new_holdings
            trades.append({"date": date, "action": "BUY", "tickers": ranked, "cash": cash})

        # Mark portfolio to market
        nav = cash
        for t, shares in holdings.items():
            price = _get_price(price_data, t, date)
            if price:
                nav += shares * price
        nav_history.append({"date": date, "nav": nav})

    nav_df = pd.DataFrame(nav_history).set_index("date")
    returns = nav_df["nav"].pct_change().dropna()

    # Benchmark: SPY buy-and-hold
    spy_df = price_data.get("SPY")
    if spy_df is not None:
        spy_close = spy_df["Close"].reindex(nav_df.index).ffill()
        spy_nav = (spy_close / spy_close.iloc[0]) * initial_cash
        spy_returns = spy_nav.pct_change().dropna()
    else:
        spy_nav = pd.Series(dtype=float)
        spy_returns = pd.Series(dtype=float)

    total_return = (nav_df["nav"].iloc[-1] / initial_cash) - 1
    spy_total_return = (spy_nav.iloc[-1] / initial_cash) - 1 if not spy_nav.empty else 0.0

    return {
        "initial_cash": initial_cash,
        "final_nav": round(nav_df["nav"].iloc[-1], 2),
        "total_return_pct": round(total_return * 100, 2),
        "benchmark_return_pct": round(spy_total_return * 100, 2),
        "num_trades": len(trades),
        "sharpe_ratio": round(_sharpe(returns), 2),
        "max_drawdown_pct": round(_max_drawdown(nav_df["nav"]) * 100, 2),
        "nav_curve": nav_df,
        "benchmark_curve": spy_nav,
        "trades": trades,
        "top_ranked": ranked,
    }


def _rank_by_momentum(tickers: List[str], price_data: Dict, date: pd.Timestamp) -> List[str]:
    """Rank tickers by 3-month + 6-month momentum at a given historical date."""
    scores = {}
    for t in tickers:
        df = price_data.get(t)
        if df is None or df.empty:
            continue
        try:
            past = df.loc[:date]
            if len(past) < 63:
                continue
            ret_3m = past["Close"].iloc[-1] / past["Close"].iloc[-63] - 1
            ret_6m = past["Close"].iloc[-1] / past["Close"].iloc[-126] - 1 if len(past) >= 126 else ret_3m
            scores[t] = ret_3m * 0.6 + ret_6m * 0.4
        except Exception:
            continue
    return sorted(scores, key=scores.get, reverse=True)


def _get_price(price_data: Dict, ticker: str, date: pd.Timestamp) -> Optional[float]:
    df = price_data.get(ticker)
    if df is None or df.empty:
        return None
    try:
        # Use as-of to find latest available price on or before date
        price = df.loc[:date, "Close"].iloc[-1]
        return float(price)
    except Exception:
        return None


def _sharpe(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    if returns.empty or returns.std() == 0:
        return 0.0
    excess = returns - risk_free / periods_per_year
    return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))


def _max_drawdown(nav: pd.Series) -> float:
    peak = nav.cummax()
    drawdown = (nav - peak) / peak
    return float(drawdown.min())
