"""
Extract and compute fundamental, valuation, and momentum metrics.
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def safe_float(value, default=0.0):
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def growth_rate(current, previous):
    if previous and previous != 0:
        return (current - previous) / abs(previous)
    return 0.0


def get_quarterly_values(df: pd.DataFrame, row_label: str, periods: int = 4) -> List[float]:
    """Get last N quarterly values for a given row label."""
    if df is None or row_label not in df.index:
        return []
    values = df.loc[row_label].dropna().tolist()
    values = [safe_float(v) for v in values]
    return values[:periods]


def ttm_sum(df: pd.DataFrame, row_label: str) -> float:
    """Sum last 4 quarters for a TTM figure."""
    return sum(get_quarterly_values(df, row_label, periods=4))


def extract_fundamentals(fundamentals: Dict) -> Dict:
    """Pull key fundamental metrics from yfinance statements."""
    inc = fundamentals.get("income")
    bal = fundamentals.get("balance")
    cf = fundamentals.get("cashflow")
    info = fundamentals.get("info", {})

    # Revenue and EPS
    revenues = get_quarterly_values(inc, "Total Revenue", 8)
    revenue_ttm = sum(revenues[:4]) if len(revenues) >= 4 else safe_float(info.get("totalRevenue"))
    revenue_ttm_py = sum(revenues[4:8]) if len(revenues) >= 8 else revenue_ttm
    revenue_growth = growth_rate(revenue_ttm, revenue_ttm_py)

    # Use diluted EPS if available, else net income / shares
    eps_values = get_quarterly_values(inc, "Diluted EPS", 8)
    eps_ttm = sum(eps_values[:4]) if len(eps_values) >= 4 else safe_float(info.get("trailingEps"))
    eps_ttm_py = sum(eps_values[4:8]) if len(eps_values) >= 8 else eps_ttm
    eps_growth = growth_rate(eps_ttm, eps_ttm_py)

    # Margins
    gross_profit = ttm_sum(inc, "Gross Profit") or safe_float(info.get("grossProfits"))
    operating_income = ttm_sum(inc, "Operating Income") or safe_float(info.get("operatingIncome"))
    net_income = ttm_sum(inc, "Net Income") or safe_float(info.get("netIncomeToCommon"))

    gross_margin = gross_profit / revenue_ttm if revenue_ttm else None
    operating_margin = operating_income / revenue_ttm if revenue_ttm else None
    net_margin = net_income / revenue_ttm if revenue_ttm else None

    # Equity / ROE
    equity = get_quarterly_values(bal, "Stockholders Equity", 1)
    equity_latest = equity[0] if equity else safe_float(info.get("totalStockholderEquity"))
    roe = net_income / equity_latest if equity_latest else None

    # ROIC = EBIT*(1-tax rate) / (debt + equity - cash)
    ebit = ttm_sum(inc, "EBIT") or operating_income
    tax_rate = 0.21
    nopat = ebit * (1 - tax_rate) if ebit else 0.0
    total_debt = ttm_sum(bal, "Total Debt") or safe_float(info.get("totalDebt"))
    cash = ttm_sum(bal, "Cash And Cash Equivalents") or safe_float(info.get("totalCash"))
    invested_capital = total_debt + equity_latest - cash
    roic = nopat / invested_capital if invested_capital else None

    # Free cash flow
    ocf = ttm_sum(cf, "Operating Cash Flow") or safe_float(info.get("operatingCashflow"))
    capex = ttm_sum(cf, "Capital Expenditure")
    if capex == 0:
        capex = abs(ttm_sum(cf, "Capital Expenditures")) if "Capital Expenditures" in (cf.index if cf is not None else []) else 0
    fcf = ocf - abs(capex) if ocf else safe_float(info.get("freeCashflow"))
    fcf_margin = fcf / revenue_ttm if revenue_ttm else None

    fcf_values = []
    if cf is not None and "Operating Cash Flow" in cf.index:
        ocf_vals = get_quarterly_values(cf, "Operating Cash Flow", 8)
        capex_vals = get_quarterly_values(cf, "Capital Expenditure", 8)
        fcf_values = [ocf_vals[i] - abs(capex_vals[i]) if i < len(capex_vals) else ocf_vals[i] for i in range(len(ocf_vals))]
    fcf_growth = growth_rate(sum(fcf_values[:4]), sum(fcf_values[4:8])) if len(fcf_values) >= 8 else 0.0

    # Debt and coverage
    debt_equity = total_debt / equity_latest if equity_latest else None
    interest_expense = abs(ttm_sum(inc, "Interest Expense")) or 0.0
    interest_coverage = operating_income / interest_expense if interest_expense else None

    # Share dilution
    shares = get_quarterly_values(bal, "Ordinary Shares Number", 8) or get_quarterly_values(bal, "Common Stock", 8)
    share_dilution = growth_rate(shares[0], shares[-1]) if len(shares) >= 2 else 0.0

    # Earnings consistency: coefficient of variation of quarterly EPS growth
    eps_consistency = 0.0
    if len(eps_values) >= 4:
        qs = eps_values[:4]
        if all(q != 0 for q in qs):
            eps_consistency = float(np.std(qs) / abs(np.mean(qs)))

    return {
        "revenue_growth": revenue_growth,
        "eps_growth": eps_growth,
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "net_margin": net_margin,
        "roe": roe,
        "roic": roic,
        "fcf": fcf,
        "fcf_growth": fcf_growth,
        "fcf_margin": fcf_margin,
        "debt_equity": debt_equity,
        "interest_coverage": interest_coverage,
        "share_dilution": share_dilution,
        "earnings_consistency": eps_consistency,
        "revenue_ttm": revenue_ttm,
        "net_income": net_income,
    }


def extract_valuation(info: Dict, fcf: float, price: Optional[float] = None, shares: Optional[float] = None) -> Dict:
    """Pull valuation ratios."""
    price = price or safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
    shares = shares or safe_float(info.get("sharesOutstanding"))
    market_cap = safe_float(info.get("marketCap"))

    pe = safe_float(info.get("trailingPE"))
    forward_pe = safe_float(info.get("forwardPE"))
    peg = safe_float(info.get("pegRatio"))
    ps = safe_float(info.get("priceToSalesTrailing12Months"))
    ev_ebitda = safe_float(info.get("enterpriseToEbitda"))
    ev_revenue = safe_float(info.get("enterpriseToRevenue"))

    # Price / FCF
    p_fcf = None
    if price and fcf and shares and fcf > 0:
        p_fcf = price / (fcf / shares)

    # FCF yield
    fcf_yield = None
    if market_cap and fcf and market_cap > 0:
        fcf_yield = fcf / market_cap

    return {
        "pe": pe,
        "forward_pe": forward_pe,
        "peg": peg,
        "ps": ps,
        "ev_ebitda": ev_ebitda,
        "ev_revenue": ev_revenue,
        "p_fcf": p_fcf,
        "fcf_yield": fcf_yield,
    }


def extract_momentum(prices: Optional[pd.DataFrame]) -> Dict:
    if prices is None or prices.empty:
        return {
            "return_1m": 0.0,
            "return_3m": 0.0,
            "return_6m": 0.0,
            "return_12m": 0.0,
            "rsi_14": 50.0,
            "sma_50_above_200": False,
            "vol_trend": 0.0,
        }
    close = prices["Close"]
    vol = prices["Volume"]

    def ret(days):
        if len(close) > days:
            return close.iloc[-1] / close.iloc[-max(days, 1)] - 1
        return 0.0

    rsi = compute_rsi(close, 14)
    sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else close.mean()
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else close.mean()

    vol_trend = 0.0
    if len(vol) >= 20:
        vol_trend = vol.iloc[-20:].mean() / vol.iloc[-60:].mean() - 1 if vol.iloc[-60:].mean() != 0 else 0.0

    return {
        "return_1m": ret(21),
        "return_3m": ret(63),
        "return_6m": ret(126),
        "return_12m": ret(252),
        "rsi_14": rsi,
        "sma_50_above_200": bool(sma50 > sma200) if pd.notna(sma50) and pd.notna(sma200) else False,
        "vol_trend": vol_trend,
    }


def compute_rsi(series: pd.Series, period: int = 14) -> float:
    if len(series) < period + 1:
        return 50.0
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean().iloc[-1]
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))
