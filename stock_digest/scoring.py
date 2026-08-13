"""
Score metrics on a 0-100 scale and compare against peers / industry.
"""
import logging
from typing import Dict, List, Optional

import numpy as np

from stock_digest.metrics import safe_float

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def score_threshold(value: Optional[float], thresholds: List[tuple]) -> float:
    """
    thresholds: list of (cutoff, score) from high to low.
    Returns 0 if value is None.
    """
    if value is None or not np.isfinite(value):
        return 50.0
    for cutoff, score in thresholds:
        if value >= cutoff:
            return float(score)
    return 0.0


def score_inverse(value: Optional[float], thresholds: List[tuple]) -> float:
    """For metrics where lower is better. thresholds from low to high."""
    if value is None or not np.isfinite(value):
        return 50.0
    for cutoff, score in thresholds:
        if value <= cutoff:
            return float(score)
    return 0.0


def relative_score(value: Optional[float], peers: List[float], higher_is_better: bool = True) -> float:
    """Score based on percentile vs peers."""
    if value is None or not np.isfinite(value):
        return 50.0
    clean = [v for v in peers if v is not None and np.isfinite(v)]
    if not clean:
        return 50.0
    if higher_is_better:
        percentile = sum(1 for v in clean if value >= v) / len(clean)
    else:
        percentile = sum(1 for v in clean if value <= v) / len(clean)
    return round(percentile * 100, 1)


def score_fundamentals(metrics: Dict, peer_metrics: List[Dict] = None) -> Dict:
    peer_metrics = peer_metrics or []

    def peer_vals(key):
        return [p.get(key) for p in peer_metrics if p.get(key) is not None]

    scores = {
        "revenue_growth": relative_score(metrics.get("revenue_growth"), peer_vals("revenue_growth")),
        "eps_growth": relative_score(metrics.get("eps_growth"), peer_vals("eps_growth")),
        "gross_margin": relative_score(metrics.get("gross_margin"), peer_vals("gross_margin")),
        "operating_margin": relative_score(metrics.get("operating_margin"), peer_vals("operating_margin")),
        "net_margin": relative_score(metrics.get("net_margin"), peer_vals("net_margin")),
        "roe": relative_score(metrics.get("roe"), peer_vals("roe")),
        "roic": relative_score(metrics.get("roic"), peer_vals("roic")),
        "fcf_growth": relative_score(metrics.get("fcf_growth"), peer_vals("fcf_growth")),
        "fcf_margin": relative_score(metrics.get("fcf_margin"), peer_vals("fcf_margin")),
    }

    # If no peers, fall back to absolute thresholds
    if not peer_metrics:
        scores["revenue_growth"] = score_threshold(metrics.get("revenue_growth"), [(0.30, 100), (0.20, 80), (0.10, 60), (0.0, 40), (-1.0, 0)])
        scores["eps_growth"] = score_threshold(metrics.get("eps_growth"), [(0.30, 100), (0.20, 80), (0.10, 60), (0.0, 40), (-1.0, 0)])
        scores["gross_margin"] = score_threshold(metrics.get("gross_margin"), [(0.50, 100), (0.40, 80), (0.30, 60), (0.20, 40), (0.0, 0)])
        scores["operating_margin"] = score_threshold(metrics.get("operating_margin"), [(0.30, 100), (0.20, 80), (0.10, 60), (0.05, 40), (-1.0, 0)])
        scores["net_margin"] = score_threshold(metrics.get("net_margin"), [(0.25, 100), (0.15, 80), (0.10, 60), (0.05, 40), (-1.0, 0)])
        scores["roe"] = score_threshold(metrics.get("roe"), [(0.30, 100), (0.20, 80), (0.15, 60), (0.10, 40), (0.0, 0)])
        scores["roic"] = score_threshold(metrics.get("roic"), [(0.25, 100), (0.18, 80), (0.12, 60), (0.08, 40), (0.0, 0)])
        scores["fcf_growth"] = score_threshold(metrics.get("fcf_growth"), [(0.30, 100), (0.20, 80), (0.10, 60), (0.0, 40), (-1.0, 0)])
        scores["fcf_margin"] = score_threshold(metrics.get("fcf_margin"), [(0.25, 100), (0.18, 80), (0.12, 60), (0.06, 40), (0.0, 0)])

    weights = {
        "revenue_growth": 0.12,
        "eps_growth": 0.18,
        "gross_margin": 0.10,
        "operating_margin": 0.12,
        "net_margin": 0.10,
        "roe": 0.12,
        "roic": 0.12,
        "fcf_growth": 0.08,
        "fcf_margin": 0.06,
    }
    composite = sum(scores[k] * weights[k] for k in weights)
    return {"score": round(composite, 1), "breakdown": scores}


def score_valuation(valuation: Dict, peer_valuations: List[Dict] = None) -> Dict:
    peer_valuations = peer_valuations or []

    def peer_vals(key):
        return [p.get(key) for p in peer_valuations if p.get(key) is not None]

    # Lower is better for most valuation ratios; higher is better for fcf_yield
    scores = {}

    pe = safe_float(valuation.get("pe"))
    if pe <= 0:
        scores["pe"] = 0.0
    else:
        scores["pe"] = relative_score(pe, peer_vals("pe"), higher_is_better=False) if peer_vals("pe") else score_inverse(pe, [(10, 100), (15, 80), (25, 60), (40, 40), (60, 20)])

    forward_pe = safe_float(valuation.get("forward_pe"))
    if forward_pe <= 0:
        scores["forward_pe"] = 0.0
    else:
        scores["forward_pe"] = relative_score(forward_pe, peer_vals("forward_pe"), higher_is_better=False) if peer_vals("forward_pe") else score_inverse(forward_pe, [(12, 100), (18, 80), (25, 60), (35, 40), (50, 20)])

    peg = safe_float(valuation.get("peg"))
    scores["peg"] = relative_score(peg, peer_vals("peg"), higher_is_better=False) if peer_vals("peg") else score_inverse(peg, [(0.8, 100), (1.0, 80), (1.5, 60), (2.0, 40), (3.0, 20)])

    ev_ebitda = safe_float(valuation.get("ev_ebitda"))
    scores["ev_ebitda"] = relative_score(ev_ebitda, peer_vals("ev_ebitda"), higher_is_better=False) if peer_vals("ev_ebitda") else score_inverse(ev_ebitda, [(8, 100), (12, 80), (18, 60), (25, 40), (40, 20)])

    p_fcf = safe_float(valuation.get("p_fcf"))
    scores["p_fcf"] = relative_score(p_fcf, peer_vals("p_fcf"), higher_is_better=False) if peer_vals("p_fcf") else score_inverse(p_fcf, [(12, 100), (18, 80), (25, 60), (40, 40), (60, 20)])

    ps = safe_float(valuation.get("ps"))
    scores["ps"] = relative_score(ps, peer_vals("ps"), higher_is_better=False) if peer_vals("ps") else score_inverse(ps, [(2, 100), (4, 80), (7, 60), (10, 40), (15, 20)])

    fcf_yield = safe_float(valuation.get("fcf_yield"))
    scores["fcf_yield"] = relative_score(fcf_yield, peer_vals("fcf_yield")) if peer_vals("fcf_yield") else score_threshold(fcf_yield, [(0.06, 100), (0.04, 80), (0.03, 60), (0.02, 40), (0.0, 0)])

    weights = {"pe": 0.20, "forward_pe": 0.20, "peg": 0.15, "ev_ebitda": 0.15, "p_fcf": 0.15, "ps": 0.08, "fcf_yield": 0.07}
    composite = sum(scores[k] * weights[k] for k in weights)
    return {"score": round(composite, 1), "breakdown": scores}


def score_financial_health(metrics: Dict, peer_metrics: List[Dict] = None) -> Dict:
    peer_metrics = peer_metrics or []

    def peer_vals(key):
        return [p.get(key) for p in peer_metrics if p.get(key) is not None]

    de = safe_float(metrics.get("debt_equity"))
    if de < 0:
        scores_de = 100.0
    else:
        scores_de = relative_score(de, peer_vals("debt_equity"), higher_is_better=False) if peer_vals("debt_equity") else score_inverse(de, [(0.2, 100), (0.5, 80), (1.0, 60), (1.5, 40), (3.0, 20)])

    ic = safe_float(metrics.get("interest_coverage"))
    scores_ic = relative_score(ic, peer_vals("interest_coverage")) if peer_vals("interest_coverage") else score_threshold(ic, [(10, 100), (5, 80), (3, 60), (1.5, 40), (0.0, 0)])

    dil = safe_float(metrics.get("share_dilution"))
    scores_dil = 100.0 if dil < -0.02 else 60.0 if dil < 0.02 else 30.0 if dil < 0.05 else 0.0

    cons = safe_float(metrics.get("earnings_consistency"))
    scores_cons = score_inverse(cons, [(0.1, 100), (0.3, 80), (0.5, 60), (0.8, 40), (10.0, 0)])

    scores = {"debt_equity": scores_de, "interest_coverage": scores_ic, "share_dilution": scores_dil, "earnings_consistency": scores_cons}
    weights = {"debt_equity": 0.30, "interest_coverage": 0.30, "share_dilution": 0.25, "earnings_consistency": 0.15}
    composite = sum(scores[k] * weights[k] for k in weights)
    return {"score": round(composite, 1), "breakdown": scores}


def score_momentum(momentum: Dict) -> Dict:
    ret_1m = safe_float(momentum.get("return_1m"))
    ret_3m = safe_float(momentum.get("return_3m"))
    ret_6m = safe_float(momentum.get("return_6m"))
    ret_12m = safe_float(momentum.get("return_12m"))
    rsi = safe_float(momentum.get("rsi_14"))
    golden_cross = bool(momentum.get("sma_50_above_200"))
    vol_trend = safe_float(momentum.get("vol_trend"))

    # Momentum score: reward consistent 3m/6m/12m uptrends, penalize extreme RSI
    trend_score = min(100, max(0, (ret_3m + ret_6m + ret_12m) / 3 * 200 + 50))
    rsi_score = 100 - abs(rsi - 55) if rsi > 30 else max(0, rsi)  # prefer RSI 45-65
    golden_score = 80 if golden_cross else 40
    vol_score = 70 if vol_trend > 0 else 50

    scores = {
        "trend": round(trend_score, 1),
        "rsi": round(rsi_score, 1),
        "golden_cross": round(golden_score, 1),
        "volume_trend": round(vol_score, 1),
    }
    weights = {"trend": 0.45, "rsi": 0.25, "golden_cross": 0.20, "volume_trend": 0.10}
    composite = sum(scores[k] * weights[k] for k in weights)
    return {"score": round(composite, 1), "breakdown": scores}


def score_sentiment(sentiment: Dict) -> Dict:
    if not sentiment:
        return {"score": 50.0, "breakdown": {"positive": 0, "negative": 0, "neutral": 0}}
    compound = safe_float(sentiment.get("compound"))
    confidence = safe_float(sentiment.get("confidence"))
    score = 50 + (compound * 50)  # map -1..1 to 0..100
    score = score * (0.7 + 0.3 * confidence)  # discount by confidence
    return {"score": round(max(0, min(100, score)), 1), "breakdown": sentiment}
