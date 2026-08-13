"""News sentiment using VADER (fast) or trained GRU model (deep)."""
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from stock_digest.gru.model import GRUSentimentModel, gru_model_available

_vader = SentimentIntensityAnalyzer()


def _vader_score_text(text: str) -> dict:
    scores = _vader.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return {
        "label": label,
        "compound": round(compound, 3),
        "positive": round(scores["pos"], 3),
        "neutral": round(scores["neu"], 3),
        "negative": round(scores["neg"], 3),
    }


def _vader_aggregate(headlines: list) -> dict:
    if not headlines:
        return {"label": "neutral", "compound": 0.0, "confidence": 0.0, "headlines_scored": 0}

    titles = [h["title"] if isinstance(h, dict) else h for h in headlines]
    results = [_vader_score_text(t) for t in titles]
    avg_compound = sum(r["compound"] for r in results) / len(results)
    pos = sum(1 for r in results if r["label"] == "positive")
    neg = sum(1 for r in results if r["label"] == "negative")
    neu = len(results) - pos - neg

    if avg_compound >= 0.05:
        label = "positive"
    elif avg_compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    non_neutral = pos + neg
    agree = pos if label == "positive" else neg if label == "negative" else neu
    confidence = round(agree / len(results), 2) if results else 0.0

    return {
        "label": label,
        "compound": round(avg_compound, 3),
        "confidence": confidence,
        "positive_count": pos,
        "negative_count": neg,
        "neutral_count": neu,
        "headlines_scored": len(results),
    }


def analyze(headlines: list, mode: str = "vader") -> dict:
    """
    Analyze sentiment of a list of headline dicts or strings.
    mode: 'vader' (fast, no model needed) or 'gru' (trained model).
    """
    if mode == "gru" and gru_model_available():
        titles = [h["title"] if isinstance(h, dict) else h for h in headlines]
        return GRUSentimentModel().aggregate(titles)
    return _vader_aggregate(headlines)
