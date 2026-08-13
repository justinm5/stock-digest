"""News sentiment using VADER."""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def score_text(text: str) -> dict:
    scores = _analyzer.polarity_scores(text)
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


def aggregate(headlines: list) -> dict:
    if not headlines:
        return {"label": "neutral", "compound": 0.0, "confidence": 0.0, "headlines_scored": 0}

    results = [score_text(h["title"]) for h in headlines]
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
