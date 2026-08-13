"""GRU sentiment model loader and predictor."""
import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

LABELS = ["negative", "neutral", "positive"]


def load_tokenizer(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    from tensorflow.keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer(num_words=data.get("num_words"), oov_token=data.get("oov_token"))
    tokenizer.word_index = {k: int(v) for k, v in data["word_index"].items()}
    tokenizer.index_word = {int(k): v for k, v in data["index_word"].items()}
    return tokenizer


class GRUSentimentModel:
    def __init__(self, model_path: str = None, tokenizer_path: str = None, max_len: int = 40):
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_path = model_path or os.path.join(root, "models", "gru_sentiment_model.keras")
        self.tokenizer_path = tokenizer_path or os.path.join(root, "models", "tokenizer.json")
        self.max_len = max_len
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"GRU model not found at {self.model_path}. Train it first with stock_digest/gru/train.py")
        self.model = tf.keras.models.load_model(self.model_path)
        self.tokenizer = load_tokenizer(self.tokenizer_path)

    def predict(self, texts: list) -> list:
        if isinstance(texts, str):
            texts = [texts]
        sequences = self.tokenizer.texts_to_sequences(texts)
        x = pad_sequences(sequences, maxlen=self.max_len, padding="post", truncating="post")
        probs = self.model.predict(x, verbose=0)
        preds = np.argmax(probs, axis=1)
        results = []
        for p, prob in zip(preds, probs):
            results.append({
                "label": LABELS[int(p)],
                "confidence": float(np.max(prob)),
                "scores": {LABELS[i]: float(prob[i]) for i in range(3)},
            })
        return results

    def aggregate(self, texts: list) -> dict:
        if not texts:
            return {"sentiment": "neutral", "score": 0.0, "breakdown": {}}
        results = self.predict(texts)
        avg_scores = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
        for r in results:
            for k, v in r["scores"].items():
                avg_scores[k] += v
        for k in avg_scores:
            avg_scores[k] /= len(results)
        composite = avg_scores["positive"] - avg_scores["negative"]
        if composite > 0.2:
            sentiment = "positive"
        elif composite < -0.2:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        return {
            "sentiment": sentiment,
            "score": round(composite, 3),
            "confidence": round(float(np.mean([r["confidence"] for r in results])), 3),
            "headlines_analyzed": len(texts),
            "breakdown": {k: round(v, 3) for k, v in avg_scores.items()},
        }


def gru_model_available() -> bool:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.exists(os.path.join(root, "models", "gru_sentiment_model.keras")) and os.path.exists(
        os.path.join(root, "models", "tokenizer.json"))
