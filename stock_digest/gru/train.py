"""
Train a 256-unit GRU sentiment classifier on financial headlines.
Produces models/gru_sentiment_model.keras and models/tokenizer.json.
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna()
    texts = df["headline"].astype(str).values
    labels = df["sentiment"].astype(str).values
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    y = np.array([label_map[l] for l in labels])
    return texts, y


def build_tokenizer(texts, num_words=20_000, oov_token="<OOV>"):
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(texts)
    return tokenizer


def build_model(vocab_size, embedding_dim=128, max_len=40):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len, mask_zero=True),
        Bidirectional(GRU(256, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(3, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_tokenizer(tokenizer, path):
    data = {
        "word_index": tokenizer.word_index,
        "index_word": tokenizer.index_word,
        "num_words": tokenizer.num_words,
        "oov_token": tokenizer.oov_token,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Train GRU sentiment model")
    parser.add_argument("--data", default="data/financial_headlines.csv", help="Path to CSV dataset")
    parser.add_argument("--epochs", type=int, default=15, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--max-len", type=int, default=40, help="Max sequence length")
    parser.add_argument("--vocab", type=int, default=20_000, help="Tokenizer vocabulary size")
    args = parser.parse_args()

    print("Loading data...")
    texts, y = load_data(args.data)
    print(f"Samples: {len(texts)}")

    print("Building tokenizer...")
    tokenizer = build_tokenizer(texts, num_words=args.vocab)
    sequences = tokenizer.texts_to_sequences(texts)
    x = pad_sequences(sequences, maxlen=args.max_len, padding="post", truncating="post")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10, random_state=42, stratify=y_train)

    print("Building model...")
    model = build_model(min(args.vocab, len(tokenizer.word_index) + 1), max_len=args.max_len)
    model.summary()

    os.makedirs("models", exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
        ModelCheckpoint("models/gru_sentiment_model.keras", monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    print("Training...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    print("\nEvaluating on test set...")
    y_pred = np.argmax(model.predict(x_test, batch_size=args.batch_size), axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["negative", "neutral", "positive"]))

    model.save("models/gru_sentiment_model.keras")
    save_tokenizer(tokenizer, "models/tokenizer.json")
    print("Saved models/gru_sentiment_model.keras and models/tokenizer.json")


if __name__ == "__main__":
    main()
