"""Unified Stock Digest Streamlit app."""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stock_digest.analyzer import StockAnalyzer
from stock_digest.backtest import run_backtest
from stock_digest.config import DEFAULT_TICKERS
from stock_digest.gru.model import gru_model_available
from stock_digest.quick_screener import run_screener

st.set_page_config(page_title="Stock Digest", page_icon="", layout="wide")


def color_card(score: float):
    if score >= 70:
        return "#10b981"
    if score >= 40:
        return "#f59e0b"
    return "#ef4444"


def sidebar_weights():
    st.header("Scoring weights")
    weights = {}
    weights["fundamentals"] = st.slider("Fundamentals", 0.0, 1.0, 0.25, 0.05)
    weights["valuation"] = st.slider("Valuation", 0.0, 1.0, 0.20, 0.05)
    weights["momentum"] = st.slider("Momentum", 0.0, 1.0, 0.20, 0.05)
    weights["financial_health"] = st.slider("Financial Health", 0.0, 1.0, 0.20, 0.05)
    weights["sentiment"] = st.slider("Sentiment", 0.0, 1.0, 0.15, 0.05)
    total = sum(weights.values())
    if abs(total - 1.0) > 0.01:
        st.warning(f"Weights sum to {total:.2f}. Normalizing automatically.")
        weights = {k: v / total for k, v in weights.items()}
    return weights


def render_quick_screener():
    st.header("Quick Screener")
    st.markdown("Fast sentiment + momentum signals. No model training needed.")

    ticker_input = st.text_area("Tickers (comma-separated)", ", ".join(DEFAULT_TICKERS), height=100)
    news_limit = st.slider("Headlines per ticker", 3, 20, 8, key="qs_news")
    top_n = st.slider("Show top", 1, 20, 10, key="qs_top")
    sentiment_mode = st.selectbox("Sentiment engine", ["vader", "gru"], index=0, key="qs_sent")
    if sentiment_mode == "gru" and not gru_model_available():
        st.warning("GRU model not found. Train it in the Train Model tab or use VADER.")
        sentiment_mode = "vader"

    if st.button("Scan now", type="primary", key="qs_run"):
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        with st.spinner("Scanning quotes and news..."):
            results = run_screener(tickers, news_limit=news_limit, sentiment_mode=sentiment_mode)

        df = pd.DataFrame([
            {
                "Rank": i + 1,
                "Ticker": r["ticker"],
                "Price": r.get("price"),
                "Change %": r.get("change_pct"),
                "Sentiment": r.get("sentiment"),
                "Confidence": r.get("confidence"),
                "Headlines": r.get("headlines_count"),
                "Signal": r.get("signal"),
                "Top headline": r.get("top_headline", ""),
            }
            for i, r in enumerate(results[:top_n])
        ])

        def color_signal(val):
            if val == "BUY NOW":
                return "color: #10b981; font-weight: 700;"
            if val == "AVOID":
                return "color: #ef4444; font-weight: 700;"
            if val in ("WATCH / BUY", "WATCH"):
                return "color: #3b82f6;"
            return "color: #f59e0b;"

        st.dataframe(df.style.applymap(color_signal, subset=["Signal"]), use_container_width=True)

        st.markdown("**How to read:** `BUY NOW` = positive news + price up. `WATCH / BUY` = positive news but price flat. `HOLD` = mixed. `AVOID` = negative news or dropping.")


def render_deep_digest(weights):
    st.header("Deep Stock Digest")
    st.markdown("Full fundamental, valuation, momentum, health, and sentiment scoring with peer comparison.")

    ticker_input = st.text_area("Tickers (comma-separated)", ", ".join(DEFAULT_TICKERS), height=100, key="dd_tickers")
    sentiment_mode = st.selectbox("Sentiment engine", ["vader", "gru"], index=0, key="dd_sent")
    if sentiment_mode == "gru" and not gru_model_available():
        st.warning("GRU model not found. Train it in the Train Model tab or use VADER.")
        sentiment_mode = "vader"

    if st.button("Analyze", type="primary", key="dd_run"):
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        analyzer = StockAnalyzer(weights=weights)

        with st.spinner("Analyzing stocks..."):
            results = analyzer.analyze_tickers(tickers, sentiment_mode=sentiment_mode)

        df = pd.DataFrame([
            {
                "Rank": i + 1,
                "Ticker": r["ticker"],
                "Name": r.get("name", ""),
                "Price": r.get("price"),
                "Change %": r.get("change_pct"),
                "Score": r["score"],
                "Fund": r["category_scores"]["fundamentals"]["score"],
                "Val": r["category_scores"]["valuation"]["score"],
                "Mom": r["category_scores"]["momentum"]["score"],
                "Health": r["category_scores"]["financial_health"]["score"],
                "Sent": r["category_scores"]["sentiment"]["score"],
            }
            for i, r in enumerate(results)
        ])

        st.subheader("Ranked stocks")
        st.dataframe(df, use_container_width=True)

        for r in results:
            c = color_card(r["score"])
            with st.expander(f"{r['ticker']} — {r.get('name', '')}  |  Score: {r['score']}", expanded=False):
                col1, col2, col3 = st.columns([1, 1, 1])
                col1.metric("Price", f"${r.get('price')}" if r.get('price') else "N/A", f"{r.get('change_pct', 0):.2f}%")
                col2.metric("Overall Score", r["score"])
                col3.markdown(f"<span style='color:{c};font-size:24px;font-weight:bold;'>{r['color'].upper()}</span>", unsafe_allow_html=True)

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Fundamentals", r["category_scores"]["fundamentals"]["score"])
                c2.metric("Valuation", r["category_scores"]["valuation"]["score"])
                c3.metric("Momentum", r["category_scores"]["momentum"]["score"])
                c4.metric("Health", r["category_scores"]["financial_health"]["score"])
                c5.metric("Sentiment", r["category_scores"]["sentiment"]["score"])

                if r.get("why_buy"):
                    st.markdown("**Why buy**")
                    for item in r["why_buy"]:
                        st.markdown(f":green[+] {item}")
                if r.get("why_not"):
                    st.markdown("**Why not**")
                    for item in r["why_not"]:
                        st.markdown(f":red[-] {item}")
                if r.get("watch"):
                    st.markdown("**Watch**")
                    for item in r["watch"]:
                        st.markdown(f":orange[~] {item}")

                st.markdown("**Latest news**")
                for n in r.get("news", [])[:5]:
                    st.markdown(f"- {n['title']} ({n['source']})")


def render_train_model():
    st.header("Train GRU Sentiment Model")
    st.markdown("Train a 256-unit GRU on 30,599 synthetic financial headlines. This replaces VADER in the screeners.")

    col1, col2 = st.columns(2)
    if col1.button("1. Generate dataset", type="primary"):
        with st.spinner("Generating 30,599 headlines..."):
            result = subprocess.run(
                [sys.executable, "-m", "stock_digest.gru.build_dataset"],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
        st.code(result.stdout + result.stderr)

    if col2.button("2. Train model", type="primary"):
        with st.spinner("Training GRU model... this may take a few minutes"):
            result = subprocess.run(
                [sys.executable, "-m", "stock_digest.gru.train"],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
        st.code(result.stdout + result.stderr)
        if gru_model_available():
            st.success("GRU model ready. Switch sentiment engine to 'gru' in the screeners.")


def render_backtest(weights):
    st.header("Backtest")
    st.markdown("Backtest a monthly top-N momentum strategy against SPY.")

    ticker_input = st.text_area("Tickers (comma-separated)", ", ".join(DEFAULT_TICKERS), height=100, key="bt_tickers")
    col1, col2, col3 = st.columns(3)
    start = col1.date_input("Start", value=pd.to_datetime("2022-01-01"), key="bt_start")
    end = col2.date_input("End", value=pd.to_datetime("today"), key="bt_end")
    top_n = col3.slider("Top N holdings", 1, 10, 5, key="bt_top")

    if st.button("Run backtest", type="primary", key="bt_run"):
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        with st.spinner("Running backtest..."):
            bt = run_backtest(
                tickers,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                top_n=top_n,
                weights=weights,
            )

        if "error" in bt:
            st.error(bt["error"])
        else:
            st.metric("Strategy return", f"{bt['total_return_pct']}%", f"vs SPY {bt['benchmark_return_pct']}%")
            st.metric("Max drawdown", f"{bt['max_drawdown_pct']}%")
            st.metric("Sharpe ratio", bt["sharpe_ratio"])

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=bt["nav_curve"].index, y=bt["nav_curve"]["nav"], name="Strategy"))
            if not bt["benchmark_curve"].empty:
                fig.add_trace(go.Scatter(x=bt["benchmark_curve"].index, y=bt["benchmark_curve"], name="SPY"))
            fig.update_layout(title="Portfolio NAV", xaxis_title="Date", yaxis_title="Value ($)")
            st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("Stock Digest")
    st.markdown("One app for quick sentiment screening, deep stock analysis, GRU model training, and backtesting.")

    tab = st.sidebar.radio("Mode", ["Quick Screener", "Deep Digest", "Train Model", "Backtest"])

    weights = None
    if tab in ("Deep Digest", "Backtest"):
        weights = sidebar_weights()

    if tab == "Quick Screener":
        render_quick_screener()
    elif tab == "Deep Digest":
        render_deep_digest(weights)
    elif tab == "Train Model":
        render_train_model()
    elif tab == "Backtest":
        render_backtest(weights)

    st.sidebar.markdown("---")
    st.sidebar.caption("Not financial advice.")


if __name__ == "__main__":
    main()
