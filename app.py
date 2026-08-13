"""Streamlit app for Stock Digest."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from stock_digest.analyzer import StockAnalyzer
from stock_digest.backtest import run_backtest
from stock_digest.config import DEFAULT_TICKERS

st.set_page_config(page_title="Stock Digest", page_icon="", layout="wide")


def color_card(score: float):
    if score >= 70:
        return "#10b981"
    if score >= 40:
        return "#f59e0b"
    return "#ef4444"


def main():
    st.title("Stock Digest")
    st.markdown("Comprehensive stock scoring: fundamentals, valuation, momentum, financial health, and sentiment.")

    with st.sidebar:
        st.header("Settings")
        ticker_input = st.text_area("Tickers (comma-separated)", ", ".join(DEFAULT_TICKERS), height=120)
        show_peers = st.checkbox("Compare vs peers / industry", value=True)
        run = st.button("Analyze", type="primary")

        st.markdown("---")
        st.markdown("**Scoring weights**")
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

    if not run:
        st.info("Enter tickers and click **Analyze**.")
        return

    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    if not tickers:
        st.warning("Enter at least one ticker.")
        return

    analyzer = StockAnalyzer(weights=weights)

    with st.spinner("Analyzing stocks..."):
        results = analyzer.analyze_tickers(tickers)

    # Top table
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

    # Detail cards
    st.subheader("Detailed breakdown")
    for r in results:
        color = color_card(r["score"])
        with st.expander(f"{r['ticker']} — {r.get('name', '')}  |  Score: {r['score']}", expanded=False):
            col1, col2, col3 = st.columns([1, 1, 1])
            col1.metric("Price", f"${r.get('price')}" if r.get('price') else "N/A", f"{r.get('change_pct', 0):.2f}%")
            col2.metric("Overall Score", r["score"])
            col3.markdown(f"<span style='color:{color};font-size:24px;font-weight:bold;'>{r['color'].upper()}</span>", unsafe_allow_html=True)

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

    # Backtest section
    st.markdown("---")
    st.subheader("Backtest")
    col_bt1, col_bt2, col_bt3 = st.columns(3)
    start = col_bt1.date_input("Start", value=pd.to_datetime("2022-01-01"))
    end = col_bt2.date_input("End", value=pd.to_datetime("today"))
    top_n_bt = col_bt3.slider("Top N holdings", 1, 10, 5)

    if st.button("Run backtest"):
        with st.spinner("Running backtest..."):
            bt = run_backtest(
                tickers,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                top_n=top_n_bt,
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


if __name__ == "__main__":
    main()
