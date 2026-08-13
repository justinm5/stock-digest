# Stock Digest

A simple, comprehensive stock analyzer for retail investors. It scores stocks across **fundamentals, valuation, momentum, financial health, and sentiment** — then tells you why or why not to buy with green, yellow, and red bullets.

## What it scores

### Fundamentals (vs peers / industry)
- Revenue growth
- EPS growth
- Gross, operating, and net margins
- ROE and ROIC
- Free cash flow growth and FCF margin

### Valuation (vs peers / industry)
- P/E, Forward P/E, PEG
- EV/EBITDA
- Price / Free Cash Flow
- Price / Sales
- FCF yield

### Financial Health
- Debt-to-equity
- Interest coverage
- Share dilution
- Earnings consistency

### Momentum
- 1m / 3m / 6m / 12m returns
- RSI
- 50-day vs 200-day moving average
- Volume trend

### Sentiment
- Latest headlines from Finnhub + NewsAPI
- VADER sentiment scoring

## How it works

1. Fetches data from **Yahoo Finance** (fundamentals, prices) and **Finnhub** (real-time quotes, peers, news).
2. Computes metrics and compares them against peers in the same sector.
3. Scores each category 0–100.
4. Combines categories into a final 0–100 score using adjustable weights.
5. Generates **green (+), yellow (~), and red (-)** bullets explaining the bullish and bearish case.
6. **Backtests** the strategy against SPY using historical momentum signals.

## Quick start

```bash
git clone https://github.com/justinm5/stock-digest.git
cd stock-digest

pip install -r requirements.txt
cp .env.example .env
```

Add your free API keys to `.env`:
- **Finnhub**: https://finnhub.io/register
- **NewsAPI**: https://newsapi.org/register

### Run the web app

```bash
streamlit run app.py
```

### Run from the command line

```bash
python cli.py --tickers AAPL MSFT TSLA NVDA META
```

### Run a backtest

```bash
python cli.py --tickers AAPL MSFT TSLA NVDA META AMZN GOOGL --backtest --start 2022-01-01 --top-n 5
```

## Project structure

```
stock-digest/
├── stock_digest/
│   ├── __init__.py
│   ├── analyzer.py       # Main scoring orchestrator
│   ├── backtest.py       # Historical backtest engine
│   ├── config.py         # API keys, weights, peer maps
│   ├── data_fetcher.py   # Yahoo + Finnhub + NewsAPI
│   ├── metrics.py        # Compute fundamentals / valuation / momentum
│   ├── scoring.py        # 0-100 scoring and peer comparison
│   └── sentiment_engine.py # VADER sentiment
├── app.py                # Streamlit UI
├── cli.py                # Command-line tool
├── requirements.txt
├── .env.example
└── README.md
```

## Understanding the score

| Score | Color | Signal |
|---|---|---|
| 70–100 | Green | Favorable setup |
| 40–69 | Yellow | Mixed / watch |
| 0–39 | Red | Unfavorable setup |

The final score is a weighted average:
- Fundamentals: 25%
- Valuation: 20%
- Momentum: 20%
- Financial Health: 20%
- Sentiment: 15%

## Backtest

The built-in backtest walks through historical prices and rebalances monthly into the top-N ranked stocks by momentum. It compares the portfolio to **SPY** buy-and-hold and reports total return, max drawdown, and Sharpe ratio.

## Notes

- Free API tiers are rate-limited (Finnhub ~60 calls/min, NewsAPI ~100 requests/day).
- Valuation and fundamental scores are relative to peers when peer data is available.
- This is a research and educational tool, not financial advice.
