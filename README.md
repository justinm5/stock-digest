# Stock Digest

One app for retail investors. It scores stocks across **fundamentals, valuation, momentum, financial health, and sentiment**, then gives you a single 0–100 score with green (+), yellow (~), and red (-) bullets explaining why or why not to buy.

It also includes a **Quick Screener**, a **GRU sentiment model trainer**, and a **backtester**.

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
- VADER sentiment (fast)
- Optional 256-unit GRU model (trained on 30,599 headlines)

## Data sources

- **Yahoo Finance** — fundamentals, prices, valuation ratios
- **Finnhub** — real-time quotes, peer comparison, company news
- **NewsAPI** — breaking headlines from major outlets

## Quick start

```bash
git clone https://github.com/justinm5/stock-digest.git
cd stock-digest

pip install -r requirements.txt
cp .env.example .env
```

Add free API keys to `.env`:
- **Finnhub**: https://finnhub.io/register
- **NewsAPI**: https://newsapi.org/register

### Run the web app

```bash
streamlit run app.py
```

It opens at `http://localhost:8501` with four tabs:
- **Quick Screener** — fast sentiment + momentum buy/watch/avoid signals
- **Deep Digest** — full fundamental + valuation + momentum + health + sentiment scoring
- **Train Model** — generate 30,599 headlines and train the 256-unit GRU model
- **Backtest** — test the strategy against SPY

### Run from the command line

```bash
# Deep analysis
python cli.py --tickers AAPL MSFT TSLA NVDA META --mode deep

# Quick screener
python cli.py --tickers AAPL MSFT TSLA NVDA META --mode quick

# Backtest
python cli.py --tickers AAPL MSFT TSLA NVDA META --mode backtest --start 2022-01-01 --top-n 5
```

### Train the GRU model

```bash
python -m stock_digest.gru.build_dataset
python -m stock_digest.gru.train
```

Then switch the sentiment engine to `gru` in the app tabs.

## Project structure

```
stock-digest/
├── stock_digest/
│   ├── __init__.py
│   ├── analyzer.py          # Main deep analysis orchestrator
│   ├── backtest.py          # Historical backtest engine
│   ├── config.py            # API keys, weights, peer maps
│   ├── data_fetcher.py      # Yahoo + Finnhub + NewsAPI
│   ├── metrics.py           # Compute fundamentals / valuation / momentum
│   ├── quick_screener.py    # Fast sentiment + momentum screener
│   ├── scoring.py           # 0-100 scoring and peer comparison
│   ├── sentiment_engine.py  # VADER + optional GRU
│   └── gru/                 # GRU training and inference
│       ├── build_dataset.py
│       ├── train.py
│       ├── model.py
│       ├── scraper.py
│       └── recommender.py
├── app.py                   # Unified Streamlit UI
├── cli.py                   # Command-line interface
├── data/
├── models/
├── requirements.txt
├── .env.example
└── README.md
```

## Score legend

| Score | Color | Signal |
|---|---|---|
| 70–100 | Green | Favorable setup |
| 40–69 | Yellow | Mixed / watch |
| 0–39 | Red | Unfavorable setup |

Weights are adjustable in the Streamlit sidebar.

## Notes

- Free API tiers are rate-limited. Finnhub ~60 calls/min; NewsAPI ~100 requests/day.
- Valuation and fundamental scores are relative to peers when peer data is available.
- This is a research and educational tool, not investment advice.
