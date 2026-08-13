"""Command-line interface for Stock Digest."""
import argparse
import sys

from stock_digest.analyzer import StockAnalyzer
from stock_digest.backtest import run_backtest
from stock_digest.config import DEFAULT_TICKERS


def main():
    parser = argparse.ArgumentParser(description="Stock Digest — comprehensive stock scoring")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS, help="Tickers to analyze")
    parser.add_argument("--top", type=int, default=10, help="Show top N results")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--start", default="2022-01-01", help="Backtest start date")
    parser.add_argument("--end", default=None, help="Backtest end date")
    parser.add_argument("--top-n", type=int, default=5, help="Backtest top N holdings")
    args = parser.parse_args()

    if args.backtest:
        import datetime
        end = args.end or datetime.date.today().isoformat()
        print("Running backtest...")
        result = run_backtest(args.tickers, args.start, end, top_n=args.top_n)
        if "error" in result:
            print(f"Error: {result['error']}")
            sys.exit(1)
        print(f"Initial cash: ${result['initial_cash']}")
        print(f"Final NAV: ${result['final_nav']}")
        print(f"Strategy return: {result['total_return_pct']}%")
        print(f"SPY return: {result['benchmark_return_pct']}%")
        print(f"Max drawdown: {result['max_drawdown_pct']}%")
        print(f"Sharpe ratio: {result['sharpe_ratio']}")
        print(f"Trades: {result['num_trades']}")
        return

    analyzer = StockAnalyzer()
    print("Analyzing tickers...")
    results = analyzer.analyze_tickers(args.tickers)

    print(f"\n{'Rank':<5} {'Ticker':<8} {'Score':<7} {'Fund':<6} {'Val':<6} {'Mom':<6} {'Health':<7} {'Sent':<6} {'Signal':<12} {'Top reason'}")
    print("-" * 110)
    for i, r in enumerate(results[:args.top], 1):
        if "error" in r:
            print(f"{i:<5} {r['ticker']:<8} ERROR: {r['error']}")
            continue
        cats = r["category_scores"]
        top_reason = (r.get("why_buy") or r.get("why_not") or ["Mixed"])[0]
        print(
            f"{i:<5} {r['ticker']:<8} {r['score']:<7.1f} "
            f"{cats['fundamentals']['score']:<6.1f} {cats['valuation']['score']:<6.1f} "
            f"{cats['momentum']['score']:<6.1f} {cats['financial_health']['score']:<7.1f} "
            f"{cats['sentiment']['score']:<6.1f} {r['color'].upper():<12} {top_reason}"
        )


if __name__ == "__main__":
    main()
