"""Command-line interface for the trading bot."""

from __future__ import annotations

import argparse
import logging
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from trading_bot.backtester import Backtester
from trading_bot.config import TradingConfig
from trading_bot.data_fetcher import DataFetcher
from trading_bot.models import Signal
from trading_bot.portfolio import Portfolio
from trading_bot.strategy_engine import StrategyEngine

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_scan(args: argparse.Namespace) -> None:
    """Scan watchlist for trading signals."""
    config = TradingConfig()

    if args.tickers:
        config.watchlist = [t.strip().upper() for t in args.tickers.split(",")]

    if args.period:
        config.data_period = args.period

    if args.consensus:
        config.min_consensus = args.consensus

    console.print(Panel("[bold blue]AI Trading Bot - Market Scanner[/bold blue]"))
    console.print(f"Watchlist: {', '.join(config.watchlist)}")
    console.print(f"Period: {config.data_period} | Interval: {config.data_interval}")
    console.print(f"Min consensus: {config.min_consensus} algorithms")
    console.print()

    # Fetch data
    with console.status("[bold green]Fetching market data..."):
        fetcher = DataFetcher(config)
        market_data = fetcher.fetch_watchlist_data()

    if not market_data:
        console.print("[red]No market data available. Check your internet connection.[/red]")
        return

    # Run strategy engine
    with console.status("[bold green]Analyzing signals..."):
        engine = StrategyEngine(config)
        signals = engine.scan_all(market_data)

    # Display results
    _display_scan_results(signals)


def _display_scan_results(signals: list) -> None:
    """Display scan results in a formatted table."""
    # Action signals (BUY/SELL)
    action_signals = [s for s in signals if s.signal != Signal.HOLD]
    hold_signals = [s for s in signals if s.signal == Signal.HOLD]

    if action_signals:
        table = Table(title="🔔 Trade Signals", show_header=True, header_style="bold magenta")
        table.add_column("Ticker", style="cyan", width=8)
        table.add_column("Signal", width=8)
        table.add_column("Price", justify="right", width=10)
        table.add_column("Strength", justify="right", width=10)
        table.add_column("Buy", justify="center", width=5)
        table.add_column("Sell", justify="center", width=5)
        table.add_column("Hold", justify="center", width=5)

        for sig in action_signals:
            signal_style = "bold green" if sig.signal == Signal.BUY else "bold red"
            table.add_row(
                sig.ticker,
                f"[{signal_style}]{sig.signal.value}[/{signal_style}]",
                f"${sig.price:.2f}",
                f"{sig.avg_strength:.2f}",
                str(sig.buy_count),
                str(sig.sell_count),
                str(sig.hold_count),
            )

        console.print(table)
        console.print()

        # Show detailed reasons
        for sig in action_signals:
            signal_style = "green" if sig.signal == Signal.BUY else "red"
            console.print(
                Panel(
                    "\n".join(sig.reasons),
                    title=f"[{signal_style}]{sig.ticker} - {sig.signal.value}[/{signal_style}]",
                    border_style=signal_style,
                )
            )
    else:
        console.print("[yellow]No actionable trade signals found.[/yellow]")

    # Hold signals summary
    if hold_signals:
        console.print()
        table = Table(title="Holds (No Action)", show_header=True, header_style="bold")
        table.add_column("Ticker", style="cyan", width=8)
        table.add_column("Price", justify="right", width=10)
        table.add_column("Buy", justify="center", width=5)
        table.add_column("Sell", justify="center", width=5)

        for sig in hold_signals:
            table.add_row(
                sig.ticker,
                f"${sig.price:.2f}",
                str(sig.buy_count),
                str(sig.sell_count),
            )

        console.print(table)


def cmd_backtest(args: argparse.Namespace) -> None:
    """Run a backtest on a ticker."""
    config = TradingConfig()

    if args.period:
        config.data_period = args.period

    if args.capital:
        config.initial_capital = args.capital

    ticker = args.ticker.upper()

    console.print(Panel("[bold blue]AI Trading Bot - Backtester[/bold blue]"))
    console.print(f"Ticker: {ticker}")
    console.print(f"Period: {config.data_period}")
    console.print(f"Initial capital: ${config.initial_capital:,.2f}")
    console.print()

    # Fetch data
    with console.status("[bold green]Fetching historical data..."):
        fetcher = DataFetcher(config)
        data = fetcher.fetch_ticker_data(ticker, period=config.data_period)

    if data.empty:
        console.print(f"[red]No data available for {ticker}[/red]")
        return

    console.print(f"Data range: {data.index[0]} to {data.index[-1]} ({len(data)} periods)")
    console.print()

    # Run backtest
    with console.status("[bold green]Running backtest..."):
        backtester = Backtester(config)
        result = backtester.run(ticker, data)

    # Display results
    summary = result.get_summary()

    table = Table(
        title=f"Backtest Results: {ticker}", show_header=True, header_style="bold magenta"
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    pnl_style = "green" if summary["total_return_pct"] >= 0 else "red"

    table.add_row("Period", summary["period"])
    table.add_row("Initial Capital", f"${summary['initial_capital']:,.2f}")
    table.add_row("Final Value", f"[{pnl_style}]${summary['final_value']:,.2f}[/{pnl_style}]")
    table.add_row("Total Return", f"[{pnl_style}]{summary['total_return_pct']:+.2f}%[/{pnl_style}]")
    table.add_row("Total Trades", str(summary["total_trades"]))
    table.add_row("Win Rate", f"{summary['win_rate_pct']:.1f}%")
    table.add_row("Max Drawdown", f"[red]{summary['max_drawdown_pct']:.2f}%[/red]")
    table.add_row("Sharpe Ratio", f"{summary['sharpe_ratio']:.2f}")
    table.add_row("Winning Trades", f"[green]{summary['winning_trades']}[/green]")
    table.add_row("Losing Trades", f"[red]{summary['losing_trades']}[/red]")

    console.print(table)

    # Show trade log
    if result.signals_log:
        console.print()
        trade_table = Table(title="Trade Log", show_header=True, header_style="bold")
        trade_table.add_column("Date", width=12)
        trade_table.add_column("Action", width=12)
        trade_table.add_column("Price", justify="right", width=10)
        trade_table.add_column("Strength", justify="right", width=10)
        trade_table.add_column("P&L", justify="right", width=12)

        for entry in result.signals_log:
            action = entry["action"]
            action_style = {
                "BUY": "green",
                "SELL": "red",
                "STOP_LOSS": "bold red",
                "TAKE_PROFIT": "bold green",
            }.get(action, "white")

            pnl_str = ""
            if "pnl" in entry:
                pnl = entry["pnl"]
                pnl_color = "green" if pnl >= 0 else "red"
                pnl_str = f"[{pnl_color}]${pnl:+,.2f}[/{pnl_color}]"

            trade_table.add_row(
                entry["date"][:10],
                f"[{action_style}]{action}[/{action_style}]",
                f"${entry['price']:.2f}",
                f"{entry['signal_strength']:.2f}",
                pnl_str,
            )

        console.print(trade_table)


def cmd_info(args: argparse.Namespace) -> None:
    """Show detailed info for a ticker."""
    config = TradingConfig()
    ticker = args.ticker.upper()

    console.print(Panel(f"[bold blue]Ticker Info: {ticker}[/bold blue]"))

    with console.status("[bold green]Fetching ticker info..."):
        fetcher = DataFetcher(config)
        info = fetcher.get_ticker_info(ticker)

    if "error" in info:
        console.print(f"[red]Error: {info['error']}[/red]")
        return

    table = Table(show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value")

    table.add_row("Symbol", info.get("symbol", "N/A"))
    table.add_row("Name", info.get("name", "N/A"))
    table.add_row("Sector", info.get("sector", "N/A"))
    table.add_row("Industry", info.get("industry", "N/A"))

    market_cap = info.get("market_cap", 0)
    if market_cap >= 1e12:
        mc_str = f"${market_cap / 1e12:.2f}T"
    elif market_cap >= 1e9:
        mc_str = f"${market_cap / 1e9:.2f}B"
    elif market_cap >= 1e6:
        mc_str = f"${market_cap / 1e6:.2f}M"
    else:
        mc_str = f"${market_cap:,.0f}"

    table.add_row("Market Cap", mc_str)
    table.add_row("P/E Ratio", f"{info.get('pe_ratio', 'N/A')}")
    table.add_row("52-Week High", f"${info.get('52w_high', 0):.2f}")
    table.add_row("52-Week Low", f"${info.get('52w_low', 0):.2f}")
    table.add_row("Avg Volume", f"{info.get('avg_volume', 0):,}")

    div_yield = info.get("dividend_yield", 0)
    if div_yield:
        table.add_row("Dividend Yield", f"{div_yield * 100:.2f}%")
    else:
        table.add_row("Dividend Yield", "N/A")

    console.print(table)


def cmd_portfolio(args: argparse.Namespace) -> None:
    """Show portfolio simulation status."""
    config = TradingConfig()

    if args.capital:
        config.initial_capital = args.capital

    console.print(Panel("[bold blue]AI Trading Bot - Portfolio Simulator[/bold blue]"))

    tickers = [t.strip().upper() for t in args.tickers.split(",")]

    with console.status("[bold green]Fetching data and generating signals..."):
        fetcher = DataFetcher(config)
        engine = StrategyEngine(config)
        portfolio = Portfolio(config)

        for ticker in tickers:
            data = fetcher.fetch_ticker_data(ticker)
            if data.empty:
                continue

            signal = engine.analyze_ticker(ticker, data)
            current_price = float(data["close"].iloc[-1])

            if signal.signal == Signal.BUY:
                portfolio.open_position(ticker, current_price)

        # Update prices
        prices = {}
        for ticker in tickers:
            price = fetcher.get_current_price(ticker)
            if price is not None:
                prices[ticker] = price

        portfolio.update_prices(prices)

    # Display portfolio
    summary = portfolio.get_summary()

    table = Table(title="Portfolio Summary", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Cash", f"${summary['cash']:,.2f}")
    table.add_row("Positions", str(summary["positions_count"]))
    table.add_row("Total Value", f"${summary['total_value']:,.2f}")

    pnl_style = "green" if summary["total_pnl"] >= 0 else "red"
    table.add_row("Total P&L", f"[{pnl_style}]${summary['total_pnl']:+,.2f}[/{pnl_style}]")
    table.add_row("Total P&L %", f"[{pnl_style}]{summary['total_pnl_pct']:+.2f}%[/{pnl_style}]")

    console.print(table)

    # Show positions
    if portfolio.positions:
        console.print()
        pos_table = Table(title="Open Positions", show_header=True, header_style="bold")
        pos_table.add_column("Ticker", style="cyan")
        pos_table.add_column("Shares", justify="right")
        pos_table.add_column("Entry", justify="right")
        pos_table.add_column("Current", justify="right")
        pos_table.add_column("P&L", justify="right")
        pos_table.add_column("P&L %", justify="right")
        pos_table.add_column("Stop Loss", justify="right")
        pos_table.add_column("Take Profit", justify="right")

        for pos in portfolio.positions.values():
            pnl_style = "green" if pos.unrealized_pnl >= 0 else "red"
            pos_table.add_row(
                pos.ticker,
                f"{pos.shares:.4f}",
                f"${pos.entry_price:.2f}",
                f"${pos.current_price:.2f}",
                f"[{pnl_style}]${pos.unrealized_pnl:+,.2f}[/{pnl_style}]",
                f"[{pnl_style}]{pos.unrealized_pnl_pct:+.2f}%[/{pnl_style}]",
                f"${pos.stop_loss:.2f}",
                f"${pos.take_profit:.2f}",
            )

        console.print(pos_table)


def cmd_web(args: argparse.Namespace) -> None:
    """Launch the web dashboard."""
    from trading_bot.web.app import run_server

    console.print(
        f"[bold green]Starting web dashboard at http://{args.host}:{args.port}[/bold green]"
    )
    console.print("[dim]Press Ctrl+C to stop[/dim]")
    run_server(host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="trading-bot",
        description="AI Trading Bot - Algorithmic trading using Yahoo Finance data",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan watchlist for trading signals")
    scan_parser.add_argument(
        "-t", "--tickers", type=str,
        help="Comma-separated list of tickers (overrides default watchlist)"
    )
    scan_parser.add_argument(
        "-p", "--period", type=str, help="Data period (e.g., 1mo, 3mo, 6mo, 1y)"
    )
    scan_parser.add_argument(
        "-c", "--consensus", type=int, help="Minimum algorithm consensus for signals"
    )

    # Backtest command
    bt_parser = subparsers.add_parser("backtest", help="Run a backtest on a ticker")
    bt_parser.add_argument("ticker", type=str, help="Stock ticker to backtest")
    bt_parser.add_argument("-p", "--period", type=str, help="Data period (e.g., 6mo, 1y, 2y)")
    bt_parser.add_argument("--capital", type=float, help="Initial capital (default: 100000)")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show detailed info for a ticker")
    info_parser.add_argument("ticker", type=str, help="Stock ticker")

    # Portfolio command
    port_parser = subparsers.add_parser("portfolio", help="Run portfolio simulation")
    port_parser.add_argument("tickers", type=str, help="Comma-separated list of tickers")
    port_parser.add_argument("--capital", type=float, help="Initial capital (default: 100000)")

    # Web command
    web_parser = subparsers.add_parser("web", help="Launch the web dashboard")
    web_parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind (default: 0.0.0.0)"
    )
    web_parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind (default: 8000)"
    )

    return parser


def main() -> None:
    """Main entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    commands = {
        "scan": cmd_scan,
        "backtest": cmd_backtest,
        "info": cmd_info,
        "portfolio": cmd_portfolio,
        "web": cmd_web,
    }

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        handler(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logging.exception("Unhandled error")
        sys.exit(1)


if __name__ == "__main__":
    main()
