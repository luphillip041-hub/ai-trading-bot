"""FastAPI web application for the trading bot dashboard."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from trading_bot.backtester import Backtester
from trading_bot.config import TradingConfig
from trading_bot.data_fetcher import DataFetcher
from trading_bot.models import Signal
from trading_bot.portfolio import Portfolio
from trading_bot.strategy_engine import StrategyEngine

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
app = FastAPI(title="AI Trading Bot", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    """Serve the main dashboard page."""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/backtest", response_class=HTMLResponse)
async def backtest_page(request: Request) -> HTMLResponse:
    """Serve the backtest page."""
    return templates.TemplateResponse("backtest.html", {"request": request})


@app.get("/portfolio", response_class=HTMLResponse)
async def portfolio_page(request: Request) -> HTMLResponse:
    """Serve the portfolio page."""
    return templates.TemplateResponse("portfolio.html", {"request": request})


@app.get("/api/scan")
async def api_scan(
    tickers: str = Query(
        default="AAPL,MSFT,GOOGL,AMZN,TSLA,META,NVDA,SPY,QQQ,AMD",
        description="Comma-separated ticker symbols",
    ),
    period: str = Query(default="6mo", description="Data period"),
    consensus: int = Query(default=2, description="Min consensus"),
) -> dict:
    """Scan tickers for trading signals."""
    config = TradingConfig()
    config.watchlist = [t.strip().upper() for t in tickers.split(",")]
    config.data_period = period
    config.min_consensus = consensus

    fetcher = DataFetcher(config)
    engine = StrategyEngine(config)

    market_data = fetcher.fetch_watchlist_data()
    signals = engine.scan_all(market_data)

    results = []
    for sig in signals:
        results.append({
            "ticker": sig.ticker,
            "signal": sig.signal.value,
            "buy_count": sig.buy_count,
            "sell_count": sig.sell_count,
            "hold_count": sig.hold_count,
            "avg_strength": round(sig.avg_strength, 3),
            "price": round(sig.price, 2),
            "reasons": sig.reasons,
        })

    return {"signals": results}


@app.get("/api/backtest/{ticker}")
async def api_backtest(
    ticker: str,
    period: str = Query(default="1y", description="Data period"),
    capital: float = Query(default=100000.0, description="Initial capital"),
) -> dict:
    """Run a backtest on a ticker."""
    ticker = ticker.upper()
    config = TradingConfig()
    config.data_period = period
    config.initial_capital = capital

    fetcher = DataFetcher(config)
    data = fetcher.fetch_ticker_data(ticker, period=period)

    if data.empty:
        return {"error": f"No data available for {ticker}"}

    backtester = Backtester(config)
    result = backtester.run(ticker, data)

    return {
        "summary": result.get_summary(),
        "equity_curve": result.equity_curve,
        "signals_log": result.signals_log,
    }


@app.get("/api/info/{ticker}")
async def api_info(ticker: str) -> dict:
    """Get detailed info for a ticker."""
    ticker = ticker.upper()
    config = TradingConfig()
    fetcher = DataFetcher(config)
    return fetcher.get_ticker_info(ticker)


@app.get("/api/portfolio")
async def api_portfolio(
    tickers: str = Query(
        default="AAPL,MSFT,GOOGL,NVDA",
        description="Comma-separated ticker symbols",
    ),
    capital: float = Query(default=100000.0, description="Initial capital"),
) -> dict:
    """Run portfolio simulation."""
    config = TradingConfig()
    config.initial_capital = capital
    ticker_list = [t.strip().upper() for t in tickers.split(",")]

    fetcher = DataFetcher(config)
    engine = StrategyEngine(config)
    portfolio = Portfolio(config)

    positions_detail = []
    signals_detail = []

    for t in ticker_list:
        data = fetcher.fetch_ticker_data(t)
        if data.empty:
            continue

        signal = engine.analyze_ticker(t, data)
        current_price = float(data["close"].iloc[-1])

        signals_detail.append({
            "ticker": t,
            "signal": signal.signal.value,
            "strength": round(signal.avg_strength, 3),
            "price": round(current_price, 2),
        })

        if signal.signal == Signal.BUY:
            portfolio.open_position(t, current_price)

    # Update prices
    prices = {}
    for t in ticker_list:
        price = fetcher.get_current_price(t)
        if price is not None:
            prices[t] = price
    portfolio.update_prices(prices)

    for pos in portfolio.positions.values():
        positions_detail.append({
            "ticker": pos.ticker,
            "shares": round(pos.shares, 4),
            "entry_price": round(pos.entry_price, 2),
            "current_price": round(pos.current_price, 2),
            "pnl": round(pos.unrealized_pnl, 2),
            "pnl_pct": round(pos.unrealized_pnl_pct, 2),
            "stop_loss": round(pos.stop_loss, 2),
            "take_profit": round(pos.take_profit, 2),
        })

    return {
        "summary": portfolio.get_summary(),
        "positions": positions_detail,
        "signals": signals_detail,
    }


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the web server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)
