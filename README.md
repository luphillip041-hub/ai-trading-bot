# AI Trading Bot

An AI-powered algorithmic trading bot that uses Yahoo Finance data to identify trading opportunities through multiple technical analysis strategies.

> **Disclaimer**: This bot is for educational and research purposes only. It does not execute real trades. Always do your own research before making investment decisions.

## Features

- **Multiple Trading Algorithms**:
  - **Moving Average Crossover**: Detects golden/death crosses using configurable short/long SMAs
  - **RSI (Relative Strength Index)**: Identifies overbought/oversold conditions
  - **MACD (Moving Average Convergence Divergence)**: Detects momentum shifts via MACD/signal line crossovers
  - **Bollinger Bands**: Identifies price extremes relative to volatility bands

- **Signal Aggregation**: Combines signals from all algorithms using a consensus voting system weighted by signal strength

- **Backtesting Engine**: Test strategies against historical data with detailed performance metrics (Sharpe ratio, max drawdown, win rate)

- **Portfolio Simulation**: Simulate portfolio management with configurable position sizing, stop-loss, and take-profit levels

- **Risk Management**: Built-in stop-loss and take-profit mechanisms, position size limits, and maximum concurrent positions

- **Rich CLI**: Beautiful terminal output with color-coded signals, formatted tables, and detailed analysis

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd ai-trading-bot

# Install with pip
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

## Usage

### Scan for Trading Signals

Scan default watchlist:
```bash
trading-bot scan
```

Scan specific tickers:
```bash
trading-bot scan -t AAPL,MSFT,GOOGL,TSLA
```

Scan with custom period and consensus:
```bash
trading-bot scan -t AAPL,NVDA -p 1y -c 3
```

### Backtest a Strategy

```bash
trading-bot backtest AAPL -p 1y
trading-bot backtest TSLA -p 2y --capital 50000
```

### Get Ticker Info

```bash
trading-bot info AAPL
```

### Portfolio Simulation

```bash
trading-bot portfolio AAPL,MSFT,GOOGL,NVDA --capital 100000
```

### Verbose Mode

Add `-v` for detailed logging:
```bash
trading-bot -v scan -t AAPL
```

## Configuration

All settings can be configured via environment variables with the `TRADING_BOT_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `TRADING_BOT_DATA_PERIOD` | `6mo` | Historical data period |
| `TRADING_BOT_DATA_INTERVAL` | `1d` | Data interval |
| `TRADING_BOT_MA_SHORT_WINDOW` | `20` | Short MA window |
| `TRADING_BOT_MA_LONG_WINDOW` | `50` | Long MA window |
| `TRADING_BOT_RSI_PERIOD` | `14` | RSI period |
| `TRADING_BOT_RSI_OVERBOUGHT` | `70` | RSI overbought threshold |
| `TRADING_BOT_RSI_OVERSOLD` | `30` | RSI oversold threshold |
| `TRADING_BOT_MIN_CONSENSUS` | `2` | Min algorithms for consensus |
| `TRADING_BOT_INITIAL_CAPITAL` | `100000` | Starting capital |
| `TRADING_BOT_STOP_LOSS_PCT` | `0.05` | Stop loss (5%) |
| `TRADING_BOT_TAKE_PROFIT_PCT` | `0.15` | Take profit (15%) |

## Project Structure

```
ai-trading-bot/
├── trading_bot/
│   ├── __init__.py
│   ├── cli.py              # CLI interface
│   ├── config.py            # Configuration management
│   ├── models.py            # Data models (signals, positions, trades)
│   ├── data_fetcher.py      # Yahoo Finance data fetcher
│   ├── strategy_engine.py   # Signal aggregation engine
│   ├── portfolio.py         # Portfolio management
│   ├── backtester.py        # Backtesting engine
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── base.py          # Base strategy class
│   │   ├── moving_average.py # MA Crossover strategy
│   │   ├── rsi.py           # RSI strategy
│   │   ├── macd.py          # MACD strategy
│   │   └── bollinger_bands.py # Bollinger Bands strategy
│   └── utils/
│       └── __init__.py
├── tests/
│   ├── __init__.py
│   ├── test_algorithms.py
│   ├── test_portfolio.py
│   └── test_backtester.py
├── pyproject.toml
└── README.md
```

## How It Works

1. **Data Collection**: Fetches historical OHLCV data from Yahoo Finance for configured tickers
2. **Signal Generation**: Each algorithm independently analyzes the data and produces BUY/SELL/HOLD signals with confidence scores
3. **Signal Aggregation**: The strategy engine combines signals using a consensus system — a trade signal is only generated when the minimum number of algorithms agree
4. **Risk Management**: Position sizing, stop-loss, and take-profit levels are automatically calculated
5. **Execution**: In simulation/backtest mode, trades are executed in the portfolio tracker

## License

MIT
