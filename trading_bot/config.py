"""Configuration management for the trading bot."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class TradingConfig(BaseSettings):
    """Trading bot configuration."""

    # Watchlist of stock tickers to monitor
    watchlist: list[str] = Field(
        default=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "SPY", "QQQ", "AMD"],
        description="List of stock tickers to monitor",
    )

    # Data settings
    data_period: str = Field(
        default="6mo", description="Historical data period (e.g., 1mo, 3mo, 6mo, 1y)"
    )
    data_interval: str = Field(default="1d", description="Data interval (e.g., 1d, 1h, 5m)")

    # Moving Average Crossover settings
    ma_short_window: int = Field(default=20, description="Short moving average window")
    ma_long_window: int = Field(default=50, description="Long moving average window")

    # RSI settings
    rsi_period: int = Field(default=14, description="RSI calculation period")
    rsi_overbought: float = Field(default=70.0, description="RSI overbought threshold")
    rsi_oversold: float = Field(default=30.0, description="RSI oversold threshold")

    # MACD settings
    macd_fast: int = Field(default=12, description="MACD fast EMA period")
    macd_slow: int = Field(default=26, description="MACD slow EMA period")
    macd_signal: int = Field(default=9, description="MACD signal line period")

    # Bollinger Bands settings
    bb_window: int = Field(default=20, description="Bollinger Bands window")
    bb_std_dev: float = Field(
        default=2.0, description="Bollinger Bands standard deviation multiplier"
    )

    # Strategy settings
    min_consensus: int = Field(
        default=2,
        description="Minimum number of algorithms that must agree for a trade signal",
    )

    # Portfolio settings
    initial_capital: float = Field(default=100000.0, description="Initial capital for backtesting")
    position_size_pct: float = Field(
        default=0.1, description="Position size as percentage of portfolio (0.0 to 1.0)"
    )
    max_positions: int = Field(default=10, description="Maximum number of concurrent positions")

    # Risk management
    stop_loss_pct: float = Field(default=0.05, description="Stop loss percentage (0.0 to 1.0)")
    take_profit_pct: float = Field(default=0.15, description="Take profit percentage (0.0 to 1.0)")

    model_config = {"env_prefix": "TRADING_BOT_"}
