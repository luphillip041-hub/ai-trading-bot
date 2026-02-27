"""Tests for trading algorithms."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_bot.algorithms.bollinger_bands import BollingerBandsStrategy
from trading_bot.algorithms.macd import MACDStrategy
from trading_bot.algorithms.moving_average import MovingAverageCrossoverStrategy
from trading_bot.algorithms.rsi import RSIStrategy
from trading_bot.config import TradingConfig
from trading_bot.models import Signal


def _make_ohlcv(prices: list[float], start_date: str = "2024-01-01") -> pd.DataFrame:
    """Create a simple OHLCV DataFrame from a list of close prices."""
    dates = pd.date_range(start=start_date, periods=len(prices), freq="D")
    df = pd.DataFrame(
        {
            "open": prices,
            "high": [p * 1.02 for p in prices],
            "low": [p * 0.98 for p in prices],
            "close": prices,
            "volume": [1000000] * len(prices),
        },
        index=dates,
    )
    df.index.name = "date"
    return df


class TestMovingAverageCrossover:
    """Tests for the Moving Average Crossover strategy."""

    def setup_method(self) -> None:
        self.config = TradingConfig(ma_short_window=5, ma_long_window=10)
        self.strategy = MovingAverageCrossoverStrategy(self.config)

    def test_insufficient_data(self) -> None:
        data = _make_ohlcv([100.0] * 5)
        signal = self.strategy.analyze("TEST", data)
        assert signal.signal == Signal.HOLD
        assert "Insufficient data" in signal.reason

    def test_generates_signal(self) -> None:
        # Create enough data for analysis
        prices = list(np.linspace(100, 120, 20))
        data = _make_ohlcv(prices)
        signal = self.strategy.analyze("TEST", data)
        assert signal.ticker == "TEST"
        assert signal.algorithm == "MA Crossover"
        assert signal.signal in [Signal.BUY, Signal.SELL, Signal.HOLD]
        assert 0.0 <= signal.strength <= 1.0

    def test_bullish_crossover(self) -> None:
        # Prices that create a bullish crossover: short MA crosses above long MA
        # Start declining, then rise sharply
        prices = [100.0] * 10 + list(np.linspace(95, 90, 5)) + list(np.linspace(91, 115, 10))
        data = _make_ohlcv(prices)
        signal = self.strategy.analyze("TEST", data)
        # Should be BUY or at least bullish HOLD
        assert signal.signal in [Signal.BUY, Signal.HOLD]

    def test_bearish_trend(self) -> None:
        # Consistently declining prices
        prices = list(np.linspace(150, 100, 25))
        data = _make_ohlcv(prices)
        signal = self.strategy.analyze("TEST", data)
        assert signal.signal in [Signal.SELL, Signal.HOLD]
        assert "Bearish" in signal.reason or signal.signal == Signal.SELL


class TestRSI:
    """Tests for the RSI strategy."""

    def setup_method(self) -> None:
        self.config = TradingConfig(rsi_period=14, rsi_overbought=70.0, rsi_oversold=30.0)
        self.strategy = RSIStrategy(self.config)

    def test_insufficient_data(self) -> None:
        data = _make_ohlcv([100.0] * 5)
        signal = self.strategy.analyze("TEST", data)
        assert signal.signal == Signal.HOLD
        assert "Insufficient data" in signal.reason

    def test_overbought_signal(self) -> None:
        # Steadily increasing prices should push RSI high
        prices = list(np.linspace(100, 200, 30))
        data = _make_ohlcv(prices)
        signal = self.strategy.analyze("TEST", data)
        assert signal.signal in [Signal.SELL, Signal.HOLD]

    def test_oversold_signal(self) -> None:
        # Steadily decreasing prices should push RSI low
        prices = list(np.linspace(200, 100, 30))
        data = _make_ohlcv(prices)
        signal = self.strategy.analyze("TEST", data)
        assert signal.signal in [Signal.BUY, Signal.HOLD]

    def test_neutral_range(self) -> None:
        # Oscillating prices should keep RSI in neutral range
        prices = [100 + (i % 5) for i in range(30)]
        data = _make_ohlcv(prices)
        signal = self.strategy.analyze("TEST", data)
        assert signal.signal == Signal.HOLD


class TestMACD:
    """Tests for the MACD strategy."""

    def setup_method(self) -> None:
        self.config = TradingConfig(macd_fast=12, macd_slow=26, macd_signal=9)
        self.strategy = MACDStrategy(self.config)

    def test_insufficient_data(self) -> None:
        data = _make_ohlcv([100.0] * 10)
        signal = self.strategy.analyze("TEST", data)
        assert signal.signal == Signal.HOLD
        assert "Insufficient data" in signal.reason

    def test_generates_signal(self) -> None:
        prices = list(np.linspace(100, 130, 50))
        data = _make_ohlcv(prices)
        signal = self.strategy.analyze("TEST", data)
        assert signal.ticker == "TEST"
        assert signal.algorithm == "MACD"
        assert signal.signal in [Signal.BUY, Signal.SELL, Signal.HOLD]

    def test_signal_has_macd_info(self) -> None:
        prices = list(np.linspace(100, 130, 50))
        data = _make_ohlcv(prices)
        signal = self.strategy.analyze("TEST", data)
        assert "MACD" in signal.reason


class TestBollingerBands:
    """Tests for the Bollinger Bands strategy."""

    def setup_method(self) -> None:
        self.config = TradingConfig(bb_window=20, bb_std_dev=2.0)
        self.strategy = BollingerBandsStrategy(self.config)

    def test_insufficient_data(self) -> None:
        data = _make_ohlcv([100.0] * 10)
        signal = self.strategy.analyze("TEST", data)
        assert signal.signal == Signal.HOLD
        assert "Insufficient data" in signal.reason

    def test_generates_signal(self) -> None:
        prices = [100 + np.random.randn() * 2 for _ in range(30)]
        data = _make_ohlcv(prices)
        signal = self.strategy.analyze("TEST", data)
        assert signal.ticker == "TEST"
        assert signal.algorithm == "Bollinger Bands"
        assert signal.signal in [Signal.BUY, Signal.SELL, Signal.HOLD]

    def test_price_at_lower_band(self) -> None:
        # Stable prices then a sharp drop
        prices = [100.0] * 25 + [85.0, 84.0, 83.0, 82.0, 80.0]
        data = _make_ohlcv(prices)
        signal = self.strategy.analyze("TEST", data)
        assert signal.signal in [Signal.BUY, Signal.HOLD]

    def test_price_at_upper_band(self) -> None:
        # Stable prices then a sharp rise
        prices = [100.0] * 25 + [115.0, 116.0, 117.0, 118.0, 120.0]
        data = _make_ohlcv(prices)
        signal = self.strategy.analyze("TEST", data)
        assert signal.signal in [Signal.SELL, Signal.HOLD]
