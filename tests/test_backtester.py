"""Tests for the backtesting engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_bot.backtester import Backtester, BacktestResult
from trading_bot.config import TradingConfig


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


class TestBacktester:
    """Tests for the Backtester."""

    def setup_method(self) -> None:
        self.config = TradingConfig(
            initial_capital=100000.0,
            ma_short_window=5,
            ma_long_window=10,
            rsi_period=14,
            macd_fast=12,
            macd_slow=26,
            macd_signal=9,
            bb_window=20,
        )
        self.backtester = Backtester(self.config)

    def test_insufficient_data(self) -> None:
        data = _make_ohlcv([100.0] * 10)
        result = self.backtester.run("TEST", data, lookback=60)
        assert isinstance(result, BacktestResult)
        assert result.total_trades == 0

    def test_basic_backtest(self) -> None:
        # Generate realistic-ish price data
        np.random.seed(42)
        prices = [100.0]
        for _ in range(149):
            change = np.random.randn() * 2
            prices.append(max(prices[-1] + change, 10.0))

        data = _make_ohlcv(prices)
        result = self.backtester.run("TEST", data, lookback=60)

        assert isinstance(result, BacktestResult)
        assert result.ticker == "TEST"
        assert len(result.equity_curve) > 0

    def test_backtest_summary(self) -> None:
        np.random.seed(42)
        prices = [100.0]
        for _ in range(149):
            change = np.random.randn() * 2
            prices.append(max(prices[-1] + change, 10.0))

        data = _make_ohlcv(prices)
        result = self.backtester.run("TEST", data, lookback=60)
        summary = result.get_summary()

        assert "ticker" in summary
        assert "period" in summary
        assert "initial_capital" in summary
        assert "final_value" in summary
        assert "total_return_pct" in summary
        assert "total_trades" in summary
        assert "win_rate_pct" in summary
        assert "max_drawdown_pct" in summary
        assert "sharpe_ratio" in summary

    def test_uptrend_backtest(self) -> None:
        # Strong uptrend should produce positive returns eventually
        prices = list(np.linspace(100, 200, 150))
        # Add some noise
        np.random.seed(42)
        prices = [p + np.random.randn() * 2 for p in prices]
        data = _make_ohlcv(prices)
        result = self.backtester.run("TEST", data, lookback=60)
        assert isinstance(result, BacktestResult)

    def test_equity_curve_recorded(self) -> None:
        np.random.seed(42)
        prices = [100.0]
        for _ in range(149):
            change = np.random.randn() * 2
            prices.append(max(prices[-1] + change, 10.0))

        data = _make_ohlcv(prices)
        result = self.backtester.run("TEST", data, lookback=60)

        # Should have equity curve entries for each day after lookback
        expected_entries = len(data) - 60
        assert len(result.equity_curve) == expected_entries

        # Each entry should have required fields
        for entry in result.equity_curve:
            assert "date" in entry
            assert "equity" in entry
            assert "cash" in entry
            assert "price" in entry

    def test_max_drawdown(self) -> None:
        np.random.seed(42)
        prices = [100.0]
        for _ in range(149):
            change = np.random.randn() * 2
            prices.append(max(prices[-1] + change, 10.0))

        data = _make_ohlcv(prices)
        result = self.backtester.run("TEST", data, lookback=60)
        assert result.max_drawdown >= 0.0
