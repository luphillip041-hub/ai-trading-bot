"""MACD (Moving Average Convergence Divergence) strategy."""

from __future__ import annotations

import logging

import pandas as pd

from trading_bot.algorithms.base import BaseStrategy
from trading_bot.config import TradingConfig
from trading_bot.models import Signal, TradeSignal

logger = logging.getLogger(__name__)


class MACDStrategy(BaseStrategy):
    """MACD-based trading strategy.

    Generates BUY signals when MACD crosses above the signal line,
    and SELL signals when MACD crosses below the signal line.
    """

    name = "MACD"

    def __init__(self, config: TradingConfig) -> None:
        super().__init__(config)
        self.fast_period = config.macd_fast
        self.slow_period = config.macd_slow
        self.signal_period = config.macd_signal

    def analyze(self, ticker: str, data: pd.DataFrame) -> TradeSignal:
        """Analyze using MACD."""
        min_periods = self.slow_period + self.signal_period
        if len(data) < min_periods:
            return self._make_signal(
                ticker=ticker,
                signal=Signal.HOLD,
                strength=0.0,
                price=float(data["close"].iloc[-1]) if len(data) > 0 else 0.0,
                reason=f"Insufficient data (need {min_periods} periods)",
            )

        data = data.copy()

        # Calculate MACD components
        ema_fast = data["close"].ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = data["close"].ewm(span=self.slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        current_price = float(data["close"].iloc[-1])
        current_macd = float(macd_line.iloc[-1])
        current_signal = float(signal_line.iloc[-1])
        current_hist = float(histogram.iloc[-1])
        prev_macd = float(macd_line.iloc[-2])
        prev_signal = float(signal_line.iloc[-2])
        prev_hist = float(histogram.iloc[-2])

        # Detect crossover
        cross_above = prev_macd <= prev_signal and current_macd > current_signal
        cross_below = prev_macd >= prev_signal and current_macd < current_signal

        # Calculate strength based on histogram magnitude relative to price
        hist_strength = abs(current_hist) / current_price * 100
        strength = min(hist_strength * 5, 1.0)

        # Momentum check: is histogram growing or shrinking?
        hist_growing = abs(current_hist) > abs(prev_hist)

        if cross_above:
            # Bullish crossover
            if hist_growing:
                strength = min(strength + 0.2, 1.0)
            return self._make_signal(
                ticker=ticker,
                signal=Signal.BUY,
                strength=strength,
                price=current_price,
                reason=(
                    f"MACD bullish crossover: MACD ({current_macd:.4f}) crossed above "
                    f"Signal ({current_signal:.4f}), Histogram: {current_hist:.4f}"
                ),
            )
        elif cross_below:
            # Bearish crossover
            if hist_growing:
                strength = min(strength + 0.2, 1.0)
            return self._make_signal(
                ticker=ticker,
                signal=Signal.SELL,
                strength=strength,
                price=current_price,
                reason=(
                    f"MACD bearish crossover: MACD ({current_macd:.4f}) crossed below "
                    f"Signal ({current_signal:.4f}), Histogram: {current_hist:.4f}"
                ),
            )
        else:
            # No crossover
            if current_macd > current_signal:
                trend = "bullish"
            else:
                trend = "bearish"

            momentum = "strengthening" if hist_growing else "weakening"

            return self._make_signal(
                ticker=ticker,
                signal=Signal.HOLD,
                strength=strength * 0.3,
                price=current_price,
                reason=(
                    f"MACD {trend} ({momentum}): "
                    f"MACD={current_macd:.4f}, Signal={current_signal:.4f}"
                ),
            )
