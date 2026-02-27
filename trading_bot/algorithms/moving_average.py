"""Moving Average Crossover strategy."""

from __future__ import annotations

import logging

import pandas as pd

from trading_bot.algorithms.base import BaseStrategy
from trading_bot.config import TradingConfig
from trading_bot.models import Signal, TradeSignal

logger = logging.getLogger(__name__)


class MovingAverageCrossoverStrategy(BaseStrategy):
    """Moving Average Crossover trading strategy.

    Generates BUY signals when the short-term MA crosses above the long-term MA,
    and SELL signals when the short-term MA crosses below the long-term MA.
    """

    name = "MA Crossover"

    def __init__(self, config: TradingConfig) -> None:
        super().__init__(config)
        self.short_window = config.ma_short_window
        self.long_window = config.ma_long_window

    def analyze(self, ticker: str, data: pd.DataFrame) -> TradeSignal:
        """Analyze using moving average crossover."""
        if len(data) < self.long_window + 1:
            return self._make_signal(
                ticker=ticker,
                signal=Signal.HOLD,
                strength=0.0,
                price=float(data["close"].iloc[-1]) if len(data) > 0 else 0.0,
                reason=f"Insufficient data (need {self.long_window + 1} periods)",
            )

        # Calculate moving averages
        data = data.copy()
        data["sma_short"] = data["close"].rolling(window=self.short_window).mean()
        data["sma_long"] = data["close"].rolling(window=self.long_window).mean()

        current_price = float(data["close"].iloc[-1])
        sma_short = float(data["sma_short"].iloc[-1])
        sma_long = float(data["sma_long"].iloc[-1])
        prev_sma_short = float(data["sma_short"].iloc[-2])
        prev_sma_long = float(data["sma_long"].iloc[-2])

        # Detect crossover
        cross_above = prev_sma_short <= prev_sma_long and sma_short > sma_long
        cross_below = prev_sma_short >= prev_sma_long and sma_short < sma_long

        # Calculate strength based on distance between MAs
        ma_spread = abs(sma_short - sma_long) / sma_long
        strength = min(ma_spread * 10, 1.0)  # Normalize to 0-1

        if cross_above:
            return self._make_signal(
                ticker=ticker,
                signal=Signal.BUY,
                strength=strength,
                price=current_price,
                reason=(
                    f"SMA{self.short_window} ({sma_short:.2f}) crossed above "
                    f"SMA{self.long_window} ({sma_long:.2f})"
                ),
            )
        elif cross_below:
            return self._make_signal(
                ticker=ticker,
                signal=Signal.SELL,
                strength=strength,
                price=current_price,
                reason=(
                    f"SMA{self.short_window} ({sma_short:.2f}) crossed below "
                    f"SMA{self.long_window} ({sma_long:.2f})"
                ),
            )
        else:
            # No crossover, but indicate trend direction
            if sma_short > sma_long:
                reason = (
                    f"Bullish: SMA{self.short_window} ({sma_short:.2f}) > "
                    f"SMA{self.long_window} ({sma_long:.2f})"
                )
            else:
                reason = (
                    f"Bearish: SMA{self.short_window} ({sma_short:.2f}) < "
                    f"SMA{self.long_window} ({sma_long:.2f})"
                )

            return self._make_signal(
                ticker=ticker,
                signal=Signal.HOLD,
                strength=strength * 0.3,
                price=current_price,
                reason=reason,
            )
