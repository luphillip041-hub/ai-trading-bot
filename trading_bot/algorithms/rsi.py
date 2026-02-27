"""RSI (Relative Strength Index) strategy."""

from __future__ import annotations

import logging

import pandas as pd

from trading_bot.algorithms.base import BaseStrategy
from trading_bot.config import TradingConfig
from trading_bot.models import Signal, TradeSignal

logger = logging.getLogger(__name__)


class RSIStrategy(BaseStrategy):
    """RSI-based trading strategy.

    Generates BUY signals when RSI is in oversold territory,
    and SELL signals when RSI is in overbought territory.
    """

    name = "RSI"

    def __init__(self, config: TradingConfig) -> None:
        super().__init__(config)
        self.period = config.rsi_period
        self.overbought = config.rsi_overbought
        self.oversold = config.rsi_oversold

    def _calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI for a price series."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=self.period, min_periods=self.period).mean()
        avg_loss = loss.rolling(window=self.period, min_periods=self.period).mean()

        # Use exponential smoothing after initial SMA
        for i in range(self.period, len(avg_gain)):
            prev_gain = avg_gain.iloc[i - 1]
            avg_gain.iloc[i] = (prev_gain * (self.period - 1) + gain.iloc[i]) / self.period
            prev_loss = avg_loss.iloc[i - 1]
            avg_loss.iloc[i] = (prev_loss * (self.period - 1) + loss.iloc[i]) / self.period

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def analyze(self, ticker: str, data: pd.DataFrame) -> TradeSignal:
        """Analyze using RSI."""
        min_periods = self.period + 1
        if len(data) < min_periods:
            return self._make_signal(
                ticker=ticker,
                signal=Signal.HOLD,
                strength=0.0,
                price=float(data["close"].iloc[-1]) if len(data) > 0 else 0.0,
                reason=f"Insufficient data (need {min_periods} periods)",
            )

        rsi = self._calculate_rsi(data["close"])
        current_rsi = float(rsi.iloc[-1])
        prev_rsi = float(rsi.iloc[-2])
        current_price = float(data["close"].iloc[-1])

        # Calculate signal strength based on how extreme the RSI is
        if current_rsi <= self.oversold:
            # Oversold - BUY signal
            strength = (self.oversold - current_rsi) / self.oversold
            # Stronger signal if RSI is recovering from oversold
            recovering = current_rsi > prev_rsi
            if recovering:
                strength = min(strength + 0.2, 1.0)

            return self._make_signal(
                ticker=ticker,
                signal=Signal.BUY,
                strength=strength,
                price=current_price,
                reason=f"RSI oversold at {current_rsi:.1f} (threshold: {self.oversold})"
                + (" [recovering]" if recovering else ""),
            )
        elif current_rsi >= self.overbought:
            # Overbought - SELL signal
            strength = (current_rsi - self.overbought) / (100 - self.overbought)
            # Stronger signal if RSI is declining from overbought
            declining = current_rsi < prev_rsi
            if declining:
                strength = min(strength + 0.2, 1.0)

            return self._make_signal(
                ticker=ticker,
                signal=Signal.SELL,
                strength=strength,
                price=current_price,
                reason=f"RSI overbought at {current_rsi:.1f} (threshold: {self.overbought})"
                + (" [declining]" if declining else ""),
            )
        else:
            # Neutral zone
            # Slight bullish/bearish bias based on position within range
            mid = (self.overbought + self.oversold) / 2
            strength = abs(current_rsi - mid) / (mid - self.oversold)
            strength = min(strength * 0.3, 0.3)

            return self._make_signal(
                ticker=ticker,
                signal=Signal.HOLD,
                strength=strength,
                price=current_price,
                reason=(
                    f"RSI neutral at {current_rsi:.1f} "
                    f"(range: {self.oversold}-{self.overbought})"
                ),
            )
