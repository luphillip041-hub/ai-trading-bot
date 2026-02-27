"""Bollinger Bands strategy."""

from __future__ import annotations

import logging

import pandas as pd

from trading_bot.algorithms.base import BaseStrategy
from trading_bot.config import TradingConfig
from trading_bot.models import Signal, TradeSignal

logger = logging.getLogger(__name__)


class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands trading strategy.

    Generates BUY signals when price touches or crosses below the lower band,
    and SELL signals when price touches or crosses above the upper band.
    """

    name = "Bollinger Bands"

    def __init__(self, config: TradingConfig) -> None:
        super().__init__(config)
        self.window = config.bb_window
        self.std_dev = config.bb_std_dev

    def analyze(self, ticker: str, data: pd.DataFrame) -> TradeSignal:
        """Analyze using Bollinger Bands."""
        if len(data) < self.window:
            return self._make_signal(
                ticker=ticker,
                signal=Signal.HOLD,
                strength=0.0,
                price=float(data["close"].iloc[-1]) if len(data) > 0 else 0.0,
                reason=f"Insufficient data (need {self.window} periods)",
            )

        data = data.copy()

        # Calculate Bollinger Bands
        sma = data["close"].rolling(window=self.window).mean()
        std = data["close"].rolling(window=self.window).std()
        upper_band = sma + (std * self.std_dev)
        lower_band = sma - (std * self.std_dev)

        current_price = float(data["close"].iloc[-1])
        current_upper = float(upper_band.iloc[-1])
        current_lower = float(lower_band.iloc[-1])
        current_sma = float(sma.iloc[-1])

        # Calculate %B (position within bands)
        band_width = current_upper - current_lower
        if band_width == 0:
            pct_b = 0.5
        else:
            pct_b = (current_price - current_lower) / band_width

        # Calculate bandwidth (volatility measure)
        bandwidth = band_width / current_sma if current_sma != 0 else 0

        # Price relative to bands
        if current_price <= current_lower:
            # Price at or below lower band - BUY signal
            # Strength increases the further below the band
            overshoot = (current_lower - current_price) / current_lower if current_lower != 0 else 0
            strength = min(0.6 + overshoot * 10, 1.0)

            return self._make_signal(
                ticker=ticker,
                signal=Signal.BUY,
                strength=strength,
                price=current_price,
                reason=(
                    f"Price ({current_price:.2f}) at/below lower band ({current_lower:.2f}), "
                    f"%B={pct_b:.2f}, Bandwidth={bandwidth:.4f}"
                ),
            )
        elif current_price >= current_upper:
            # Price at or above upper band - SELL signal
            overshoot = (current_price - current_upper) / current_upper if current_upper != 0 else 0
            strength = min(0.6 + overshoot * 10, 1.0)

            return self._make_signal(
                ticker=ticker,
                signal=Signal.SELL,
                strength=strength,
                price=current_price,
                reason=(
                    f"Price ({current_price:.2f}) at/above upper band ({current_upper:.2f}), "
                    f"%B={pct_b:.2f}, Bandwidth={bandwidth:.4f}"
                ),
            )
        else:
            # Price within bands
            # Provide directional bias based on position
            if pct_b < 0.2:
                reason = f"Near lower band: %B={pct_b:.2f} (potential bounce)"
                strength = 0.3
            elif pct_b > 0.8:
                reason = f"Near upper band: %B={pct_b:.2f} (potential reversal)"
                strength = 0.3
            else:
                reason = f"Mid-band: %B={pct_b:.2f}, SMA={current_sma:.2f}"
                strength = 0.1

            return self._make_signal(
                ticker=ticker,
                signal=Signal.HOLD,
                strength=strength,
                price=current_price,
                reason=reason,
            )
