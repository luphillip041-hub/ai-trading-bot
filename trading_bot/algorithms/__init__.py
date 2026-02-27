"""Trading algorithms for signal generation."""

from trading_bot.algorithms.bollinger_bands import BollingerBandsStrategy
from trading_bot.algorithms.macd import MACDStrategy
from trading_bot.algorithms.moving_average import MovingAverageCrossoverStrategy
from trading_bot.algorithms.rsi import RSIStrategy

__all__ = [
    "MovingAverageCrossoverStrategy",
    "RSIStrategy",
    "MACDStrategy",
    "BollingerBandsStrategy",
]
