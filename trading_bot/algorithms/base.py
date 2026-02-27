"""Base class for trading algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

from trading_bot.config import TradingConfig
from trading_bot.models import Signal, TradeSignal


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""

    name: str = "base"

    def __init__(self, config: TradingConfig) -> None:
        self.config = config

    @abstractmethod
    def analyze(self, ticker: str, data: pd.DataFrame) -> TradeSignal:
        """Analyze data and generate a trading signal.

        Args:
            ticker: Stock ticker symbol.
            data: DataFrame with OHLCV data.

        Returns:
            A TradeSignal with the recommendation.
        """
        ...

    def _make_signal(
        self,
        ticker: str,
        signal: Signal,
        strength: float,
        price: float,
        reason: str,
    ) -> TradeSignal:
        """Helper to create a TradeSignal."""
        return TradeSignal(
            ticker=ticker,
            signal=signal,
            algorithm=self.name,
            strength=min(max(strength, 0.0), 1.0),
            price=price,
            timestamp=datetime.now(),
            reason=reason,
        )
