"""Strategy engine that aggregates signals from multiple algorithms."""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from trading_bot.algorithms.base import BaseStrategy
from trading_bot.algorithms.bollinger_bands import BollingerBandsStrategy
from trading_bot.algorithms.macd import MACDStrategy
from trading_bot.algorithms.moving_average import MovingAverageCrossoverStrategy
from trading_bot.algorithms.rsi import RSIStrategy
from trading_bot.config import TradingConfig
from trading_bot.models import AggregatedSignal, Signal, TradeSignal

logger = logging.getLogger(__name__)


class StrategyEngine:
    """Aggregates signals from multiple trading strategies."""

    def __init__(self, config: TradingConfig) -> None:
        self.config = config
        self.strategies: list[BaseStrategy] = [
            MovingAverageCrossoverStrategy(config),
            RSIStrategy(config),
            MACDStrategy(config),
            BollingerBandsStrategy(config),
        ]

    def analyze_ticker(self, ticker: str, data: pd.DataFrame) -> AggregatedSignal:
        """Run all strategies on a ticker and aggregate the signals.

        Args:
            ticker: Stock ticker symbol.
            data: DataFrame with OHLCV data.

        Returns:
            An AggregatedSignal with the consensus recommendation.
        """
        signals: list[TradeSignal] = []

        for strategy in self.strategies:
            try:
                signal = strategy.analyze(ticker, data)
                signals.append(signal)
                logger.debug(
                    "%s - %s: %s (strength: %.2f) - %s",
                    ticker,
                    strategy.name,
                    signal.signal.value,
                    signal.strength,
                    signal.reason,
                )
            except Exception:
                logger.exception("Error running %s on %s", strategy.name, ticker)

        return self._aggregate_signals(ticker, signals)

    def _aggregate_signals(
        self, ticker: str, signals: list[TradeSignal]
    ) -> AggregatedSignal:
        """Aggregate individual signals into a consensus signal.

        Uses a voting system where each algorithm gets one vote.
        The consensus signal is determined by the majority,
        weighted by signal strength.
        """
        if not signals:
            return AggregatedSignal(
                ticker=ticker,
                signal=Signal.HOLD,
                timestamp=datetime.now(),
                reasons=["No signals generated"],
            )

        buy_count = sum(1 for s in signals if s.signal == Signal.BUY)
        sell_count = sum(1 for s in signals if s.signal == Signal.SELL)
        hold_count = sum(1 for s in signals if s.signal == Signal.HOLD)

        # Calculate weighted scores
        buy_strength = sum(s.strength for s in signals if s.signal == Signal.BUY)
        sell_strength = sum(s.strength for s in signals if s.signal == Signal.SELL)

        # Determine consensus
        if buy_count >= self.config.min_consensus and buy_strength > sell_strength:
            consensus = Signal.BUY
        elif sell_count >= self.config.min_consensus and sell_strength > buy_strength:
            consensus = Signal.SELL
        else:
            consensus = Signal.HOLD

        # Average strength of the consensus signals
        consensus_signals = [s for s in signals if s.signal == consensus]
        avg_strength = (
            sum(s.strength for s in consensus_signals) / len(consensus_signals)
            if consensus_signals
            else 0.0
        )

        # Get the latest price from any signal
        price = signals[-1].price if signals else 0.0

        reasons = [f"[{s.algorithm}] {s.signal.value}: {s.reason}" for s in signals]

        return AggregatedSignal(
            ticker=ticker,
            signal=consensus,
            buy_count=buy_count,
            sell_count=sell_count,
            hold_count=hold_count,
            avg_strength=avg_strength,
            price=price,
            timestamp=datetime.now(),
            contributing_signals=signals,
            reasons=reasons,
        )

    def scan_all(
        self, market_data: dict[str, pd.DataFrame]
    ) -> list[AggregatedSignal]:
        """Scan all tickers and return aggregated signals.

        Args:
            market_data: Dictionary mapping ticker symbols to DataFrames.

        Returns:
            List of AggregatedSignals, sorted by strength.
        """
        results: list[AggregatedSignal] = []

        for ticker, data in market_data.items():
            signal = self.analyze_ticker(ticker, data)
            results.append(signal)

        # Sort by signal type (BUY first, then SELL, then HOLD) and strength
        signal_priority = {Signal.BUY: 0, Signal.SELL: 1, Signal.HOLD: 2}
        results.sort(key=lambda s: (signal_priority[s.signal], -s.avg_strength))

        return results
