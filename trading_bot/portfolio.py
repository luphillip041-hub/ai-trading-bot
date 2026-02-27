"""Portfolio management and tracking."""

from __future__ import annotations

import logging
from datetime import datetime

from trading_bot.config import TradingConfig
from trading_bot.models import PortfolioSnapshot, Position, Signal, Trade

logger = logging.getLogger(__name__)


class Portfolio:
    """Manages portfolio positions and tracks P&L."""

    def __init__(self, config: TradingConfig) -> None:
        self.config = config
        self.cash: float = config.initial_capital
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.snapshots: list[PortfolioSnapshot] = []

    @property
    def total_value(self) -> float:
        """Total portfolio value (cash + positions)."""
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.cash + positions_value

    @property
    def total_pnl(self) -> float:
        """Total profit/loss."""
        return self.total_value - self.config.initial_capital

    @property
    def total_pnl_pct(self) -> float:
        """Total profit/loss percentage."""
        if self.config.initial_capital == 0:
            return 0.0
        return (self.total_pnl / self.config.initial_capital) * 100

    def can_open_position(self) -> bool:
        """Check if we can open a new position."""
        return len(self.positions) < self.config.max_positions

    def calculate_position_size(self, price: float) -> float:
        """Calculate the number of shares to buy based on position sizing rules.

        Args:
            price: Current price per share.

        Returns:
            Number of shares to buy.
        """
        allocation = self.total_value * self.config.position_size_pct
        allocation = min(allocation, self.cash)  # Can't spend more than we have

        if price <= 0:
            return 0.0

        shares = allocation / price
        return round(shares, 4)

    def open_position(
        self,
        ticker: str,
        price: float,
        timestamp: datetime | None = None,
    ) -> Trade | None:
        """Open a new position.

        Args:
            ticker: Stock ticker symbol.
            price: Entry price.
            timestamp: Trade timestamp.

        Returns:
            Trade record or None if position couldn't be opened.
        """
        if ticker in self.positions:
            logger.warning("Already have a position in %s", ticker)
            return None

        if not self.can_open_position():
            logger.warning("Maximum positions reached (%d)", self.config.max_positions)
            return None

        shares = self.calculate_position_size(price)
        if shares <= 0:
            logger.warning("Position size too small for %s at %.2f", ticker, price)
            return None

        cost = shares * price
        if cost > self.cash:
            logger.warning(
                "Insufficient cash for %s: need %.2f, have %.2f",
                ticker, cost, self.cash,
            )
            return None

        timestamp = timestamp or datetime.now()

        position = Position(
            ticker=ticker,
            shares=shares,
            entry_price=price,
            entry_date=timestamp,
            current_price=price,
            stop_loss=price * (1 - self.config.stop_loss_pct),
            take_profit=price * (1 + self.config.take_profit_pct),
        )

        self.cash -= cost
        self.positions[ticker] = position

        trade = Trade(
            ticker=ticker,
            side=Signal.BUY,
            shares=shares,
            price=price,
            timestamp=timestamp,
        )
        self.trades.append(trade)

        logger.info(
            "Opened position: %s - %s shares @ %.2f (cost: %.2f)",
            ticker,
            shares,
            price,
            cost,
        )
        return trade

    def close_position(
        self,
        ticker: str,
        price: float,
        timestamp: datetime | None = None,
    ) -> Trade | None:
        """Close an existing position.

        Args:
            ticker: Stock ticker symbol.
            price: Exit price.
            timestamp: Trade timestamp.

        Returns:
            Trade record or None if no position exists.
        """
        if ticker not in self.positions:
            logger.warning("No position in %s to close", ticker)
            return None

        position = self.positions[ticker]
        timestamp = timestamp or datetime.now()

        proceeds = position.shares * price
        pnl = proceeds - position.cost_basis
        pnl_pct = (pnl / position.cost_basis) * 100 if position.cost_basis > 0 else 0.0

        self.cash += proceeds
        del self.positions[ticker]

        trade = Trade(
            ticker=ticker,
            side=Signal.SELL,
            shares=position.shares,
            price=price,
            timestamp=timestamp,
            pnl=pnl,
            pnl_pct=pnl_pct,
        )
        self.trades.append(trade)

        logger.info(
            "Closed position: %s - %s shares @ %.2f (P&L: %.2f / %.1f%%)",
            ticker,
            position.shares,
            price,
            pnl,
            pnl_pct,
        )
        return trade

    def update_prices(self, prices: dict[str, float]) -> list[str]:
        """Update current prices for all positions and check stop-loss/take-profit.

        Args:
            prices: Dictionary mapping tickers to current prices.

        Returns:
            List of tickers that hit stop-loss or take-profit.
        """
        triggered: list[str] = []

        for ticker, position in self.positions.items():
            if ticker in prices:
                position.current_price = prices[ticker]

                # Check stop-loss
                if position.current_price <= position.stop_loss:
                    logger.warning(
                        "STOP LOSS triggered for %s: price %.2f <= stop %.2f",
                        ticker,
                        position.current_price,
                        position.stop_loss,
                    )
                    triggered.append(ticker)

                # Check take-profit
                elif position.current_price >= position.take_profit:
                    logger.info(
                        "TAKE PROFIT triggered for %s: price %.2f >= target %.2f",
                        ticker,
                        position.current_price,
                        position.take_profit,
                    )
                    triggered.append(ticker)

        return triggered

    def take_snapshot(self) -> PortfolioSnapshot:
        """Take a snapshot of the current portfolio state."""
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            cash=self.cash,
            positions=list(self.positions.values()),
            total_value=self.total_value,
            total_pnl=self.total_pnl,
            total_pnl_pct=self.total_pnl_pct,
            trades=list(self.trades),
        )
        self.snapshots.append(snapshot)
        return snapshot

    def get_summary(self) -> dict:
        """Get a summary of the portfolio state."""
        return {
            "cash": round(self.cash, 2),
            "positions_count": len(self.positions),
            "total_value": round(self.total_value, 2),
            "total_pnl": round(self.total_pnl, 2),
            "total_pnl_pct": round(self.total_pnl_pct, 2),
            "total_trades": len(self.trades),
            "winning_trades": sum(1 for t in self.trades if t.pnl > 0),
            "losing_trades": sum(1 for t in self.trades if t.pnl < 0),
        }
