"""Backtesting engine for evaluating trading strategies."""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from trading_bot.config import TradingConfig
from trading_bot.models import Signal
from trading_bot.portfolio import Portfolio
from trading_bot.strategy_engine import StrategyEngine

logger = logging.getLogger(__name__)


class BacktestResult:
    """Results from a backtest run."""

    def __init__(
        self,
        ticker: str,
        portfolio: Portfolio,
        equity_curve: list[dict],
        signals_log: list[dict],
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        self.ticker = ticker
        self.portfolio = portfolio
        self.equity_curve = equity_curve
        self.signals_log = signals_log
        self.start_date = start_date
        self.end_date = end_date

    @property
    def total_return(self) -> float:
        """Total return percentage."""
        return self.portfolio.total_pnl_pct

    @property
    def total_trades(self) -> int:
        """Total number of trades."""
        return len(self.portfolio.trades)

    @property
    def winning_trades(self) -> int:
        """Number of winning trades."""
        return sum(1 for t in self.portfolio.trades if t.pnl > 0)

    @property
    def losing_trades(self) -> int:
        """Number of losing trades."""
        return sum(1 for t in self.portfolio.trades if t.pnl < 0)

    @property
    def win_rate(self) -> float:
        """Win rate percentage."""
        sell_trades = [t for t in self.portfolio.trades if t.side == Signal.SELL]
        if not sell_trades:
            return 0.0
        winners = sum(1 for t in sell_trades if t.pnl > 0)
        return (winners / len(sell_trades)) * 100

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown percentage."""
        if not self.equity_curve:
            return 0.0

        peak = self.equity_curve[0]["equity"]
        max_dd = 0.0

        for point in self.equity_curve:
            equity = point["equity"]
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd

        return max_dd

    @property
    def sharpe_ratio(self) -> float:
        """Approximate Sharpe ratio (annualized, assuming risk-free rate = 0)."""
        if len(self.equity_curve) < 2:
            return 0.0

        returns = []
        for i in range(1, len(self.equity_curve)):
            prev_eq = self.equity_curve[i - 1]["equity"]
            curr_eq = self.equity_curve[i]["equity"]
            if prev_eq > 0:
                returns.append((curr_eq - prev_eq) / prev_eq)

        if not returns:
            return 0.0

        import numpy as np

        returns_arr = np.array(returns)
        mean_return = np.mean(returns_arr)
        std_return = np.std(returns_arr)

        if std_return == 0:
            return 0.0

        # Annualize (assuming daily data)
        return float(mean_return / std_return * np.sqrt(252))

    def get_summary(self) -> dict:
        """Get a summary of the backtest results."""
        return {
            "ticker": self.ticker,
            "period": (
                f"{self.start_date.strftime('%Y-%m-%d')} to "
                f"{self.end_date.strftime('%Y-%m-%d')}"
            ),
            "initial_capital": self.portfolio.config.initial_capital,
            "final_value": round(self.portfolio.total_value, 2),
            "total_return_pct": round(self.total_return, 2),
            "total_trades": self.total_trades,
            "win_rate_pct": round(self.win_rate, 2),
            "max_drawdown_pct": round(self.max_drawdown, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
        }


class Backtester:
    """Backtesting engine for evaluating trading strategies on historical data."""

    def __init__(self, config: TradingConfig) -> None:
        self.config = config
        self.engine = StrategyEngine(config)

    def run(
        self,
        ticker: str,
        data: pd.DataFrame,
        lookback: int = 60,
    ) -> BacktestResult:
        """Run a backtest on historical data.

        Args:
            ticker: Stock ticker symbol.
            data: DataFrame with OHLCV data.
            lookback: Number of periods of data to use for each analysis window.

        Returns:
            BacktestResult with performance metrics.
        """
        portfolio = Portfolio(self.config)
        equity_curve: list[dict] = []
        signals_log: list[dict] = []

        if len(data) < lookback:
            logger.warning(
                "Insufficient data for backtesting %s: have %d, need %d",
                ticker,
                len(data),
                lookback,
            )
            return BacktestResult(
                ticker=ticker,
                portfolio=portfolio,
                equity_curve=[],
                signals_log=[],
                start_date=datetime.now(),
                end_date=datetime.now(),
            )

        start_date = data.index[lookback]
        end_date = data.index[-1]

        logger.info(
            "Running backtest for %s from %s to %s (%d periods)",
            ticker,
            start_date,
            end_date,
            len(data) - lookback,
        )

        for i in range(lookback, len(data)):
            window = data.iloc[i - lookback : i + 1]
            current_date = data.index[i]
            current_price = float(data["close"].iloc[i])

            # Get aggregated signal
            signal = self.engine.analyze_ticker(ticker, window)

            # Update portfolio prices
            if ticker in portfolio.positions:
                portfolio.positions[ticker].current_price = current_price

                # Check stop-loss and take-profit
                position = portfolio.positions[ticker]
                if current_price <= position.stop_loss:
                    portfolio.close_position(
                        ticker, current_price, timestamp=current_date.to_pydatetime()
                    )
                    signals_log.append({
                        "date": str(current_date),
                        "action": "STOP_LOSS",
                        "price": current_price,
                        "signal_strength": signal.avg_strength,
                    })
                elif current_price >= position.take_profit:
                    portfolio.close_position(
                        ticker, current_price, timestamp=current_date.to_pydatetime()
                    )
                    signals_log.append({
                        "date": str(current_date),
                        "action": "TAKE_PROFIT",
                        "price": current_price,
                        "signal_strength": signal.avg_strength,
                    })

            # Execute signals
            if signal.signal == Signal.BUY and ticker not in portfolio.positions:
                trade = portfolio.open_position(
                    ticker, current_price, timestamp=current_date.to_pydatetime()
                )
                if trade:
                    signals_log.append({
                        "date": str(current_date),
                        "action": "BUY",
                        "price": current_price,
                        "signal_strength": signal.avg_strength,
                        "reasons": signal.reasons,
                    })

            elif signal.signal == Signal.SELL and ticker in portfolio.positions:
                trade = portfolio.close_position(
                    ticker, current_price, timestamp=current_date.to_pydatetime()
                )
                if trade:
                    signals_log.append({
                        "date": str(current_date),
                        "action": "SELL",
                        "price": current_price,
                        "signal_strength": signal.avg_strength,
                        "pnl": trade.pnl,
                        "reasons": signal.reasons,
                    })

            # Record equity curve
            equity_curve.append({
                "date": str(current_date),
                "equity": portfolio.total_value,
                "cash": portfolio.cash,
                "price": current_price,
            })

        # Close any remaining positions at the last price
        if ticker in portfolio.positions:
            final_price = float(data["close"].iloc[-1])
            portfolio.close_position(
                ticker, final_price, timestamp=end_date.to_pydatetime()
            )

        return BacktestResult(
            ticker=ticker,
            portfolio=portfolio,
            equity_curve=equity_curve,
            signals_log=signals_log,
            start_date=start_date.to_pydatetime(),
            end_date=end_date.to_pydatetime(),
        )
