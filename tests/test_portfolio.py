"""Tests for portfolio management."""

from __future__ import annotations

from trading_bot.config import TradingConfig
from trading_bot.models import Signal
from trading_bot.portfolio import Portfolio


class TestPortfolio:
    """Tests for the Portfolio class."""

    def setup_method(self) -> None:
        self.config = TradingConfig(
            initial_capital=100000.0,
            position_size_pct=0.1,
            max_positions=5,
            stop_loss_pct=0.05,
            take_profit_pct=0.15,
        )
        self.portfolio = Portfolio(self.config)

    def test_initial_state(self) -> None:
        assert self.portfolio.cash == 100000.0
        assert len(self.portfolio.positions) == 0
        assert self.portfolio.total_value == 100000.0
        assert self.portfolio.total_pnl == 0.0

    def test_open_position(self) -> None:
        trade = self.portfolio.open_position("AAPL", 150.0)
        assert trade is not None
        assert trade.ticker == "AAPL"
        assert trade.side == Signal.BUY
        assert trade.price == 150.0
        assert "AAPL" in self.portfolio.positions

    def test_position_sizing(self) -> None:
        trade = self.portfolio.open_position("AAPL", 150.0)
        assert trade is not None
        # 10% of 100000 = 10000, at $150 per share ≈ 66.6667 shares
        expected_shares = round(10000.0 / 150.0, 4)
        assert trade.shares == expected_shares

    def test_close_position_profit(self) -> None:
        self.portfolio.open_position("AAPL", 100.0)
        trade = self.portfolio.close_position("AAPL", 120.0)
        assert trade is not None
        assert trade.pnl > 0
        assert trade.pnl_pct > 0
        assert "AAPL" not in self.portfolio.positions

    def test_close_position_loss(self) -> None:
        self.portfolio.open_position("AAPL", 100.0)
        trade = self.portfolio.close_position("AAPL", 80.0)
        assert trade is not None
        assert trade.pnl < 0
        assert trade.pnl_pct < 0

    def test_max_positions(self) -> None:
        for i, ticker in enumerate(["A", "B", "C", "D", "E"]):
            result = self.portfolio.open_position(ticker, 50.0)
            assert result is not None

        # 6th position should fail
        result = self.portfolio.open_position("F", 50.0)
        assert result is None

    def test_duplicate_position(self) -> None:
        self.portfolio.open_position("AAPL", 100.0)
        result = self.portfolio.open_position("AAPL", 110.0)
        assert result is None

    def test_close_nonexistent_position(self) -> None:
        result = self.portfolio.close_position("AAPL", 100.0)
        assert result is None

    def test_stop_loss(self) -> None:
        self.portfolio.open_position("AAPL", 100.0)
        position = self.portfolio.positions["AAPL"]
        assert position.stop_loss == 95.0  # 5% below entry

    def test_take_profit(self) -> None:
        self.portfolio.open_position("AAPL", 100.0)
        position = self.portfolio.positions["AAPL"]
        assert abs(position.take_profit - 115.0) < 1e-10  # 15% above entry

    def test_update_prices_stop_loss(self) -> None:
        self.portfolio.open_position("AAPL", 100.0)
        triggered = self.portfolio.update_prices({"AAPL": 94.0})
        assert "AAPL" in triggered

    def test_update_prices_take_profit(self) -> None:
        self.portfolio.open_position("AAPL", 100.0)
        triggered = self.portfolio.update_prices({"AAPL": 116.0})
        assert "AAPL" in triggered

    def test_snapshot(self) -> None:
        self.portfolio.open_position("AAPL", 100.0)
        snapshot = self.portfolio.take_snapshot()
        assert snapshot.total_value > 0
        assert len(snapshot.positions) == 1

    def test_get_summary(self) -> None:
        self.portfolio.open_position("AAPL", 100.0)
        summary = self.portfolio.get_summary()
        assert "cash" in summary
        assert "positions_count" in summary
        assert "total_value" in summary
        assert summary["positions_count"] == 1

    def test_total_value_with_position(self) -> None:
        self.portfolio.open_position("AAPL", 100.0)
        position = self.portfolio.positions["AAPL"]
        position.current_price = 110.0
        # Cash should be reduced, but total value should reflect position value
        assert self.portfolio.total_value > 0
        assert self.portfolio.cash < 100000.0
