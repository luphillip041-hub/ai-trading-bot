"""Yahoo Finance data fetcher module."""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import yfinance as yf

from trading_bot.config import TradingConfig

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches market data from Yahoo Finance."""

    def __init__(self, config: TradingConfig) -> None:
        self.config = config

    def fetch_ticker_data(
        self,
        ticker: str,
        period: str | None = None,
        interval: str | None = None,
    ) -> pd.DataFrame:
        """Fetch historical data for a single ticker.

        Args:
            ticker: Stock ticker symbol.
            period: Data period (overrides config).
            interval: Data interval (overrides config).

        Returns:
            DataFrame with OHLCV data.
        """
        period = period or self.config.data_period
        interval = interval or self.config.data_interval

        logger.info("Fetching data for %s (period=%s, interval=%s)", ticker, period, interval)

        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)

            if df.empty:
                logger.warning("No data returned for %s", ticker)
                return pd.DataFrame()

            # Ensure consistent column names
            df.columns = [col.lower().replace(" ", "_") for col in df.columns]
            df.index.name = "date"

            logger.info("Fetched %d rows for %s", len(df), ticker)
            return df

        except Exception:
            logger.exception("Error fetching data for %s", ticker)
            return pd.DataFrame()

    def fetch_watchlist_data(
        self,
        tickers: list[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch data for all tickers in the watchlist.

        Args:
            tickers: Optional list of tickers (overrides config watchlist).

        Returns:
            Dictionary mapping ticker symbols to DataFrames.
        """
        tickers = tickers or self.config.watchlist
        data: dict[str, pd.DataFrame] = {}

        for ticker in tickers:
            df = self.fetch_ticker_data(ticker)
            if not df.empty:
                data[ticker] = df

        logger.info("Fetched data for %d/%d tickers", len(data), len(tickers))
        return data

    def get_current_price(self, ticker: str) -> float | None:
        """Get the current price for a ticker.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Current price or None if unavailable.
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.fast_info
            return float(info.last_price)
        except Exception:
            logger.exception("Error getting current price for %s", ticker)
            return None

    def get_ticker_info(self, ticker: str) -> dict:
        """Get detailed info for a ticker.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Dictionary with ticker information.
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                "symbol": ticker,
                "name": info.get("longName", "N/A"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "52w_high": info.get("fiftyTwoWeekHigh", 0),
                "52w_low": info.get("fiftyTwoWeekLow", 0),
                "avg_volume": info.get("averageVolume", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "fetched_at": datetime.now().isoformat(),
            }
        except Exception:
            logger.exception("Error getting info for %s", ticker)
            return {"symbol": ticker, "error": "Failed to fetch info"}
