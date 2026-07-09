from __future__ import annotations

import numpy as np
import yfinance as yf

from hedging.simulation.market_path import MarketPath


def load_yahoo_prices(
    ticker: str,
    start: str,
    end: str,
    dt: float = 1 / 252,
) -> MarketPath:
    """
    Download real daily close prices from Yahoo Finance and wrap them
    in a MarketPath so they can be fed into the same Backtester used
    for synthetic (GBM) paths.

    Parameters
    ----------
    ticker:
        Yahoo Finance ticker symbol, e.g. "SPY".
    start, end:
        Date strings understood by yfinance, e.g. "2023-01-01".
    dt:
        Time step in years associated with one row (daily = 1/252).
    """
    data = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
    )

    if data.empty:
        raise ValueError(
            f"No data returned for ticker={ticker!r} between {start} and {end}."
        )

    closes = data["Close"].to_numpy().reshape(-1)

    n_steps = len(closes) - 1
    time_grid = np.arange(n_steps + 1) * dt

    return MarketPath(
        prices=closes.reshape(1, -1),
        dt=dt,
        time_grid=time_grid,
    )
