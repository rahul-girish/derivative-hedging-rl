from __future__ import annotations

import matplotlib.pyplot as plt

from hedging.evaluation.backtester import BacktestResult


def plot_stock_price(result: BacktestResult) -> None:
    """
    Plot the simulated stock price.
    """

    plt.figure(figsize=(10, 5))
    plt.plot(result.prices)
    plt.title("Stock Price")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_delta(result: BacktestResult) -> None:
    """
    Plot option delta over time.
    """

    plt.figure(figsize=(10, 5))
    plt.plot(result.deltas)
    plt.title("Delta")
    plt.xlabel("Time Step")
    plt.ylabel("Delta")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_shares(result: BacktestResult) -> None:
    """
    Plot hedge position (shares held).
    """

    plt.figure(figsize=(10, 5))
    plt.plot(result.share_history)
    plt.title("Shares Held")
    plt.xlabel("Time Step")
    plt.ylabel("Shares")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_portfolio_value(result: BacktestResult) -> None:
    """
    Plot portfolio value through time.
    """

    plt.figure(figsize=(10, 5))
    plt.plot(result.portfolio_values)
    plt.title("Portfolio Value")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_transaction_costs(result: BacktestResult) -> None:
    """
    Plot cumulative transaction costs.
    """

    cumulative = []
    total = 0.0

    for cost in result.transaction_costs:
        total += cost
        cumulative.append(total)

    plt.figure(figsize=(10, 5))
    plt.plot(cumulative)
    plt.title("Cumulative Transaction Costs")
    plt.xlabel("Time Step")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_summary(result: BacktestResult) -> None:
    """
    Generate all plots.
    """

    plot_stock_price(result)
    plot_delta(result)
    plot_shares(result)
    plot_portfolio_value(result)
    plot_transaction_costs(result)