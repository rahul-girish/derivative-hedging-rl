from __future__ import annotations

from hedging.evaluation.backtester import BacktestResult
from hedging.evaluation.metrics import (
    max_drawdown,
    mean_abs_hedging_error,
    realized_volatility,
    rmse,
    total_return,
    total_transaction_cost,
)


class BacktestReport:
    """
    Generates a formatted performance report for a hedging backtest.
    """

    @staticmethod
    def print(result: BacktestResult) -> None:

        print("\n" + "=" * 60)
        print("DELTA HEDGING REPORT")
        print("=" * 60)

        print(f"Initial Option Premium : {result.option_values[0]:>12.4f}")
        print(f"Final Option Value     : {result.option_values[-1]:>12.4f}")

        print("-" * 60)

        print(f"Final Portfolio Value  : {result.portfolio_values[-1]:>12.4f}")
        print(f"Final Hedge Value      : {result.hedge_values[-1]:>12.4f}")
        print(f"Profit / Loss (PnL)    : {result.pnl:>12.4f}")

        print("-" * 60)

        print(
            f"Total Transaction Cost : "
            f"{total_transaction_cost(result.transaction_costs):>12.4f}"
        )

        print(
            f"Mean Hedge Error       : "
            f"{mean_abs_hedging_error(result.hedging_errors):>12.4f}"
        )

        print(
            f"RMSE                  : "
            f"{rmse(result.hedging_errors):>12.4f}"
        )

        print(
            f"Total Return          : "
            f"{total_return(result.portfolio_values):>12.4f}"
        )

        print(
            f"Max Drawdown          : "
            f"{max_drawdown(result.portfolio_values):>12.4f}"
        )

        print(
            f"Realized Volatility   : "
            f"{realized_volatility(result.portfolio_values):>12.4f}"
        )

        print("-" * 60)

        print(f"Final Shares Held      : {result.share_history[-1]:>12.4f}")
        print(f"Final Cash Balance     : {result.cash_history[-1]:>12.4f}")

        print("=" * 60)