from __future__ import annotations

from hedging.simulation.portfolio import Portfolio
from hedging.simulation.transaction_costs import TransactionCostModel


class ExecutionEngine:
    """
    Executes trades and updates the portfolio.
    """

    def __init__(
        self,
        cost_model: TransactionCostModel,
    ) -> None:
        self.cost_model = cost_model

    def rebalance(
        self,
        portfolio: Portfolio,
        target_position: float,
        stock_price: float,
    ) -> float:
        """
        Rebalance the portfolio to the target number of shares.

        Returns
        -------
        float
            Transaction cost incurred.
        """

        trade = target_position - portfolio.shares

        cost = self.cost_model.calculate(
            shares=trade,
            price=stock_price,
        )

        if trade > 0:
            portfolio.buy(trade, stock_price, cost)

        elif trade < 0:
            portfolio.sell(-trade, stock_price, cost)

        return cost