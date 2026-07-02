from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hedging.pricing.black_scholes import black_scholes_price
from hedging.simulation.execution import ExecutionEngine
from hedging.simulation.market_path import MarketPath
from hedging.simulation.portfolio import Portfolio


@dataclass(slots=True)
class BacktestResult:
    prices: np.ndarray
    portfolio_values: list[float] = field(default_factory=list)
    cash_history: list[float] = field(default_factory=list)
    share_history: list[float] = field(default_factory=list)
    option_values: list[float] = field(default_factory=list)
    deltas: list[float] = field(default_factory=list)
    transaction_costs: list[float] = field(default_factory=list)

    @property
    def pnl(self) -> float:
        return self.portfolio_values[-1] - self.portfolio_values[0]


class Backtester:

    def __init__(self, execution_engine: ExecutionEngine):
        self.execution_engine = execution_engine

    def run(
        self,
        path: MarketPath,
        contract,
        hedger,
        volatility: float,
        rate: float = 0.0,
    ) -> BacktestResult:

        prices = path.prices[0]

        portfolio = Portfolio()

        initial_option_price = black_scholes_price(
            contract,
            prices[0],
            volatility,
            rate,
        )

        portfolio.initialize_option_sale(initial_option_price)

        result = BacktestResult(prices=path.prices[0])

        for step, stock_price in enumerate(prices):

            remaining = max(
                contract.maturity - step * path.dt,
                1e-8,
            )

            option = type(contract)(
                strike=contract.strike,
                maturity=remaining,
                option_type=contract.option_type,
            )

            option_value = black_scholes_price(
                option,
                stock_price,
                volatility,
                rate,
            )

            target = hedger.target_position(
                option,
                stock_price,
                volatility,
                rate,
            )

            cost = self.execution_engine.rebalance(
                portfolio,
                target,
                stock_price,
            )

            value = portfolio.total_value(
                stock_price,
                option_value,
            )

            result.portfolio_values.append(value)
            result.cash_history.append(portfolio.cash)
            result.share_history.append(portfolio.shares)
            result.option_values.append(option_value)
            result.deltas.append(target)
            result.transaction_costs.append(cost)

        return result