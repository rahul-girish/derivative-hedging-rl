from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hedging.pricing.black_scholes import black_scholes_price
from hedging.simulation.execution import ExecutionEngine
from hedging.simulation.market_path import MarketPath
from hedging.simulation.portfolio import Portfolio
from hedging.pricing.option import OptionContract


@dataclass(slots=True)
class BacktestResult:
    """
    Stores the complete results of a hedging simulation.
    """

    prices: np.ndarray

    portfolio_values: list[float] = field(default_factory=list)
    hedge_values: list[float] = field(default_factory=list)

    cash_history: list[float] = field(default_factory=list)
    share_history: list[float] = field(default_factory=list)

    option_values: list[float] = field(default_factory=list)

    deltas: list[float] = field(default_factory=list)

    transaction_costs: list[float] = field(default_factory=list)

    hedging_errors: list[float] = field(default_factory=list)

    @property
    def pnl(self) -> float:
        """
        Final portfolio profit/loss.
        """
        return self.portfolio_values[-1] - self.portfolio_values[0]


class Backtester:
    """
    Runs a complete hedging backtest over one simulated price path.
    """

    def __init__(
        self,
        execution_engine: ExecutionEngine,
    ) -> None:

        self.execution_engine = execution_engine

    def run(
        self,
        path: MarketPath,
        contract: OptionContract,
        hedger,
        volatility: float,
        rate: float = 0.0,
    ) -> BacktestResult:

        prices = path.prices[0]

        portfolio = Portfolio()

        result = BacktestResult(prices=prices)

        # ---------------------------------------------------
        # Sell one option initially and receive the premium
        # ---------------------------------------------------

        initial_option_price = black_scholes_price(
            contract=contract,
            spot=prices[0],
            volatility=volatility,
            rate=rate,
        )

        portfolio.initialize_option_sale(initial_option_price)

        # ---------------------------------------------------
        # Simulate every timestep
        # ---------------------------------------------------

        for step, stock_price in enumerate(prices):

            remaining_maturity = max(
                contract.maturity - step * path.dt,
                1e-8,
            )

            current_contract = OptionContract(
                strike=contract.strike,
                maturity=remaining_maturity,
                option_type=contract.option_type,
            )

            option_value = black_scholes_price(
                contract=current_contract,
                spot=stock_price,
                volatility=volatility,
                rate=rate,
            )

            target_position = hedger.target_position(
                contract=current_contract,
                stock_price=stock_price,
                volatility=volatility,
                rate=rate,
            )

            transaction_cost = self.execution_engine.rebalance(
                portfolio=portfolio,
                target_position=target_position,
                stock_price=stock_price,
            )

            hedge_value = portfolio.hedge_value(
                stock_price=stock_price,
            )

            total_value = portfolio.total_value(
                stock_price=stock_price,
                option_price=option_value,
            )

            hedging_error = total_value

            # -------------------------
            # Save everything
            # -------------------------

            result.portfolio_values.append(total_value)

            result.hedge_values.append(hedge_value)

            result.cash_history.append(
                portfolio.cash
            )

            result.share_history.append(
                portfolio.shares
            )

            result.option_values.append(
                option_value
            )

            result.deltas.append(
                target_position
            )

            result.transaction_costs.append(
                transaction_cost
            )

            result.hedging_errors.append(
                hedging_error
            )

        return result