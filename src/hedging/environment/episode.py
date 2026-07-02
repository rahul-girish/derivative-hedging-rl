from __future__ import annotations

from dataclasses import dataclass

from hedging.pricing.black_scholes import black_scholes_price
from hedging.pricing.greeks import delta
from hedging.pricing.option import OptionContract
from hedging.simulation.execution import ExecutionEngine
from hedging.simulation.market_path import MarketPath
from hedging.simulation.portfolio import Portfolio


@dataclass
class EpisodeStep:

    stock_price: float
    option_price: float
    delta: float
    portfolio_value: float
    transaction_cost: float
    done: bool


class HedgingEpisode:

    def __init__(
        self,
        path: MarketPath,
        contract: OptionContract,
        execution_engine: ExecutionEngine,
        volatility: float,
        rate: float = 0.0,
    ):

        self.path = path
        self.contract = contract
        self.execution_engine = execution_engine
        self.volatility = volatility
        self.rate = rate

        self.portfolio = Portfolio()

        initial_price = black_scholes_price(
            contract,
            path.prices[0, 0],
            volatility,
            rate,
        )

        self.portfolio.initialize_option_sale(initial_price)

        self.current_step = 0

    def step(
        self,
        target_position: float,
    ) -> EpisodeStep:

        stock_price = self.path.prices[
            0,
            self.current_step,
        ]

        remaining = max(
            self.contract.maturity
            - self.current_step * self.path.dt,
            1e-8,
        )

        option = OptionContract(
            strike=self.contract.strike,
            maturity=remaining,
            option_type=self.contract.option_type,
        )

        option_price = black_scholes_price(
            option,
            stock_price,
            self.volatility,
            self.rate,
        )

        option_delta = delta(
            option,
            stock_price,
            self.volatility,
            self.rate,
        )

        cost = self.execution_engine.rebalance(
            self.portfolio,
            target_position,
            stock_price,
        )

        portfolio_value = self.portfolio.total_value(
            stock_price,
            option_price,
        )

        self.current_step += 1

        done = (
            self.current_step
            >= self.path.prices.shape[1]
        )

        return EpisodeStep(
            stock_price=stock_price,
            option_price=option_price,
            delta=option_delta,
            portfolio_value=portfolio_value,
            transaction_cost=cost,
            done=done,
        )