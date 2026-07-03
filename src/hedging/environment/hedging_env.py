from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hedging.environment.reward import RewardFunction
from hedging.environment.state import EnvironmentState
from hedging.pricing.black_scholes import black_scholes_price
from hedging.pricing.greeks import delta
from hedging.pricing.option import OptionContract
from hedging.simulation.execution import ExecutionEngine
from hedging.simulation.gbm import GBMSimulator
from hedging.simulation.portfolio import Portfolio
from hedging.simulation.transaction_costs import TransactionCostModel


class HedgingEnv(gym.Env):
    """
    Gymnasium environment for option hedging using
    continuous actions.

    Observation
    -----------
    [
        stock_price,
        time_remaining,
        option_delta,
        shares_held,
        cash_balance,
    ]

    Action
    ------
    Continuous value in [-1, 1].

    This is scaled internally to

        [-max_position, max_position]

    shares.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        contract: OptionContract,
        volatility: float = 0.20,
        rate: float = 0.05,
        mu: float = 0.05,
        n_steps: int = 252,
        max_position: float = 5.0,
        s0_range: tuple[float, float] = (95.0, 105.0),
        volatility_range: tuple[float, float] = (0.15, 0.25),
        rate_range: tuple[float, float] = (0.01, 0.06),
        strike_range: tuple[float, float] = (90.0, 110.0),
        maturity_range: tuple[float, float] = (0.5, 1.5),
    ) -> None:

        super().__init__()

        # --------------------------------------------------
        # Base parameters
        # --------------------------------------------------

        self.base_contract = contract
        self.base_volatility = volatility
        self.base_rate = rate

        self.contract = contract
        self.volatility = volatility
        self.rate = rate

        self.n_steps = n_steps
        self.max_position = max_position

        # --------------------------------------------------
        # Randomization ranges
        # --------------------------------------------------

        self.s0_range = s0_range
        self.volatility_range = volatility_range
        self.rate_range = rate_range
        self.strike_range = strike_range
        self.maturity_range = maturity_range

        # --------------------------------------------------
        # Market simulator
        # --------------------------------------------------

        self.simulator = GBMSimulator(
            mu=mu,
            sigma=volatility,
        )

        # --------------------------------------------------
        # Execution engine
        # --------------------------------------------------

        self.execution_engine = ExecutionEngine(
            TransactionCostModel(rate=0.001)
        )

        self.reward_function = RewardFunction()

        # --------------------------------------------------
        # Spaces
        # --------------------------------------------------

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(5,),
            dtype=np.float32,
        )

        # --------------------------------------------------
        # Episode state
        # --------------------------------------------------

        self.path = None
        self.portfolio = None
        self.current_step = 0

    def _randomize_market(self) -> tuple[float, float, float]:
        """
        Randomize the market parameters for a new episode.
        """

        initial_price = float(
            self.np_random.uniform(*self.s0_range)
        )

        volatility = float(
            self.np_random.uniform(*self.volatility_range)
        )

        rate = float(
            self.np_random.uniform(*self.rate_range)
        )

        return initial_price, volatility, rate

    def _create_contract(self) -> None:
        """
        Create a randomized option contract.
        """

        strike = float(
            self.np_random.uniform(*self.strike_range)
        )

        maturity = float(
            self.np_random.uniform(*self.maturity_range)
        )

        self.contract = OptionContract(
            strike=strike,
            maturity=maturity,
            option_type=self.base_contract.option_type,
        )

    def _initialize_portfolio(
        self,
        initial_price: float,
    ) -> None:
        """
        Create a fresh portfolio and receive the option premium.
        """

        self.portfolio = Portfolio()

        option_price = black_scholes_price(
            contract=self.contract,
            spot=initial_price,
            volatility=self.volatility,
            rate=self.rate,
        )

        self.portfolio.initialize_option_sale(
            option_price
        )

        self.current_step = 0

    def reset(self, seed=None, options=None):
        """
        Reset the environment and begin a new episode.
        """

        super().reset(seed=seed)

        (
            initial_price,
            self.volatility,
            self.rate,
        ) = self._randomize_market()

        self._create_contract()

        # Update simulator volatility
        self.simulator.sigma = self.volatility

        # Generate a fresh market path
        self.path = self.simulator.simulate(
            s0=initial_price,
            n_steps=self.n_steps,
            n_paths=1,
            seed=seed,
        )

        # Create a fresh portfolio
        self._initialize_portfolio(
            initial_price,
        )

        observation = self._get_observation()

        info = {
            "initial_price": initial_price,
            "strike": self.contract.strike,
            "maturity": self.contract.maturity,
            "volatility": self.volatility,
            "rate": self.rate,
        }

        return observation, info

    def step(self, action):
        """
        Execute one hedging step.
        """

        # Target hedge position
        target_position = (
            float(action[0]) * self.max_position
        )

        stock_price = self.path.prices[
            0,
            self.current_step,
        ]

        remaining = max(
            self.contract.maturity
            - self.current_step * self.path.dt,
            1e-8,
        )

        current_contract = OptionContract(
            strike=self.contract.strike,
            maturity=remaining,
            option_type=self.contract.option_type,
        )

        option_price = black_scholes_price(
            current_contract,
            stock_price,
            self.volatility,
            self.rate,
        )

    # -------------------------------------------------
    # Calculate trade size BEFORE rebalancing
    # -------------------------------------------------

        trade_size = (
            target_position
            - self.portfolio.shares
        )

    # -------------------------------------------------
    # Execute hedge
    # -------------------------------------------------

        transaction_cost = self.execution_engine.rebalance(
            self.portfolio,
            target_position,
            stock_price,
        )

    # -------------------------------------------------
    # Portfolio value after hedge
    # -------------------------------------------------

        portfolio_value = self.portfolio.total_value(
            stock_price,
            option_price,
        )

    # -------------------------------------------------
    # Compute reward
    # -------------------------------------------------

        reward = float(
            self.reward_function(
                hedging_error=portfolio_value,
                transaction_cost=transaction_cost,
                trade_size=trade_size,
            )
        )

    # -------------------------------------------------
    # Advance environment
    # -------------------------------------------------

        self.current_step += 1

        terminated = (
            self.current_step >= self.n_steps
        )

        truncated = False

        if terminated:

            observation = np.zeros(
                self.observation_space.shape,
                dtype=np.float32,
            )

        else:

            observation = self._get_observation()

    # -------------------------------------------------
    # Extra information for evaluation
    # -------------------------------------------------

        info = {
            "portfolio_value": portfolio_value,
            "transaction_cost": transaction_cost,
            "trade_size": trade_size,
            "stock_price": stock_price,
            "strike": self.contract.strike,
            "maturity": remaining,
            "volatility": self.volatility,
            "rate": self.rate,
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )


    def _get_observation(self) -> np.ndarray:
        """
        Construct the normalized observation.
        """

        stock_price = self.path.prices[
            0,
            self.current_step,
        ]

        remaining = max(
            self.contract.maturity
            - self.current_step * self.path.dt,
            1e-8,
        )

        current_contract = OptionContract(
            strike=self.contract.strike,
            maturity=remaining,
            option_type=self.contract.option_type,
        )

        option_delta = delta(
            current_contract,
            stock_price,
            self.volatility,
            self.rate,
        )

        # -----------------------------------------
        # Normalize observations
        # -----------------------------------------

        normalized_stock = (
            stock_price
            / self.contract.strike
        )

        normalized_time = (
            remaining
            / self.contract.maturity
        )

        normalized_delta = option_delta

        normalized_shares = (
            self.portfolio.shares
            / self.max_position
        )

        initial_option_price = black_scholes_price(
            contract=self.contract,
            spot=self.path.prices[0, 0],
            volatility=self.volatility,
            rate=self.rate,
        )

        normalized_cash = (
            self.portfolio.cash
            / max(initial_option_price, 1e-8)
        )

        state = EnvironmentState(
            stock_price=normalized_stock,
            time_to_maturity=normalized_time,
            delta=normalized_delta,
            shares_held=normalized_shares,
            cash_balance=normalized_cash,
        )

        return state.to_numpy()

    def render(self):
        """
        Display the current portfolio.
        """

        index = min(
            self.current_step,
            self.n_steps - 1,
        )

        print(
            f"Step: {self.current_step:3d} | "
            f"Stock: {self.path.prices[0, index]:8.2f} | "
            f"Shares: {self.portfolio.shares:6.2f} | "
            f"Cash: {self.portfolio.cash:10.2f}"
        )