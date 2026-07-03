from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class EvaluationResult:

    episodes: int

    total_reward: float

    final_portfolio_value: float

    total_transaction_cost: float

    mean_hedging_error: float

    rmse_hedging_error: float

    std_hedging_error: float

    max_hedging_error: float


class Evaluator:
    """
    Generic evaluator for any hedging strategy.
    """

    def __init__(self, env):

        self.env = env

    def evaluate(
        self,
        agent: Any,
        episodes: int = 10,
        seed: int = 42,
    ) -> EvaluationResult:

        rewards = []
        portfolio_values = []
        transaction_costs = []
        hedging_errors = []

        for episode in range(episodes):

            obs, info = self.env.reset(
                seed=seed + episode
            )

            terminated = False
            truncated = False

            episode_reward = 0.0
            final_portfolio = 0.0
            total_cost = 0.0

            while not (terminated or truncated):

                action = agent.predict(
                    obs,
                    info,
                )

                (
                    obs,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = self.env.step(action)

                episode_reward += reward

                final_portfolio = info.get(
                    "portfolio_value",
                    final_portfolio,
                )

                cost = info.get(
                    "transaction_cost",
                    0.0,
                )

                total_cost += cost

                # Using portfolio value as the current
                # hedging-error proxy.
                hedging_errors.append(
                    abs(final_portfolio)
                )

            rewards.append(
                episode_reward
            )

            portfolio_values.append(
                final_portfolio
            )

            transaction_costs.append(
                total_cost
            )

        # -------------------------------------
        # Episode averages
        # -------------------------------------

        average_reward = float(
            np.mean(rewards)
        )

        average_portfolio = float(
            np.mean(portfolio_values)
        )

        average_cost = float(
            np.mean(transaction_costs)
        )

        # -------------------------------------
        # Hedging-error statistics
        # -------------------------------------

        hedging_errors = np.asarray(
            hedging_errors,
            dtype=float,
        )

        mean_error = float(
            np.mean(hedging_errors)
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    hedging_errors ** 2
                )
            )
        )

        std_error = float(
            np.std(hedging_errors)
        )

        max_error = float(
            np.max(hedging_errors)
        )

        return EvaluationResult(
            episodes=episodes,
            total_reward=average_reward,
            final_portfolio_value=average_portfolio,
            total_transaction_cost=average_cost,
            mean_hedging_error=mean_error,
            rmse_hedging_error=rmse,
            std_hedging_error=std_error,
            max_hedging_error=max_error,
        )