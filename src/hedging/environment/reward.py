from __future__ import annotations


class RewardFunction:
    """
    Reward function for reinforcement learning hedging.

    Encourages:
    - Low hedging error
    - Low transaction costs
    - Smooth trading behaviour
    """

    def __init__(
        self,
        hedging_weight: float = 1.0,
        transaction_cost_weight: float = 0.01,
        trade_penalty_weight: float = 0.001,
    ) -> None:

        self.hedging_weight = hedging_weight
        self.transaction_cost_weight = transaction_cost_weight
        self.trade_penalty_weight = trade_penalty_weight

    def __call__(
        self,
        hedging_error: float,
        transaction_cost: float,
        trade_size: float = 0.0,
    ) -> float:
        """
        Reward is higher when

        - hedging error is small
        - transaction costs are small
        - unnecessary trading is avoided
        """

        reward = -(
            self.hedging_weight * (hedging_error ** 2)
            + self.transaction_cost_weight * transaction_cost
            + self.trade_penalty_weight * abs(trade_size)
        )

        return float(reward)