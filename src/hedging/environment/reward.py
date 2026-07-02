from __future__ import annotations


class RewardFunction:
    """
    Computes the reinforcement learning reward.
    """

    def __init__(
        self,
        transaction_cost_weight: float = 1.0,
    ) -> None:

        self.transaction_cost_weight = transaction_cost_weight

    def __call__(
        self,
        hedging_error: float,
        transaction_cost: float,
    ) -> float:
        """
        Reward is higher when
        - hedging error is small
        - transaction cost is small
        """

        return -(
            hedging_error**2
            + 0.05 * transaction_cost
        )