from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TransactionCostModel:
    """
    Linear proportional transaction cost model.

    Cost = rate × |shares traded| × stock price
    """

    rate: float = 0.001

    def __post_init__(self) -> None:
        if self.rate < 0:
            raise ValueError("Transaction cost rate cannot be negative.")

    def calculate(
        self,
        shares: float,
        price: float,
    ) -> float:
        """
        Calculate the transaction cost of a trade.
        """
        if price <= 0:
            raise ValueError("Price must be positive.")

        return abs(shares) * price * self.rate