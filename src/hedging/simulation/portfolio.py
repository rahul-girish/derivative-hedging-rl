from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Portfolio:
    """
    Represents the hedging portfolio.
    """

    cash: float = 0.0
    shares: float = 0.0

    def buy(
        self,
        quantity: float,
        price: float,
        cost: float = 0.0,
    ) -> None:
        """
        Buy shares.
        """
        self.cash -= quantity * price
        self.cash -= cost
        self.shares += quantity

    def sell(
        self,
        quantity: float,
        price: float,
        cost: float = 0.0,
    ) -> None:
        """
        Sell shares.
        """
        self.cash += quantity * price
        self.cash -= cost
        self.shares -= quantity

    def value(
        self,
        stock_price: float,
    ) -> float:
        """
        Current portfolio value.
        """
        return self.cash + self.shares * stock_price