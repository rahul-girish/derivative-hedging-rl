from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Portfolio:
    """
    Hedging portfolio consisting of:
    - Cash account
    - Stock inventory
    - Short option liability
    """

    cash: float = 0.0
    shares: float = 0.0
    option_position: float = -1.0

    def buy(
        self,
        quantity: float,
        price: float,
        cost: float = 0.0,
    ) -> None:

        self.cash -= quantity * price
        self.cash -= cost
        self.shares += quantity

    def sell(
        self,
        quantity: float,
        price: float,
        cost: float = 0.0,
    ) -> None:

        self.cash += quantity * price
        self.cash -= cost
        self.shares -= quantity

    def initialize_option_sale(
        self,
        option_price: float,
    ) -> None:
        """
        Receive the option premium for selling one option.
        """
        self.cash += option_price

    def hedge_value(
        self,
        stock_price: float,
    ) -> float:
        """
        Value of cash + stock only.
        """
        return self.cash + self.shares * stock_price

    def total_value(
        self,
        stock_price: float,
        option_price: float,
    ) -> float:
        """
        Total marked-to-market portfolio value.
        """
        return (
            self.cash
            + self.shares * stock_price
            + self.option_position * option_price
        )