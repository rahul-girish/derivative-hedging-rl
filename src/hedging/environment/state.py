from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class EnvironmentState:
    """
    Represents the observation provided to the RL agent.
    """

    stock_price: float
    time_to_maturity: float
    delta: float
    shares_held: float
    cash_balance: float

    def to_numpy(self) -> np.ndarray:
        """
        Convert the state into a NumPy observation vector.
        """

        return np.array(
            [
                self.stock_price,
                self.time_to_maturity,
                self.delta,
                self.shares_held,
                self.cash_balance,
            ],
            dtype=np.float32,
        )