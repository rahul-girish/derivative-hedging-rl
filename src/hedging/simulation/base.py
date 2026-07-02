from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from hedging.simulation.market_path import MarketPath


class MarketSimulator(ABC):
    """
    Abstract base class for market simulators.

    Every simulator must generate one or more price paths with a
    consistent interface.
    """

    @abstractmethod
    def simulate(
        self,
        s0: float,
        n_steps: int,
        n_paths: int = 1,
        seed: int | None = None,
    ) -> MarketPath:
        """
        Generate simulated price paths.

        Parameters
        ----------
        s0
            Initial asset price.
        n_steps
            Number of simulation time steps.
        n_paths
            Number of independent simulated paths.
        seed
            Optional random seed.

        Returns
        -------
        np.ndarray
            Shape = (n_paths, n_steps + 1)
        """
        raise NotImplementedError