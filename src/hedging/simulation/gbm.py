from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hedging.simulation.base import MarketSimulator
from hedging.simulation.market_path import MarketPath


@dataclass(slots=True)
class GBMSimulator(MarketSimulator):
    """
    Geometric Brownian Motion simulator.

    dS = μSdt + σSdW
    """

    mu: float
    sigma: float
    dt: float = 1 / 252

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("sigma must be positive.")

        if self.dt <= 0:
            raise ValueError("dt must be positive.")

    def simulate(
        self,
        s0: float,
        n_steps: int,
        n_paths: int = 1,
        seed: int | None = None,
    ) -> MarketPath:

        if s0 <= 0:
            raise ValueError("Initial price must be positive.")

        if n_steps <= 0:
            raise ValueError("n_steps must be positive.")

        if n_paths <= 0:
            raise ValueError("n_paths must be positive.")

        rng = np.random.default_rng(seed)

        drift = (
            self.mu - 0.5 * self.sigma**2
        ) * self.dt

        diffusion = self.sigma * np.sqrt(self.dt)

        shocks = rng.standard_normal((n_paths, n_steps))

        log_returns = drift + diffusion * shocks

        cumulative = np.cumsum(log_returns, axis=1)

        prices = np.empty((n_paths, n_steps + 1))

        prices[:, 0] = s0
        prices[:, 1:] = s0 * np.exp(cumulative)

        time_grid = np.arange(n_steps + 1) * self.dt

        return MarketPath(
            prices=prices,
            dt=self.dt,
            time_grid=time_grid,
        )