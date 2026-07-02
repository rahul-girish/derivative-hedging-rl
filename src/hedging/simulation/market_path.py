from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass(slots=True)
class MarketPath:
    prices: np.ndarray
    dt: float
    time_grid: np.ndarray