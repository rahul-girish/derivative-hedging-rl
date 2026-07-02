from __future__ import annotations

import numpy as np


def total_return(values: list[float]) -> float:
    return values[-1] - values[0]


def max_drawdown(values: list[float]) -> float:

    values = np.asarray(values)

    running_max = np.maximum.accumulate(values)

    drawdown = values - running_max

    return float(drawdown.min())


def realized_volatility(values: list[float]) -> float:

    values = np.asarray(values)

    returns = np.diff(values)

    return float(np.std(returns))


def total_transaction_cost(costs: list[float]) -> float:
    return float(np.sum(costs))


def mean_abs_hedging_error(errors: list[float]) -> float:
    return float(np.mean(np.abs(errors)))


def rmse(errors: list[float]) -> float:
    return float(np.sqrt(np.mean(np.square(errors))))