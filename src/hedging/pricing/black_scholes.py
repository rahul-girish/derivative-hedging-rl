from __future__ import annotations

import math

from scipy.stats import norm

from hedging.pricing.option import OptionContract, OptionType


def _d1(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend: float = 0.0,
) -> float:
    return (
        math.log(spot / strike)
        + (rate - dividend + 0.5 * volatility**2) * maturity
    ) / (volatility * math.sqrt(maturity))


def _d2(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend: float = 0.0,
) -> float:
    return _d1(
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend,
    ) - volatility * math.sqrt(maturity)


def black_scholes_price(
    contract: OptionContract,
    spot: float,
    volatility: float,
    rate: float = 0.0,
    dividend: float = 0.0,
) -> float:
    """
    Computes the Black-Scholes price of a European option.
    """

    if spot <= 0:
        raise ValueError("Spot price must be positive.")

    if volatility <= 0:
        raise ValueError("Volatility must be positive.")

    d1 = _d1(
        spot,
        contract.strike,
        contract.maturity,
        rate,
        volatility,
        dividend,
    )

    d2 = _d2(
        spot,
        contract.strike,
        contract.maturity,
        rate,
        volatility,
        dividend,
    )

    if contract.option_type == OptionType.CALL:
        return (
            spot
            * math.exp(-dividend * contract.maturity)
            * norm.cdf(d1)
            - contract.strike
            * math.exp(-rate * contract.maturity)
            * norm.cdf(d2)
        )

    return (
        contract.strike
        * math.exp(-rate * contract.maturity)
        * norm.cdf(-d2)
        - spot
        * math.exp(-dividend * contract.maturity)
        * norm.cdf(-d1)
    )