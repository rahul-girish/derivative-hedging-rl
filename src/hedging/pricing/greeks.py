from __future__ import annotations

import math

from scipy.stats import norm

from hedging.pricing.black_scholes import _d1
from hedging.pricing.option import OptionContract, OptionType


def delta(
    contract: OptionContract,
    spot: float,
    volatility: float,
    rate: float = 0.0,
    dividend: float = 0.0,
) -> float:
    """
    Computes Black-Scholes delta.
    """

    d1 = _d1(
        spot,
        contract.strike,
        contract.maturity,
        rate,
        volatility,
        dividend,
    )

    if contract.option_type == OptionType.CALL:
        return math.exp(-dividend * contract.maturity) * norm.cdf(d1)

    return math.exp(-dividend * contract.maturity) * (
        norm.cdf(d1) - 1.0
    )


def gamma(
    contract: OptionContract,
    spot: float,
    volatility: float,
    rate: float = 0.0,
    dividend: float = 0.0,
) -> float:
    """
    Computes Black-Scholes gamma.
    """

    d1 = _d1(
        spot,
        contract.strike,
        contract.maturity,
        rate,
        volatility,
        dividend,
    )

    return (
        math.exp(-dividend * contract.maturity)
        * norm.pdf(d1)
        / (spot * volatility * math.sqrt(contract.maturity))
    )