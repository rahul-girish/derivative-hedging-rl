from hedging.pricing.greeks import delta, gamma
from hedging.pricing.option import OptionContract, OptionType


def test_delta():
    option = OptionContract(
        strike=100,
        maturity=1.0,
        option_type=OptionType.CALL,
    )

    d = delta(
        option,
        spot=100,
        volatility=0.2,
        rate=0.05,
    )

    assert abs(d - 0.6368) < 1e-3


def test_gamma():
    option = OptionContract(
        strike=100,
        maturity=1.0,
        option_type=OptionType.CALL,
    )

    g = gamma(
        option,
        spot=100,
        volatility=0.2,
        rate=0.05,
    )

    assert abs(g - 0.01876) < 1e-4