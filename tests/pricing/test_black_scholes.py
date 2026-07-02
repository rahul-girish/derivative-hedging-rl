from hedging.pricing.black_scholes import black_scholes_price
from hedging.pricing.option import OptionContract, OptionType


def test_call_price():
    option = OptionContract(
        strike=100,
        maturity=1.0,
        option_type=OptionType.CALL,
    )

    price = black_scholes_price(
        contract=option,
        spot=100,
        volatility=0.2,
        rate=0.05,
    )

    assert abs(price - 10.4506) < 1e-3


def test_put_price():
    option = OptionContract(
        strike=100,
        maturity=1.0,
        option_type=OptionType.PUT,
    )

    price = black_scholes_price(
        contract=option,
        spot=100,
        volatility=0.2,
        rate=0.05,
    )

    assert abs(price - 5.5735) < 1e-3