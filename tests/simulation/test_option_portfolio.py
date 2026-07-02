from hedging.simulation.portfolio import Portfolio


def test_option_sale():

    p = Portfolio()

    p.initialize_option_sale(10)

    assert p.cash == 10


def test_total_value():

    p = Portfolio()

    p.initialize_option_sale(10)

    p.buy(
        quantity=1,
        price=100,
    )

    value = p.total_value(
        stock_price=100,
        option_price=10,
    )

    assert value == 0