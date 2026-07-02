from hedging.simulation.portfolio import Portfolio


def test_buy():

    p = Portfolio()

    p.buy(
        quantity=10,
        price=100,
    )

    assert p.shares == 10
    assert p.cash == -1000


def test_sell():

    p = Portfolio()

    p.buy(10, 100)
    p.sell(5, 110)

    assert p.shares == 5
    assert p.cash == -450


def test_value():

    p = Portfolio()

    p.buy(10, 100)

    assert p.total_value(120, 0.0) == 200