from hedging.simulation.transaction_costs import TransactionCostModel


def test_transaction_cost():

    model = TransactionCostModel(rate=0.001)

    cost = model.calculate(
        shares=100,
        price=50,
    )

    assert cost == 5.0