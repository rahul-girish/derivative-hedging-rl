import pytest

from hedging.environment.reward import RewardFunction


def test_reward():

    reward = RewardFunction()

    r = reward(
        hedging_error=2,
        transaction_cost=0.5,
        trade_size=10,
    )

    expected = -(
        4
        + 0.005
        + 0.01
    )

    assert r == pytest.approx(expected)