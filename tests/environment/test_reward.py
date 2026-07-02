import pytest

from hedging.environment.reward import RewardFunction

def test_reward():

    reward = RewardFunction()

    r = reward(
        hedging_error=2,
        transaction_cost=0.5,
    )

    assert r == pytest.approx(-4.025)