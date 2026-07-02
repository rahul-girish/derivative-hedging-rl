import numpy as np

from hedging.environment.hedging_env import HedgingEnv
from hedging.pricing.option import OptionContract, OptionType


def test_environment():

    contract = OptionContract(
        strike=100,
        maturity=1.0,
        option_type=OptionType.CALL,
    )

    env = HedgingEnv(contract)

    obs, info = env.reset(seed=42)

    assert isinstance(obs, np.ndarray)

    assert obs.shape == (5,)

    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(obs, np.ndarray)

    assert isinstance(reward, float)