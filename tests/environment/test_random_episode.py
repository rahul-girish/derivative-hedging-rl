from hedging.environment.hedging_env import HedgingEnv
from hedging.pricing.option import OptionContract, OptionType


def test_random_episode():

    contract = OptionContract(
        strike=100,
        maturity=1.0,
        option_type=OptionType.CALL,
    )

    env = HedgingEnv(contract)

    obs, info = env.reset(seed=42)

    done = False

    while not done:

        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

    assert done