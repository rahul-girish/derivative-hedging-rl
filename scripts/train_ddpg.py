from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor

from hedging.environment.hedging_env import HedgingEnv
from hedging.pricing.option import OptionContract, OptionType


def main():

    contract = OptionContract(
        strike=100,
        maturity=1.0,
        option_type=OptionType.CALL,
    )

    env = HedgingEnv(contract)
    env = Monitor(env)

    model = DDPG(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=100000,
        batch_size=256,
        tensorboard_log="outputs/logs/",
    )

    model.learn(total_timesteps=50_000)

    model.save("outputs/models/ddpg_hedger")


if __name__ == "__main__":
    main()