from __future__ import annotations

import numpy as np
from torch import nn

from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from hedging.environment.hedging_env import HedgingEnv
from hedging.pricing.option import (
    OptionContract,
    OptionType,
)


def main():

    # -------------------------------------------------
    # Option Contract
    # -------------------------------------------------

    contract = OptionContract(
        strike=100,
        maturity=1.0,
        option_type=OptionType.CALL,
    )

    env = HedgingEnv(contract)
    env = Monitor(env)

    # -------------------------------------------------
    # Exploration Noise
    # -------------------------------------------------

    n_actions = env.action_space.shape[-1]

    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.10 * np.ones(n_actions),
    )

    # -------------------------------------------------
    # Neural Network Architecture
    # -------------------------------------------------

    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=dict(
            pi=[256, 256],
            qf=[256, 256],
        ),
    )

    # -------------------------------------------------
    # DDPG Model
    # -------------------------------------------------

    model = DDPG(
        policy="MlpPolicy",
        env=env,

        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=256,

        gamma=0.99,
        tau=0.005,

        action_noise=action_noise,

        policy_kwargs=policy_kwargs,

        tensorboard_log="outputs/logs/ddpg_v2",

        verbose=1,
    )

    # -------------------------------------------------
    # Train
    # -------------------------------------------------

    model.learn(
        total_timesteps=500_000,
    )

    # -------------------------------------------------
    # Save Model
    # -------------------------------------------------

    model.save(
        "outputs/models/ddpg_hedger_v2"
    )


if __name__ == "__main__":
    main()