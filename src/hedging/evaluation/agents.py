from __future__ import annotations

import numpy as np
from stable_baselines3 import DDPG

from hedging.baseline.delta_hedging import DeltaHedger
from hedging.pricing.option import OptionContract, OptionType


class RandomAgent:
    """
    Random baseline.
    """

    def predict(self, observation, info,):

        return np.array(
            [
                np.random.uniform(-1.0, 1.0)
            ],
            dtype=np.float32,
        )


class DeltaAgent:
    """
    Classical Black-Scholes delta hedging agent.
    """

    def __init__(self):

        self.hedger = DeltaHedger()

    def predict(
        self,
        observation,
        info,
    ):

        stock_price = observation[0] * info["strike"]

        contract = OptionContract(
            strike=info["strike"],
            maturity=max(info["maturity"], 1e-6),
            option_type=OptionType.CALL,
        )

        target_delta = self.hedger.target_position(
            contract=contract,
            stock_price=stock_price,
            volatility=info["volatility"],
            rate=info["rate"],
        )

        return np.array(
            [target_delta],
            dtype=np.float32,
        )

        
class DDPGAgent:
    """
    Stable-Baselines3 wrapper.
    """

    def __init__(
        self,
        model_path: str,
    ):

        self.model = DDPG.load(model_path)

    def predict(
        self,
        observation,
        info,
    ):

        action, _ = self.model.predict(
            observation,
            deterministic=True,
        )

        return action