from stable_baselines3 import DDPG

from hedging.environment.hedging_env import HedgingEnv
from hedging.pricing.option import OptionContract, OptionType


contract = OptionContract(
    strike=100,
    maturity=1.0,
    option_type=OptionType.CALL,
)

env = HedgingEnv(contract)

model = DDPG.load("outputs/models/ddpg_hedger")

obs, info = env.reset(seed=42)

done = False

total_reward = 0.0

while not done:

    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward

    done = terminated or truncated

print(f"Total Reward : {total_reward:.4f}")