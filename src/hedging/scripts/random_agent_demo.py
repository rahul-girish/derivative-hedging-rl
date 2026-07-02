from hedging.environment.hedging_env import HedgingEnv
from hedging.pricing.option import OptionContract, OptionType


def main():

    contract = OptionContract(
        strike=100,
        maturity=1.0,
        option_type=OptionType.CALL,
    )

    env = HedgingEnv(contract)

    observation, info = env.reset(seed=42)

    done = False

    total_reward = 0.0

    step = 0

    while not done:

        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        step += 1

        done = terminated or truncated

    print("=" * 50)
    print("Random Agent Demo")
    print("=" * 50)
    print(f"Episode Length : {step}")
    print(f"Total Reward   : {total_reward:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()