from hedging.environment.hedging_env import HedgingEnv
from hedging.evaluation.agents import DDPGAgent
from hedging.evaluation.evaluator import Evaluator
from hedging.pricing.option import OptionContract, OptionType


def main():

    contract = OptionContract(
        strike=100,
        maturity=1.0,
        option_type=OptionType.CALL,
    )

    env = HedgingEnv(contract)

    evaluator = Evaluator(env)

    agent = DDPGAgent(
        "outputs/models/ddpg_hedger.zip"
    )

    results = evaluator.evaluate(
        agent,
        episodes=50,
        seed=42,
    )

    print("\n===== DDPG Evaluation =====")
    print(f"Episodes                 : {results.episodes}")
    print(f"Average Reward           : {results.total_reward:.2f}")
    print(f"Average Portfolio Value  : {results.final_portfolio_value:.2f}")
    print(f"Average Transaction Cost : {results.total_transaction_cost:.2f}")


if __name__ == "__main__":
    main()