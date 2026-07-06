from __future__ import annotations

import csv
from pathlib import Path

from hedging.environment.hedging_env import HedgingEnv
from hedging.evaluation.agents import (
    RandomAgent,
    DeltaAgent,
    DDPGAgent,
)
from hedging.evaluation.evaluator import Evaluator
from hedging.pricing.option import (
    OptionContract,
    OptionType,
)


def evaluate_agent(
    name: str,
    agent,
    evaluator: Evaluator,
):
    """
    Evaluate one agent and print its statistics.
    """

    result = evaluator.evaluate(
        agent=agent,
        episodes=50,
        seed=42,
    )

    print(
        f"{name:<12}"
        f"{result.total_reward:>15.2f}"
        f"{result.final_portfolio_value:>15.2f}"
        f"{result.total_transaction_cost:>12.2f}"
        f"{result.mean_hedging_error:>12.2f}"
        f"{result.rmse_hedging_error:>12.2f}"
    )

    return result


def save_results(
    random_result,
    delta_result,
    ddpg_result,
):
    """
    Save evaluation results to CSV.
    """

    output_dir = Path("outputs/results")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = output_dir / "comparison.csv"

    with open(
        csv_path,
        "w",
        newline="",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "Agent",
                "Reward",
                "Portfolio",
                "Transaction Cost",
                "Mean Hedging Error",
                "RMSE",
                "Std Dev",
                "Max Error",
            ]
        )

        writer.writerow(
            [
                "Random",
                random_result.total_reward,
                random_result.final_portfolio_value,
                random_result.total_transaction_cost,
                random_result.mean_hedging_error,
                random_result.rmse_hedging_error,
                random_result.std_hedging_error,
                random_result.max_hedging_error,
            ]
        )

        writer.writerow(
            [
                "Delta",
                delta_result.total_reward,
                delta_result.final_portfolio_value,
                delta_result.total_transaction_cost,
                delta_result.mean_hedging_error,
                delta_result.rmse_hedging_error,
                delta_result.std_hedging_error,
                delta_result.max_hedging_error,
            ]
        )

        writer.writerow(
            [
                "DDPG",
                ddpg_result.total_reward,
                ddpg_result.final_portfolio_value,
                ddpg_result.total_transaction_cost,
                ddpg_result.mean_hedging_error,
                ddpg_result.rmse_hedging_error,
                ddpg_result.std_hedging_error,
                ddpg_result.max_hedging_error,
            ]
        )

    print(f"\nResults saved to: {csv_path}")


def main():

    contract = OptionContract(
        strike=100,
        maturity=1.0,
        option_type=OptionType.CALL,
    )

    env = HedgingEnv(contract)

    evaluator = Evaluator(env)

    print()
    print("=" * 100)
    print(
        f"{'Agent':<12}"
        f"{'Reward':>15}"
        f"{'Portfolio':>15}"
        f"{'Cost':>12}"
        f"{'MeanErr':>12}"
        f"{'RMSE':>12}"
    )
    print("=" * 100)

    random_result = evaluate_agent(
        "Random",
        RandomAgent(),
        evaluator,
    )

    delta_result = evaluate_agent(
        "Delta",
        DeltaAgent(),
        evaluator,
    )

    ddpg_result = evaluate_agent(
        "DDPG",
        DDPGAgent(
            "outputs/models/ddpg_hedger_v2",
        ),
        evaluator,
    )

    print("=" * 100)

    save_results(
        random_result,
        delta_result,
        ddpg_result,
    )


if __name__ == "__main__":
    main()