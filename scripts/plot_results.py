from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULT_FILE = "outputs/results/comparison.csv"
OUTPUT_DIR = Path("outputs/figures")


def plot_metric(df, column, filename, title):

    plt.figure(figsize=(7, 5))

    plt.bar(df["Agent"], df[column])

    plt.title(title)

    plt.ylabel(column)

    plt.grid(axis="y", alpha=0.3)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
    )

    plt.close()


def main():

    df = pd.read_csv(RESULT_FILE)

    plot_metric(
        df,
        "Reward",
        "reward_comparison.png",
        "Average Reward",
    )

    plot_metric(
        df,
        "Portfolio",
        "portfolio_value.png",
        "Portfolio Value",
    )

    plot_metric(
        df,
        "Transaction Cost",
        "transaction_cost.png",
        "Transaction Cost",
    )

    plot_metric(
        df,
        "Mean Hedging Error",
        "hedging_error.png",
        "Mean Hedging Error",
    )

    plot_metric(
        df,
        "RMSE",
        "rmse.png",
        "RMSE Hedging Error",
    )

    print()

    print("Graphs saved to")

    print(OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()