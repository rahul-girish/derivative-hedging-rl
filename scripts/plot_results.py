from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULT_FILE = "outputs/results/comparison.csv"
OUTPUT_DIR = Path("outputs/figures")

# Professional colors
COLORS = [
    "#4C78A8",  # Random
    "#F58518",  # Delta
    "#54A24B",  # DDPG
]


def format_value(value: float) -> str:
    """
    Format numbers nicely.
    """

    if abs(value) >= 1000:
        return f"{value:,.0f}"

    return f"{value:.2f}"


def draw_plot(
    ax,
    df,
    column,
    title,
    higher_is_better,
):
    """
    Draw one publication-quality bar chart.
    """

    values = df[column]

    bars = ax.bar(
        df["Agent"],
        values,
        color=COLORS,
        edgecolor="black",
        linewidth=1.2,
        width=0.8,
    )

    # Highlight best performer
    if higher_is_better:
        best = values.idxmax()
    else:
        best = values.idxmin()

    bars[best].set_edgecolor("darkgreen")
    bars[best].set_linewidth(3)

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        pad=18,
    )

    ax.set_xlabel(
        "Hedging Strategy",
        fontsize=12,
    )

    ax.set_ylabel(
        column,
        fontsize=12,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.25,
    )

    ymin = values.min()
    ymax = values.max()
    yrange = ymax - ymin

    if yrange == 0:
        yrange = 1

    # Add extra space so labels never overlap
    ax.set_ylim(
        ymin - 0.18 * yrange,
        ymax + 0.18 * yrange,
    )

    offset = 0.05 * yrange

    for i, (bar, value) in enumerate(zip(bars, values)):

        color = "darkgreen" if i == best else "black"

        if value >= 0:
            y = value + offset
            va = "bottom"
        else:
            y = value - offset
            va = "top"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            format_value(value),
            ha="center",
            va=va,
            fontsize=11,
            fontweight="bold",
            color=color,
        )


def save_single_plot(
    df,
    column,
    filename,
    title,
    higher_is_better,
):

    fig, ax = plt.subplots(
        figsize=(8.5, 6.2),
    )

    draw_plot(
        ax,
        df,
        column,
        title,
        higher_is_better,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_summary(df):

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16, 9),
    )

    plots = [

        (
            "Reward",
            "Average Reward",
            True,
        ),

        (
            "Portfolio",
            "Portfolio Value",
            True,
        ),

        (
            "Transaction Cost",
            "Transaction Cost",
            False,
        ),

        (
            "Mean Hedging Error",
            "Mean Hedging Error",
            False,
        ),

        (
            "RMSE",
            "RMSE Hedging Error",
            False,
        ),

    ]

    axes = axes.flatten()

    for ax, (column, title, hib) in zip(
        axes,
        plots,
    ):

        draw_plot(
            ax,
            df,
            column,
            title,
            hib,
        )

    axes[-1].axis("off")

    fig.suptitle(
        "Comparison of Hedging Strategies",
        fontsize=20,
        fontweight="bold",
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "summary.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        RESULT_FILE,
    )

    save_single_plot(
        df,
        "Reward",
        "reward_comparison.png",
        "Average Reward",
        True,
    )

    save_single_plot(
        df,
        "Portfolio",
        "portfolio_value.png",
        "Portfolio Value",
        True,
    )

    save_single_plot(
        df,
        "Transaction Cost",
        "transaction_cost.png",
        "Transaction Cost",
        False,
    )

    save_single_plot(
        df,
        "Mean Hedging Error",
        "hedging_error.png",
        "Mean Hedging Error",
        False,
    )

    save_single_plot(
        df,
        "RMSE",
        "rmse.png",
        "RMSE Hedging Error",
        False,
    )

    save_summary(df)

    print()
    print("Graphs saved to:")
    print(OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()