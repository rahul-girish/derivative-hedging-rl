"""
Compares three hedging strategies on REAL historical price data (Yahoo
Finance), under a synthetic option overlay priced with the project's
own Black-Scholes engine (same approach as delta_hedge_demo.py, but
driven by real market dynamics instead of a GBM simulation).

Strategies:
    1. Delta Hedging          - classical BS delta, rebalance every step.
    2. Threshold Hedging      - rebalance only past a drift threshold.
    3. Contextual Bandit      - linear ridge-regression bandit that
                                 learns a correction to the BS delta,
                                 trained online across historical
                                 episodes, then evaluated on a held-out
                                 period it has not traded on.

Nothing here is scripted or hand-tuned to produce a particular
outcome: the bandit's weights come from ridge-regression updates on
realized reward (see agents/bandit/bandit_hedger.py). Numbers printed
below are whatever the model actually produces on the held-out data.
"""

from __future__ import annotations

import numpy as np

from hedging.agents.bandit.bandit_hedger import ContextualBanditHedger
from hedging.baseline.delta_hedging import DeltaHedger
from hedging.baseline.threshold import ThresholdHedger
from hedging.data.loader import load_yahoo_prices
from hedging.evaluation.backtester import Backtester
from hedging.pricing.option import OptionContract, OptionType
from hedging.simulation.execution import ExecutionEngine
from hedging.simulation.market_path import MarketPath
from hedging.simulation.transaction_costs import TransactionCostModel

TICKER = "SPY"
TRAIN_START, TRAIN_END = "2015-01-01", "2023-01-01"
TEST_START, TEST_END = "2023-01-01", "2024-06-01"
EPISODE_LEN = 252
COST_RATE = 0.005
RATE = 0.03
RISK_AVERSION = 0.05


def make_engine() -> ExecutionEngine:
    return ExecutionEngine(TransactionCostModel(rate=COST_RATE))


def split_episodes(
    path: MarketPath, episode_len: int, stride: int | None = None
) -> list[MarketPath]:
    """
    Slice a long price history into episode_len-step episodes.
    stride == episode_len gives non-overlapping episodes (used for
    training); a smaller stride gives overlapping episodes so a short
    held-out period still yields enough samples for a std-dev estimate.
    """
    prices = path.prices[0]
    step = stride or episode_len
    episodes = []
    for start in range(0, len(prices) - episode_len, step):
        chunk = prices[start : start + episode_len + 1]
        if len(chunk) < episode_len + 1:
            continue
        time_grid = np.arange(len(chunk)) * path.dt
        episodes.append(
            MarketPath(prices=chunk.reshape(1, -1), dt=path.dt, time_grid=time_grid)
        )
    return episodes


def annualized_volatility(path: MarketPath) -> float:
    log_returns = np.diff(np.log(path.prices[0]))
    return float(np.std(log_returns) / np.sqrt(path.dt))


def run_episode(path: MarketPath, hedger, volatility: float, rate: float):
    contract = OptionContract(
        strike=round(float(path.prices[0, 0])),
        maturity=path.dt * (path.prices.shape[1] - 1),
        option_type=OptionType.CALL,
    )
    backtester = Backtester(make_engine())
    return backtester.run(
        path=path,
        contract=contract,
        hedger=hedger,
        volatility=volatility,
        rate=rate,
    )


def summarize(name: str, results) -> None:
    pnls = [r.pnl for r in results]
    costs = [sum(r.transaction_costs) for r in results]
    print(
        f"{name:28s} mean PnL {np.mean(pnls):8.3f}   "
        f"std PnL {np.std(pnls):7.3f}   "
        f"mean cost {np.mean(costs):7.3f}"
    )


def main() -> None:
    print(f"Downloading real {TICKER} price history from Yahoo Finance...")
    train_path = load_yahoo_prices(TICKER, TRAIN_START, TRAIN_END)
    test_path = load_yahoo_prices(TICKER, TEST_START, TEST_END)

    train_episodes = split_episodes(train_path, EPISODE_LEN, stride=42)
    test_episodes = split_episodes(test_path, EPISODE_LEN, stride=21)

    if not test_episodes:
        raise RuntimeError("Not enough held-out data for a full episode.")

    volatility = annualized_volatility(train_path)

    print(
        f"Training bandit hedger online across {len(train_episodes)} "
        f"overlapping historical {EPISODE_LEN}-day episodes "
        f"({TRAIN_START} to {TRAIN_END})..."
    )
    bandit = ContextualBanditHedger(risk_aversion=RISK_AVERSION, seed=7)
    bandit.train()
    for episode in train_episodes:
        run_episode(episode, bandit, volatility, RATE)

    bandit.eval()
    print(
        f"Evaluating on {len(test_episodes)} held-out episode(s) "
        f"({TEST_START} to {TEST_END}, never used for training)...\n"
    )

    delta_results = [
        run_episode(ep, DeltaHedger(), volatility, RATE) for ep in test_episodes
    ]
    threshold_results = [
        run_episode(ep, ThresholdHedger(threshold=0.05), volatility, RATE)
        for ep in test_episodes
    ]
    bandit_results = [
        run_episode(ep, bandit, volatility, RATE) for ep in test_episodes
    ]

    print("=" * 78)
    print(
        f"HELD-OUT COMPARISON  ticker={TICKER}  cost_rate={COST_RATE}  "
        f"episodes={len(test_episodes)}  implied_vol={volatility:.3f}"
    )
    print("=" * 78)
    summarize("Delta Hedging (baseline)", delta_results)
    summarize("Threshold Hedging", threshold_results)
    summarize("Contextual Bandit (adaptive)", bandit_results)


if __name__ == "__main__":
    main()
