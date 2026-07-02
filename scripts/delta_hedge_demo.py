from hedging.baseline.delta_hedging import DeltaHedger

from hedging.evaluation.backtester import Backtester
from hedging.evaluation.plots import plot_summary
from hedging.evaluation.report import BacktestReport

from hedging.pricing.option import (
    OptionContract,
    OptionType,
)

from hedging.simulation.execution import ExecutionEngine
from hedging.simulation.gbm import GBMSimulator
from hedging.simulation.transaction_costs import (
    TransactionCostModel,
)


def main():

    # -------------------------------------------------
    # Market Simulator
    # -------------------------------------------------

    simulator = GBMSimulator(
        mu=0.05,
        sigma=0.20,
    )

    path = simulator.simulate(
        s0=100,
        n_steps=252,
        n_paths=1,
        seed=42,
    )

    # -------------------------------------------------
    # European Call Option
    # -------------------------------------------------

    contract = OptionContract(
        strike=100,
        maturity=1.0,
        option_type=OptionType.CALL,
    )

    # -------------------------------------------------
    # Transaction Costs
    # -------------------------------------------------

    cost_model = TransactionCostModel(
        rate=0.001,
    )

    # -------------------------------------------------
    # Execution Engine
    # -------------------------------------------------

    execution_engine = ExecutionEngine(
        cost_model=cost_model,
    )

    # -------------------------------------------------
    # Delta Hedger
    # -------------------------------------------------

    hedger = DeltaHedger()

    # -------------------------------------------------
    # Backtester
    # -------------------------------------------------

    backtester = Backtester(
        execution_engine=execution_engine,
    )

    result = backtester.run(
        path=path,
        contract=contract,
        hedger=hedger,
        volatility=0.20,
        rate=0.05,
    )

    # -------------------------------------------------
    # Print Performance Report
    # -------------------------------------------------

    BacktestReport.print(result)

    # -------------------------------------------------
    # Show Plots
    # -------------------------------------------------

    plot_summary(result)


if __name__ == "__main__":
    main()