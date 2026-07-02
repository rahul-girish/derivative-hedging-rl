from hedging.baseline.delta_hedging import DeltaHedger
from hedging.evaluation.backtester import Backtester
from hedging.pricing.option import OptionContract, OptionType
from hedging.simulation.execution import ExecutionEngine
from hedging.simulation.gbm import GBMSimulator
from hedging.simulation.transaction_costs import TransactionCostModel

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

contract = OptionContract(
    strike=100,
    maturity=1.0,
    option_type=OptionType.CALL,
)

engine = ExecutionEngine(
    TransactionCostModel(rate=0.001),
)

backtester = Backtester(engine)

hedger = DeltaHedger()

result = backtester.run(
    path=path,
    contract=contract,
    hedger=hedger,
    volatility=0.20,
    rate=0.05,
)

print("=" * 50)
print("DELTA HEDGING BACKTEST")
print("=" * 50)

print(f"Final PnL            : {result.pnl:.4f}")
print(f"Final Portfolio Value: {result.portfolio_values[-1]:.4f}")
print(f"Final Shares Held    : {result.share_history[-1]:.4f}")
print(f"Total Transaction Cost: {sum(result.transaction_costs):.4f}")