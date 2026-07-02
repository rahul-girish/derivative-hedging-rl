from hedging.simulation.execution import ExecutionEngine
from hedging.simulation.portfolio import Portfolio
from hedging.simulation.transaction_costs import TransactionCostModel

portfolio = Portfolio()
engine = ExecutionEngine(TransactionCostModel(rate=0.001))

print(f"Initial: cash={portfolio.cash}, shares={portfolio.shares}")

engine.rebalance(portfolio, target_position=50, stock_price=100)

print(f"After buying: cash={portfolio.cash:.2f}, shares={portfolio.shares}")
print(f"Portfolio value @100 = {portfolio.value(100):.2f}")

engine.rebalance(portfolio, target_position=20, stock_price=110)

print(f"After selling: cash={portfolio.cash:.2f}, shares={portfolio.shares}")
print(f"Portfolio value @110 = {portfolio.value(110):.2f}")