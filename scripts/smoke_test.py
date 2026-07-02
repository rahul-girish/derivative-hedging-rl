from hedging.simulation.gbm import GBMSimulator

from hedging.pricing.option import OptionContract, OptionType

from hedging.pricing.black_scholes import black_scholes_price

from hedging.pricing.greeks import delta

sim = GBMSimulator(mu=0.05, sigma=0.20)

paths = sim.simulate(s0=100, n_steps=252, n_paths=3, seed=42)

option = OptionContract(

strike=100,

maturity=1.0,

option_type=OptionType.CALL,

)

spot = paths.prices[0, 0]

price = black_scholes_price(option, spot, 0.20, 0.05)

d = delta(option, spot, 0.20, 0.05)

print("Path shape:", paths.prices.shape)

print("First path first 5 prices:", paths.prices[0, :5])

print("Option price:", round(price, 4))

print("Delta:", round(d, 4))