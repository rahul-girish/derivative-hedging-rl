from __future__ import annotations

from hedging.pricing.greeks import delta


class DeltaHedger:
    """
    Classical Black-Scholes delta hedging strategy.
    """

    def target_position(
        self,
        contract,
        stock_price: float,
        volatility: float,
        rate: float = 0.0,
        **_: object,
    ) -> float:

        return delta(
            contract=contract,
            spot=stock_price,
            volatility=volatility,
            rate=rate,
        )